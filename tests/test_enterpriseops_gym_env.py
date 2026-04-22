"""Tests for the Prime Intellect Verifiers environment adapter.

These tests run without Docker or API keys by mocking the MCP and verification layers.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from verifiers.types import AssistantMessage, State, ToolCall, ToolMessage as VFToolMessage, TrajectoryStep

from enterpriseops_gym_env import (
    ALL_DOMAINS,
    DEFAULT_SERVER_URLS,
    EOpsGymEnv,
    _build_dataset,
    _build_rubric,
    _collect_tool_calls,
    _mcp_content_to_str,
    _noop,
    _parse_info,
    _resolve_sql_path,
)


# -- Helpers ------------------------------------------------------------------


class TestParseInfo:
    def test_json_string(self):
        result = _parse_info('{"key": "value"}')
        assert result == {"key": "value"}

    def test_dict_passthrough(self):
        d = {"key": "value"}
        assert _parse_info(d) is d

    def test_nested_json(self):
        data = {"verifiers": [{"type": "database_state"}], "selected_tools": ["a", "b"]}
        result = _parse_info(json.dumps(data))
        assert result == data


class TestResolveSqlPath:
    def test_absolute_path(self, tmp_path):
        sql = tmp_path / "test.sql"
        sql.write_text("CREATE TABLE t;")
        assert _resolve_sql_path(str(sql), "/nonexistent") == str(sql)

    def test_relative_to_gym_dbs(self, tmp_path):
        sql = tmp_path / "domain" / "db.sql"
        sql.parent.mkdir()
        sql.write_text("CREATE TABLE t;")
        assert _resolve_sql_path("domain/db.sql", str(tmp_path)) == str(sql)

    def test_basename_fallback(self, tmp_path):
        sql = tmp_path / "db.sql"
        sql.write_text("CREATE TABLE t;")
        assert _resolve_sql_path("some/nested/path/db.sql", str(tmp_path)) == str(sql)

    def test_not_found(self):
        assert _resolve_sql_path("nonexistent.sql", "/nonexistent") is None


class TestMcpContentToStr:
    def test_string_passthrough(self):
        assert _mcp_content_to_str("hello") == "hello"

    def test_text_content_list(self):
        result = {"content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}
        assert _mcp_content_to_str(result) == "line1\nline2"

    def test_empty_dict(self):
        assert _mcp_content_to_str({}) == "{}"

    def test_non_text_content(self):
        result = {"content": [{"type": "image", "url": "http://example.com"}]}
        assert "image" in _mcp_content_to_str(result)

    def test_other_type(self):
        assert _mcp_content_to_str(42) == "42"
        assert _mcp_content_to_str(None) == "None"


# -- _collect_tool_calls ------------------------------------------------------


class TestCollectToolCalls:
    def _make_state(self, steps: list) -> State:
        return State(trajectory=steps, database_ids={}, server_url_map={})

    def test_extracts_tool_names(self):
        step = TrajectoryStep(
            completion=[
                AssistantMessage(role="assistant", content=None, tool_calls=[
                    ToolCall(id="tc1", name="list_users", arguments="{}"),
                    ToolCall(id="tc2", name="create_chat", arguments="{}"),
                ]),
                VFToolMessage(role="tool", content="result1", tool_call_id="tc1"),
                VFToolMessage(role="tool", content="result2", tool_call_id="tc2"),
            ],
            prompt=[], response=None, tokens=None,
            reward=None, advantage=None, is_truncated=False, trajectory_id="t1", extras={},
        )
        assert _collect_tool_calls(self._make_state([step])) == ["list_users", "create_chat"]

    def test_skips_non_assistant_messages(self):
        step = TrajectoryStep(
            completion=[VFToolMessage(role="tool", content="result", tool_call_id="tc1")],
            prompt=[], response=None, tokens=None,
            reward=None, advantage=None, is_truncated=False, trajectory_id="t1", extras={},
        )
        assert _collect_tool_calls(self._make_state([step])) == []

    def test_handles_none_tool_calls(self):
        step = TrajectoryStep(
            completion=[AssistantMessage(role="assistant", content="done", tool_calls=None)],
            prompt=[], response=None, tokens=None,
            reward=None, advantage=None, is_truncated=False, trajectory_id="t1", extras={},
        )
        assert _collect_tool_calls(self._make_state([step])) == []

    def test_multiple_steps(self):
        make_step = lambda names: TrajectoryStep(
            completion=[AssistantMessage(
                role="assistant", content=None,
                tool_calls=[ToolCall(id=f"tc_{n}", name=n, arguments="{}") for n in names],
            )],
            prompt=[], response=None, tokens=None,
            reward=None, advantage=None, is_truncated=False, trajectory_id="t1", extras={},
        )
        state = self._make_state([make_step(["a", "b"]), make_step(["c"])])
        assert _collect_tool_calls(state) == ["a", "b", "c"]

    def test_empty_trajectory(self):
        assert _collect_tool_calls(self._make_state([])) == []


# -- EOpsGymEnv ---------------------------------------------------------------


class TestEOpsGymEnvInit:
    @patch("enterpriseops_gym_env.MCPClient")
    def test_max_workers_forced_to_one(self, mock_mcp_cls):
        mock_mcp_cls.return_value = AsyncMock()
        mock_mcp_cls.return_value.connect.return_value = False
        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={},
            gym_dbs_path="gym_dbs",
            max_workers=4,
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )
        assert env.max_workers == 1

    @patch("enterpriseops_gym_env.MCPClient")
    def test_connects_and_discovers_tools(self, mock_mcp_cls):
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.list_tools.return_value = [
            {"name": "tool_a", "description": "A", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "tool_b", "description": "B"},
        ]
        mock_mcp_cls.return_value = mock_client

        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={"server1": "http://localhost:9999"},
            gym_dbs_path="gym_dbs",
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )

        assert len(env._all_tool_defs) == 2
        assert env._all_tool_defs[0].name == "tool_a"
        assert env._all_tool_defs[1].name == "tool_b"
        assert env.tool_to_server == {"tool_a": "server1", "tool_b": "server1"}
        assert "server1" in env.clients

    @patch("enterpriseops_gym_env.MCPClient")
    def test_skips_unreachable_servers(self, mock_mcp_cls):
        mock_client = AsyncMock()
        mock_client.connect.return_value = False
        mock_mcp_cls.return_value = mock_client

        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={"dead_server": "http://localhost:9999"},
            gym_dbs_path="gym_dbs",
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )

        assert len(env.clients) == 0
        assert len(env._all_tool_defs) == 0

    @patch("enterpriseops_gym_env.MCPClient")
    def test_deduplicates_tools_across_servers(self, mock_mcp_cls):
        tool_schema = [{"name": "shared_tool", "description": "shared"}]

        def make_client():
            c = AsyncMock()
            c.connect.return_value = True
            c.list_tools.return_value = tool_schema
            return c

        mock_mcp_cls.side_effect = [make_client(), make_client()]

        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={"server1": "http://localhost:8001", "server2": "http://localhost:8002"},
            gym_dbs_path="gym_dbs",
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )

        assert len(env._all_tool_defs) == 1
        assert env.tool_to_server["shared_tool"] == "server1"


class TestCallTool:
    @patch("enterpriseops_gym_env.MCPClient")
    def test_routes_to_correct_server(self, mock_mcp_cls):
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.list_tools.return_value = [{"name": "my_tool", "description": ""}]
        mock_client.call_tool = AsyncMock(return_value={
            "success": True,
            "result": {"content": [{"type": "text", "text": "tool result"}]},
        })
        mock_mcp_cls.return_value = mock_client

        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={"srv": "http://localhost:9999"},
            gym_dbs_path="gym_dbs",
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )

        result = asyncio.run(env.call_tool("my_tool", {"arg": "val"}, "call-1"))
        assert result["content"] == "tool result"
        assert result["tool_call_id"] == "call-1"
        mock_client.call_tool.assert_called_once_with("my_tool", {"arg": "val"})

    @patch("enterpriseops_gym_env.MCPClient")
    def test_unknown_tool(self, mock_mcp_cls):
        mock_mcp_cls.return_value = AsyncMock()
        mock_mcp_cls.return_value.connect.return_value = False

        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={},
            gym_dbs_path="gym_dbs",
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )

        result = asyncio.run(env.call_tool("nonexistent", {}, "call-1"))
        assert "Error" in result["content"]
        assert "nonexistent" in result["content"]

    @patch("enterpriseops_gym_env.MCPClient")
    def test_tool_call_failure(self, mock_mcp_cls):
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.list_tools.return_value = [{"name": "failing_tool", "description": ""}]
        mock_client.call_tool = AsyncMock(return_value={"success": False, "error": "server error"})
        mock_mcp_cls.return_value = mock_client

        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={"srv": "http://localhost:9999"},
            gym_dbs_path="gym_dbs",
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )

        result = asyncio.run(env.call_tool("failing_tool", {}, "call-1"))
        assert "server error" in result["content"]


# -- Rubric / Verification ---------------------------------------------------


class TestBuildRubric:
    def test_response_check_skipped_without_llm_client(self):
        rubric = _build_rubric(DEFAULT_SERVER_URLS, llm_client=None)
        verify_fn = rubric.funcs[0]

        info = json.dumps({
            "verifiers": [{"verifier_type": "response_check", "validation_config": {}}],
        })
        state = State(database_ids={}, server_url_map={}, trajectory=[])

        score = asyncio.run(verify_fn(completion=[], answer="", state=state, info=info))
        assert score == 0.0

    def test_tool_execution_verifier(self):
        rubric = _build_rubric(DEFAULT_SERVER_URLS, llm_client=None)
        verify_fn = rubric.funcs[0]

        info = json.dumps({
            "verifiers": [{
                "verifier_type": "tool_execution",
                "validation_config": {"selected_tools": ["list_users"], "minimum_tool_calls": 1},
            }],
        })

        step = TrajectoryStep(
            completion=[AssistantMessage(
                role="assistant", content=None,
                tool_calls=[ToolCall(id="tc1", name="list_users", arguments="{}")],
            )],
            prompt=[], response=None, tokens=None,
            reward=None, advantage=None, is_truncated=False, trajectory_id="t1", extras={},
        )
        state = State(database_ids={}, server_url_map={}, trajectory=[step])

        with patch("enterpriseops_gym_env.VerifierEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.execute_verifier = AsyncMock(return_value={"passed": True})
            mock_engine_cls.return_value = mock_engine

            score = asyncio.run(verify_fn(completion=[], answer="", state=state, info=info))

        assert score == 1.0

    def test_tool_execution_verifier_fails(self):
        rubric = _build_rubric(DEFAULT_SERVER_URLS, llm_client=None)
        verify_fn = rubric.funcs[0]

        info = json.dumps({
            "verifiers": [{
                "verifier_type": "tool_execution",
                "validation_config": {"selected_tools": ["list_users"], "minimum_tool_calls": 1},
            }],
        })
        state = State(database_ids={}, server_url_map={}, trajectory=[])

        with patch("enterpriseops_gym_env.VerifierEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.execute_verifier = AsyncMock(return_value={"passed": False})
            mock_engine_cls.return_value = mock_engine

            score = asyncio.run(verify_fn(completion=[], answer="", state=state, info=info))

        assert score == 0.0

    def test_empty_verifiers(self):
        rubric = _build_rubric(DEFAULT_SERVER_URLS)
        verify_fn = rubric.funcs[0]

        info = json.dumps({"verifiers": []})
        state = State(database_ids={}, server_url_map={}, trajectory=[])

        score = asyncio.run(verify_fn(completion=[], answer="", state=state, info=info))
        assert score == 0.0

    def test_all_pass_metric(self):
        rubric = _build_rubric(DEFAULT_SERVER_URLS)
        all_pass_fn = rubric.funcs[1]  # add_metric appends to funcs with weight=0

        info = json.dumps({
            "verifiers": [{
                "verifier_type": "tool_execution",
                "validation_config": {"selected_tools": ["a"], "minimum_tool_calls": 1},
            }],
        })

        step = TrajectoryStep(
            completion=[AssistantMessage(
                role="assistant", content=None,
                tool_calls=[ToolCall(id="tc1", name="a", arguments="{}")],
            )],
            prompt=[], response=None, tokens=None,
            reward=None, advantage=None, is_truncated=False, trajectory_id="t1", extras={},
        )
        state = State(database_ids={}, server_url_map={}, trajectory=[step])

        with patch("enterpriseops_gym_env.VerifierEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.execute_verifier = AsyncMock(return_value={"passed": True})
            mock_engine_cls.return_value = mock_engine

            score = asyncio.run(all_pass_fn(completion=[], answer="", state=state, info=info))

        assert score == 1.0

    def test_cleanup_registered_on_rubric_not_env(self):
        """Database cleanup must run AFTER scoring (via rubric.cleanup), not during
        rollout (via @vf.cleanup), because verifiers need the database alive."""
        rubric = _build_rubric({"srv": "http://localhost:8002"})
        assert len(rubric._cleanup_handlers) == 1, "Expected 1 cleanup handler on rubric"

        # Verify EOpsGymEnv has NO @vf.cleanup methods
        import inspect
        import verifiers as vf
        cleanup_methods = [
            name for name, method in inspect.getmembers(EOpsGymEnv, predicate=inspect.isfunction)
            if hasattr(method, "__vf_cleanup__") or (hasattr(method, "__wrapped__") and "cleanup" in name)
        ]
        assert cleanup_methods == [], f"EOpsGymEnv should not have @vf.cleanup methods, found: {cleanup_methods}"

    @patch("enterpriseops_gym_env.delete_database")
    def test_cleanup_deletes_databases(self, mock_delete):
        rubric = _build_rubric({"srv": "http://localhost:8002"})
        cleanup_fn = rubric._cleanup_handlers[0]

        state = State(
            database_ids={"srv": "db_123"},
            server_url_map={"srv": "http://localhost:8002"},
            trajectory=[],
        )
        asyncio.run(cleanup_fn(state))
        mock_delete.assert_called_once_with("http://localhost:8002", "db_123")

    def test_mixed_verifiers_partial_pass(self):
        rubric = _build_rubric(DEFAULT_SERVER_URLS)
        verify_fn = rubric.funcs[0]

        info = json.dumps({
            "verifiers": [
                {"verifier_type": "database_state", "validation_config": {"query": "SELECT 1"}, "gym_name": "srv"},
                {"verifier_type": "database_state", "validation_config": {"query": "SELECT 2"}, "gym_name": "srv"},
            ],
        })
        state = State(database_ids={"srv": "db1"}, server_url_map={"srv": "http://localhost:8002"}, trajectory=[])

        with patch("enterpriseops_gym_env.VerifierEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.execute_verifier = AsyncMock(side_effect=[
                {"passed": True}, {"passed": False},
            ])
            mock_engine_cls.return_value = mock_engine

            score = asyncio.run(verify_fn(completion=[], answer="", state=state, info=info))

        assert score == 0.5


# -- Setup state --------------------------------------------------------------


class TestSetupState:
    @patch("enterpriseops_gym_env.create_database_from_file")
    @patch("enterpriseops_gym_env.MCPClient")
    def test_seeds_database_and_filters_tools(self, mock_mcp_cls, mock_create_db):
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.list_tools.return_value = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B"},
            {"name": "tool_c", "description": "C"},
        ]
        mock_mcp_cls.return_value = mock_client
        mock_create_db.return_value = "db_123"

        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={"srv": "http://localhost:8002"},
            gym_dbs_path="/fake/gym_dbs",
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )

        assert len(env._all_tool_defs) == 3

        info = json.dumps({
            "gym_servers_config": [{"mcp_server_name": "srv", "mcp_server_url": "http://localhost:8002"}],
            "selected_tools": ["tool_a", "tool_c"],
        })
        state = State(input={"info": info, "prompt": [], "answer": ""})

        async def run():
            return await env.setup_state(state)

        result = asyncio.run(run())

        # No seed file, so no DB created
        assert result["database_ids"] == {}
        assert result["server_url_map"] == {"srv": "http://localhost:8002"}
        # Tools filtered to selected
        assert len(env.tool_defs) == 2
        assert {t.name for t in env.tool_defs} == {"tool_a", "tool_c"}

    @patch("enterpriseops_gym_env._resolve_sql_path", return_value="/fake/db.sql")
    @patch("enterpriseops_gym_env.create_database_from_file")
    @patch("enterpriseops_gym_env.MCPClient")
    def test_handles_create_db_failure(self, mock_mcp_cls, mock_create_db, mock_resolve):
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.list_tools.return_value = []
        mock_mcp_cls.return_value = mock_client
        mock_create_db.return_value = None  # failure

        import verifiers as vf
        env = EOpsGymEnv(
            server_urls={"srv": "http://localhost:8002"},
            gym_dbs_path="/fake/gym_dbs",
            dataset=vf.load_example_dataset("gsm8k"),
            rubric=vf.Rubric(funcs=[]),
        )

        info = json.dumps({
            "gym_servers_config": [{
                "mcp_server_name": "srv",
                "mcp_server_url": "http://localhost:8002",
                "seed_database_file": "db.sql",
            }],
            "selected_tools": [],
        })
        state = State(input={"info": info, "prompt": [], "answer": ""})

        result = asyncio.run(env.setup_state(state))

        assert result["database_ids"] == {}  # None was not stored


# -- Constants ----------------------------------------------------------------


class TestConstants:
    def test_all_domains_includes_hybrid(self):
        assert "hybrid" in ALL_DOMAINS
        assert len(ALL_DOMAINS) == 8

    def test_default_server_urls_match_hf_data(self):
        expected_servers = {
            "sn-csm-server", "gym-teams-mcp", "gym-calendar",
            "gym-email-mcp", "gym-itsm-mcp", "sn-hr-internal", "gym-google-drive-mcp",
        }
        assert set(DEFAULT_SERVER_URLS.keys()) == expected_servers
