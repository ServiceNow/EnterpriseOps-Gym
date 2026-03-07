import httpx
import json
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage

from benchmark.llm_client import LLMClient
from benchmark.mcp_client import MCPClient
from benchmark.models import VerifierConfig

logger = logging.getLogger(__name__)


class VerifierEngine:
    """
    Verifier engine for validating benchmark results.
    Supports: database_state, response_check
    Multi-gym aware: Can query different gym databases using mcp_clients dict
    """

    def __init__(
        self, mcp_clients: Union[MCPClient, Dict[str, MCPClient]], llm_client: LLMClient
    ):
        """
        Initialize VerifierEngine with MCP client(s)

        Args:
            mcp_clients: Either a single MCPClient (legacy) or dict of gym_name -> MCPClient (multi-gym)
            llm_client: LLM client for judge-based verification
        """
        if isinstance(mcp_clients, dict):
            self.mcp_clients = mcp_clients
            self.mcp_client = (
                list(mcp_clients.values())[0] if mcp_clients else None
            )  # Backward compat
        else:
            self.mcp_client = mcp_clients
            self.mcp_clients = {"default": mcp_clients}
        self.llm_client = llm_client

    def _get_mcp_client_for_gym(self, gym_name: Optional[str] = None) -> MCPClient:
        """Get the appropriate MCP client for a gym, or default"""
        if gym_name and gym_name in self.mcp_clients:
            return self.mcp_clients[gym_name]
        return self.mcp_client or list(self.mcp_clients.values())[0]

    async def execute_verifier(
        self,
        verifier: VerifierConfig,
        model_response: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
        gym_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single verifier"""
        logger.info(f"Executing verifier: {verifier.verifier_type}")

        if verifier.verifier_type == "database_state":
            return await self._execute_database_state_verifier(
                verifier.validation_config, database_id, context, gym_name
            )
        elif verifier.verifier_type == "response_check":
            return await self._execute_response_check_verifier(
                verifier.validation_config,
                model_response,
                database_id,
                context,
                gym_name,
            )
        elif verifier.verifier_type == "tool_execution":
            return await self._execute_tool_execution_verifier(
                verifier.validation_config, model_response
            )
        else:
            return {
                "passed": False,
                "error": f"Unsupported verifier type: {verifier.verifier_type}",
            }

    async def _execute_database_state_verifier(
        self,
        validation_config: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
        gym_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute database state verifier"""
        sql_query = validation_config.get("query")
        expected_value = validation_config.get("expected_value")
        comparison_type = validation_config.get("comparison_type", "equals")

        if not sql_query:
            return {"passed": False, "error": "No SQL query provided"}

        # logger.info(f"Executing SQL query: {sql_query}")

        # Execute SQL query via MCP (use correct client for gym)
        result = await self._execute_sql_query(
            sql_query, database_id, context, gym_name
        )

        if not result["success"]:
            return {
                "passed": False,
                "error": f"SQL query failed: {result.get('error')}",
                "query": sql_query,
            }

        # Extract value from result
        actual_value = self._extract_value_from_sql_result(result)

        # logger.info(f"SQL result - Expected: {expected_value}, Actual: {actual_value}")

        # Compare values
        comparison_result = self._compare_values(
            actual_value, expected_value, comparison_type
        )

        return {
            "passed": comparison_result["passed"],
            "expected": expected_value,
            "actual": actual_value,
            "comparison_type": comparison_type,
            "query": sql_query,
            "details": comparison_result.get("details"),
        }

    async def _execute_response_check_verifier(
        self,
        validation_config: Dict[str, Any],
        model_response: Dict[str, Any],
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
        gym_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute response check verifier using LLM-as-judge"""
        sql_query = validation_config.get("sql_query")
        comparison_prompt = validation_config.get("comparison_prompt")
        minimum_comparison_value = validation_config.get("minimum_comparison_value", 7)

        if not sql_query or not comparison_prompt:
            return {"passed": False, "error": "Missing sql_query or comparison_prompt"}

        # Execute SQL query
        sql_result = await self._execute_sql_query(
            sql_query, database_id, context, gym_name
        )

        if not sql_result["success"]:
            return {
                "passed": False,
                "error": f"SQL query failed: {sql_result.get('error')}",
            }

        # Extract LLM response text
        llm_response_text = self._extract_llm_content(model_response)

        # Use LLM as judge
        judge_result = await self._compare_with_llm(
            sql_result, llm_response_text, comparison_prompt, minimum_comparison_value
        )

        return judge_result

    async def _execute_tool_execution_verifier(
        self, validation_config: Dict[str, Any], model_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool execution verifier"""
        selected_tools = validation_config.get("selected_tools", [])
        minimum_tool_calls = validation_config.get("minimum_tool_calls", 1)

        # Extract tools called from model response
        tools_called = []
        if "tool_calls" in model_response and model_response["tool_calls"]:
            tools_called = [tc["name"] for tc in model_response["tool_calls"]]

        logger.info(f"Expected tools: {selected_tools}, Called: {tools_called}")

        # Check if expected tools were called
        missing_tools = [tool for tool in selected_tools if tool not in tools_called]

        # Check minimum tool calls
        passed = len(missing_tools) == 0 and len(tools_called) >= minimum_tool_calls

        return {
            "passed": passed,
            "selected_tools": selected_tools,
            "tools_called": tools_called,
            "missing_tools": missing_tools,
            "minimum_tool_calls": minimum_tool_calls,
            "actual_tool_calls": len(tools_called),
        }

    async def _execute_sql_query(
        self,
        query: str,
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
        gym_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute SQL query via MCP sql-runner API endpoint"""
        try:
            # Get the correct MCP client for this gym
            mcp_client = self._get_mcp_client_for_gym(gym_name)

            # Use the sql-runner API endpoint directly instead of MCP tool
            # This matches the production implementation
            base_url = mcp_client.base_url
            api_url = f"{base_url.rstrip('/')}/api/sql-runner"

            payload = {
                "query": query,
                "database_id": database_id or mcp_client.database_id,
            }

            # Build headers
            headers = {
                "Content-Type": "application/json",
                "x-database-id": database_id or mcp_client.database_id,
            }

            # Add authentication headers
            headers.update(mcp_client._get_auth_headers())

            # Add context headers
            if mcp_client.context:
                for key, value in mcp_client.context.items():
                    if not key.lower().startswith("x-"):
                        header_key = f"x-{key.lower().replace('_', '-')}"
                    else:
                        header_key = key
                    headers[header_key] = str(value)

            # Add any override context
            if context:
                for key, value in context.items():
                    if not key.lower().startswith("x-"):
                        header_key = f"x-{key.lower().replace('_', '-')}"
                    else:
                        header_key = key
                    headers[header_key] = str(value)

            logger.info(f"Executing SQL query via {api_url}: {query}")

            # Make HTTP request
            timeout = httpx.Timeout(30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                api_result = response.json()

                logger.debug(f"SQL API response: {api_result}")

                return {"success": True, "result": api_result}

        except Exception as e:
            logger.error(f"SQL query execution failed: {e}")

            # Handle HTTP errors with detailed messages
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_details = ""
                    if hasattr(e.response, "text"):
                        response_text = e.response.text
                        try:
                            error_json = json.loads(response_text)
                            if isinstance(error_json, dict):
                                error_details = error_json.get(
                                    "detail", error_json.get("message", response_text)
                                )
                            else:
                                error_details = response_text
                        except json.JSONDecodeError:
                            error_details = response_text

                    status_code = getattr(e.response, "status_code", "Unknown")
                    logger.error(
                        f"HTTP error calling sql-runner API: {status_code} - {error_details}"
                    )

                    return {
                        "success": False,
                        "error": f"SQL API call failed (HTTP {status_code}): {error_details}",
                    }
                except Exception as parse_error:
                    logger.error(f"Error parsing HTTP error response: {parse_error}")

            return {"success": False, "error": str(e)}

    def _extract_value_from_sql_result(self, result: dict) -> Any:
        """Extract the actual value from SQL query result (matches production implementation)"""
        if not result:
            return None

        # If the result itself is not successful, check for error content
        if not result.get("success"):
            # Try to extract error message from MCP response format
            result_data = result.get("result", {})
            if isinstance(result_data, dict) and "content" in result_data:
                content = result_data["content"]
                if isinstance(content, list) and len(content) > 0:
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item.get("text", "Error")
            return None

        result_data = result.get("result", {})

        # Handle different result formats from MCP sql-runner
        if isinstance(result_data, dict):
            # If result has 'data' field (common format)
            if "data" in result_data:
                data = result_data["data"]
                if isinstance(data, list) and len(data) > 0:
                    # If single row with single column, return the value directly
                    if (
                        len(data) == 1
                        and isinstance(data[0], dict)
                        and len(data[0]) == 1
                    ):
                        return list(data[0].values())[0]
                    # If single row with multiple columns, return the row dict
                    elif len(data) == 1:
                        return data[0]
                    # Multiple rows, return the full result
                    else:
                        return data
                return data

            # If result has 'rows' field
            elif "rows" in result_data:
                rows = result_data["rows"]
                if isinstance(rows, list) and len(rows) > 0:
                    # Single value from single row
                    if (
                        len(rows) == 1
                        and isinstance(rows[0], dict)
                        and len(rows[0]) == 1
                    ):
                        return list(rows[0].values())[0]
                    # Single row as list
                    elif (
                        len(rows) == 1
                        and isinstance(rows[0], list)
                        and len(rows[0]) == 1
                    ):
                        return rows[0][0]
                    # Single row (dict or list)
                    elif len(rows) == 1:
                        return rows[0]
                    # Multiple rows
                    else:
                        return rows
                return rows

            # If result has 'content' field (MCP error format)
            elif "content" in result_data:
                content = result_data["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from content array
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item.get("text", result_data)
                return content

            # Direct result format (nested 'result' field)
            elif "result" in result_data:
                return result_data["result"]

        # Return as-is if we can't extract anything specific
        return result_data

    def _compare_values(
        self, actual: Any, expected: Any, comparison_type: str
    ) -> Dict[str, Any]:
        """Compare actual vs expected values"""
        try:
            if comparison_type == "equals":
                passed = actual == expected
            elif comparison_type == "greater_than":
                passed = actual > expected
            elif comparison_type == "less_than":
                passed = actual < expected
            elif comparison_type == "contains":
                passed = expected in str(actual)
            else:
                return {
                    "passed": False,
                    "details": f"Unknown comparison type: {comparison_type}",
                }

            return {
                "passed": passed,
                "details": f"Comparison {comparison_type}: {actual} vs {expected}",
            }

        except Exception as e:
            return {"passed": False, "details": f"Comparison error: {e}"}

    def _extract_llm_content(self, model_response: Dict[str, Any]) -> str:
        """Extract text content from LLM response"""
        if "content" in model_response:
            return str(model_response["content"])
        elif "text" in model_response:
            return str(model_response["text"])
        elif "response" in model_response:
            return str(model_response["response"])

        return str(model_response)

    async def _compare_with_llm(
        self,
        sql_result: Dict[str, Any],
        llm_response: str,
        comparison_prompt: str,
        minimum_score: int,
    ) -> Dict[str, Any]:
        """Use LLM as judge to compare SQL result with LLM response"""
        # Build judge prompt
        system_prompt = """You are an AI judge evaluating the quality and accuracy of an AI assistant's response.
Compare the database query result with the AI's response and rate how well they match.
Provide a score from 1-10 where:
- 1-3: Poor match, incorrect or missing information
- 4-6: Partial match, some correct information
- 7-8: Good match, mostly correct
- 9-10: Excellent match, fully accurate

Respond with ONLY a JSON object in this format:
{
  "score": <number 1-10>,
  "reasoning": "<brief explanation>"
}"""

        sql_result_str = json.dumps(sql_result.get("result", {}), indent=2)

        user_prompt = f"""Database Query Result:
{sql_result_str}

AI Assistant Response:
{llm_response}

Comparison Task:
{comparison_prompt}

Please provide your judgment as JSON."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await self.llm_client.llm.ainvoke(messages)
            response_text = response.content

            # Parse JSON response
            # Try to extract JSON from markdown code blocks
            if "```json" in response_text:
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            judge_result = json.loads(response_text)
            score = judge_result.get("score", 0)
            reasoning = judge_result.get("reasoning", "")

            passed = score >= minimum_score

            return {
                "passed": passed,
                "score": score,
                "minimum_score": minimum_score,
                "reasoning": reasoning,
                "sql_result": sql_result_str,
                "llm_response": llm_response,
            }

        except Exception as e:
            logger.error(f"LLM judge comparison failed: {e}")
            return {"passed": False, "error": f"Judge comparison failed: {e}"}
