import httpx
import json
import logging
import os
import random
import string
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================


def create_database_from_file(gym_url: str, sql_file_path: str) -> Optional[str]:
    """Create a new database from a SQL file and return database_id."""
    try:
        # Generate unique database_id
        timestamp = int(time.time() * 1000)
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))
        database_id = f"db_{timestamp}_{suffix}"
        headers = {"Content-Type": "application/json"}

        # Read SQL content from file
        logger.info(f"📥 Reading SQL from file: {sql_file_path}...")

        if not os.path.exists(sql_file_path):
            logger.error(f"❌ SQL file not found: {sql_file_path}")
            return None

        with open(sql_file_path, "r", encoding="utf-8") as f:
            sql_content = f.read()

        logger.info(f"   SQL size: {len(sql_content) / 1024:.2f} KB")

        # Create database
        db_name = f"Auto DB {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🔨 Creating database '{db_name}' from file...")
        payload = {
            "database_id": database_id,
            "name": db_name,
            "description": f"Auto-created from {os.path.basename(sql_file_path)}",
            "sql_content": sql_content,
        }

        timeout = max(1200, int(120 + len(sql_content) / 102400))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{gym_url}/api/seed-database", headers=headers, json=payload
            )
            response.raise_for_status()

        logger.info(f"✅ Database created from file: {database_id}")
        return database_id

    except Exception as e:
        logger.error(f"❌ Error creating database from file: {e}")
        raise e


def delete_database(gym_url: str, database_id: str) -> bool:
    """Delete a database from the Gym server."""
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"database_id": database_id}

        logger.info(f"🗑️  Deleting database: {database_id}...")

        with httpx.Client(timeout=30) as client:
            response = client.request(
                "DELETE",
                f"{gym_url}/api/delete-database",
                headers=headers,
                json=payload,
            )

            # Handle servers that don't have this API
            if response.status_code == 404:
                logger.warning(f"⚠️  Server does not support database deletion API")
                return False
            elif response.status_code == 405:
                logger.warning(f"⚠️  Database deletion not allowed on this server")
                return False

            response.raise_for_status()

        logger.info(f"✅ Database deleted successfully")
        return True

    except httpx.HTTPStatusError as e:
        if e.response.status_code in [404, 405]:
            logger.warning(
                f"⚠️  Server does not support database deletion (HTTP {e.response.status_code})"
            )
        else:
            logger.error(f"❌ Error deleting database: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error deleting database: {e}")
        return False


# ============================================================================
# MCP CLIENT (JSON-RPC HTTP Implementation)
# ============================================================================


class MCPClient:
    """
    HTTP-based MCP Client for JSON-RPC communication with MCP servers.
    Implements the same protocol as fastmcp_http_client.py
    """

    def __init__(
        self,
        base_url: str,
        auth_config: Optional[Dict[str, Any]] = None,
        mcp_endpoint: str = "/mcp",
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.mcp_endpoint = mcp_endpoint
        self.session_id = 1
        self.connected = False
        self.mcp_session_id = None
        self.auth_config = auth_config
        self.database_id = database_id
        self.context = context or {}

    def _get_request_id(self) -> int:
        """Get next request ID"""
        self.session_id += 1
        return self.session_id

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for HTTP requests"""
        if not self.auth_config:
            return {}

        auth_type = self.auth_config.get("type")
        token = self.auth_config.get("token")
        header_name = self.auth_config.get("header_name", "Authorization")

        if auth_type == "bearer":
            return {header_name: f"Bearer {token}"}
        elif auth_type == "api_key":
            return {header_name: token}

        return {}

    async def _send_request(
        self,
        method: str,
        params: Dict[str, Any] = None,
        extra_headers: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": method,
                "params": params or {},
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }

            # Add authentication headers
            headers.update(self._get_auth_headers())

            # Add session ID if we have one
            if self.mcp_session_id:
                headers["mcp-session-id"] = self.mcp_session_id

            # Add database ID header if set
            if self.database_id:
                headers["x-database-id"] = self.database_id

            # Add context headers (dynamic - convert all context key-value pairs to x-* headers)
            if self.context and isinstance(self.context, dict):
                for key, value in self.context.items():
                    # Convert context keys to header format: user_id -> x-user-id
                    if not key.lower().startswith("x-"):
                        header_key = f"x-{key.lower().replace('_', '-')}"
                    else:
                        header_key = key
                    headers[header_key] = str(value)

            # Add extra headers (these override defaults if same key exists)
            if extra_headers:
                headers.update(extra_headers)

            timeout = httpx.Timeout(30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(
                    f"Sending MCP request: {method} to {self.base_url}{self.mcp_endpoint}"
                )
                logger.debug(f"Headers: {headers}")
                logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

                response = await client.post(
                    f"{self.base_url}{self.mcp_endpoint}", json=payload, headers=headers
                )

                # Capture session ID from response headers
                if "mcp-session-id" in response.headers:
                    self.mcp_session_id = response.headers["mcp-session-id"]
                    logger.info(f"Captured MCP session ID: {self.mcp_session_id}")

                if response.status_code == 200:
                    data = response.json()
                    logger.debug(f"MCP response: {json.dumps(data, indent=2)}")
                    return {"success": True, "data": data}
                else:
                    error_msg = (
                        f"MCP request failed: {response.status_code} - {response.text}"
                    )
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}

        except Exception as e:
            logger.error(f"MCP request exception: {e}")
            return {"success": False, "error": str(e)}

    async def connect(self) -> bool:
        """Connect to HTTP MCP server"""
        try:
            result = await self.initialize()
            if result.get("success"):
                self.connected = True
                logger.info(f"Connected to MCP server: {self.base_url}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False

    async def initialize(self) -> Dict[str, Any]:
        """Initialize MCP session"""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "standalone-benchmark-executor", "version": "1.0.0"},
        }

        result = await self._send_request("initialize", params)
        if result.get("success"):
            # Send notifications/initialized
            await self._send_notification("notifications/initialized", {})
            logger.info("MCP session initialized successfully")
        return result

    async def _send_notification(
        self,
        method: str,
        params: Dict[str, Any] = None,
        extra_headers: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Send notification (no response expected)"""
        try:
            payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }

            headers.update(self._get_auth_headers())

            if self.mcp_session_id:
                headers["mcp-session-id"] = self.mcp_session_id

            if extra_headers:
                headers.update(extra_headers)

            timeout = httpx.Timeout(30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}{self.mcp_endpoint}", json=payload, headers=headers
                )

                return {
                    "success": response.status_code in [200, 204],
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        result = await self._send_request("tools/list", {})
        if result.get("success"):
            data = result.get("data", {})
            return data.get("result", {}).get("tools", [])
        return []

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a specific tool"""
        params = {"name": tool_name, "arguments": arguments or {}}

        # Build extra headers (override instance values if provided)
        extra_headers = {}
        if database_id:
            extra_headers["x-database-id"] = database_id

        # Add any additional context headers (these override instance context)
        if context and isinstance(context, dict):
            for key, value in context.items():
                if not key.lower().startswith("x-"):
                    header_key = f"x-{key.lower().replace('_', '-')}"
                else:
                    header_key = key
                extra_headers[header_key] = str(value)

        logger.info(f"Calling tool '{tool_name}' with args: {arguments}")
        if extra_headers:
            logger.info(f"Override headers: {extra_headers}")

        result = await self._send_request("tools/call", params, extra_headers)
        if result.get("success"):
            data = result.get("data", {})
            return {
                "success": True,
                "result": data.get("result"),
                "error": data.get("error"),
            }
        return result
