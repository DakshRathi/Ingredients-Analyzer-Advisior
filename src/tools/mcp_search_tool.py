# src/tools/mcp_search_tool.py
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

class MCPSearchTool:
    """
    MCP client wrapper that creates a new, isolated client instance for each async operation.
    This prevents async context conflicts when used in parallel by LangGraph nodes.
    """
    def __init__(self, server_path: str):
        self.server_path = server_path

    async def _execute_tool(self, tool_name: str, **kwargs):
        """
        A generic, isolated async method to create a client, execute a tool, and close resources.
        """
        client_context = None
        session_context = None
        try:
            # 1. Create a new MCP client instance for this specific operation
            server_params = StdioServerParameters(command=["python", self.server_path])
            client_context = stdio_client(server_params)
            read, write = await client_context.__aenter__()

            session_context = ClientSession(read, write)
            session = await session_context.__aenter__()
            await session.initialize()

            # 2. Load tools for this session
            tools = await load_mcp_tools(session)
            target_tool = next((t for t in tools if t.name == tool_name), None)

            if not target_tool:
                return json.dumps({"error": f"Tool '{tool_name}' not found in MCP server."})

            # 3. Execute the tool
            result = await target_tool.ainvoke(kwargs)
            return result

        except Exception as e:
            return json.dumps({"error": f"MCP tool execution failed for '{tool_name}': {str(e)}"})
        finally:
            # 4. Properly close the session and client in the same task
            if session_context:
                try:
                    await session_context.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    print(f"Warning: Error closing MCP session context: {cleanup_error}")
            if client_context:
                try:
                    await client_context.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    print(f"Warning: Error closing MCP client context: {cleanup_error}")

    async def search_health_info(self, query: str, location: str = "United States") -> str:
        """Search for health information using a dedicated MCP client."""
        return await self._execute_tool("search_health_info", query=query, location=location)

    async def search_food_alternatives(self, product_name: str, dietary_restrictions: str = "") -> str:
        """Search for food alternatives using a dedicated MCP client."""
        return await self._execute_tool("search_food_alternatives", product_name=product_name, dietary_restrictions=dietary_restrictions)
