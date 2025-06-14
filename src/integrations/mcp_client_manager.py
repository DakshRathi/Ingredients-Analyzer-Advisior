# src/integrations/mcp_client_manager.py
import json
from contextlib import AsyncExitStack
from typing import List, Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain.tools import tool as langchain_tool_decorator

class MCPClientManager:
    """
    Manages persistent connections to multiple MCP servers as background processes.
    Initializes on application startup and cleans up on shutdown.
    """
    def __init__(self, config_path: str):
        """
        Initializes the MCP client manager.

        Args:
            config_path (str): The path to the configuration file that specifies the
                MCP servers to connect to.
        """
        self.config_path = config_path
        self.exit_stack = AsyncExitStack()
        self.tool_to_session: Dict[str, ClientSession] = {}
        self._available_langchain_tools: List[Any] = []

    async def connect_to_servers(self):
        """Reads config, starts all servers, and populates available tools."""
        print("--- Connecting to all configured MCP servers... ---")
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            servers = config.get("mcp_servers", {})

            for name, conf in servers.items():
                await self._connect_to_server(name, conf)

            print("--- All MCP servers connected and tools discovered. ---")
        except Exception as e:
            print(f"Error initializing MCP servers: {e}")
            raise

    async def _connect_to_server(self, server_name: str, server_config: dict):
        """Connects to a single MCP server and registers its tools."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            response = await session.list_tools()
            tools = response.tools
            print(f"Connected to '{server_name}' with tools: {[t.name for t in tools]}")
            
            for t in tools:
                self.tool_to_session[t.name] = session
                self._available_langchain_tools.append(self._create_langchain_tool(t))
        except Exception as e:
            print(f"Failed to connect to MCP server '{server_name}': {e}")
            raise

    def _create_langchain_tool(self, mcp_tool):
        """Dynamically creates a LangChain tool from an MCP tool definition."""
        
        @langchain_tool_decorator(name_or_callable=mcp_tool.name, description=mcp_tool.description, args_schema=mcp_tool.inputSchema)
        async def dynamic_tool(**kwargs) -> str:
            return await self.execute_tool(mcp_tool.name, kwargs)
        
        return dynamic_tool

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Executes a tool call on the appropriate MCP server."""
        session = self.tool_to_session.get(tool_name)
        if not session:
            return f"Error: Tool '{tool_name}' is not available."
        
        try:
            print(f"MCP MANAGER: Executing tool '{tool_name}' with args: {arguments}")
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.content
        except Exception as e:
            return f"Error calling tool '{tool_name}': {str(e)}"

    def get_tools(self) -> List[Any]:
        """
        Public method to safely retrieve the list of available LangChain tools.
        """
        return self._available_langchain_tools
    
    async def cleanup(self):
        """Closes all connections and shuts down servers gracefully."""
        print("--- Cleaning up MCP connections and shutting down servers... ---")
        await self.exit_stack.aclose()
        print("--- MCP cleanup complete. ---")

if __name__ == "__main__":
    import asyncio

    manager = MCPClientManager("server_config.json")

    async def main():
        await manager.connect_to_servers()
        print("Available tools:", manager.get_tools())
        await manager.cleanup()

    asyncio.run(main())