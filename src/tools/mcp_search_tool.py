# src/tools/mcp_search_tool.py
import asyncio
import json
from typing import Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

class MCPSearchTool:
    """
    MCP client wrapper for SerpAPI search functionality
    """
    
    def __init__(self, server_path: str):
        self.server_path = server_path
    
    async def _create_client(self):
        """Create a new MCP client instance for each operation"""

        server_params = StdioServerParameters(
            command="python",
            args=[self.server_path]
        )
        
        client_context = stdio_client(server_params)
        read, write = await client_context.__aenter__()
        
        session_context = ClientSession(read, write)
        session = await session_context.__aenter__()
        
        # Initialize the session
        await session.initialize()
        
        return client_context, session_context, session
    
    async def search_health_info(self, query: str, location: str = "United States", num: int = 10) -> str:
        """Search for health information using a dedicated MCP client"""

        client_context, session_context, session = await self._create_client()
        
        try:
            # Load tools for this specific session
            tools = await load_mcp_tools(session)
            
            # Find the health search tool
            health_search_tool = next(
                (tool for tool in tools if tool.name == "search_health_info"), 
                None
            )
            
            if not health_search_tool:
                return json.dumps({"error": "Health search tool not found"})
            
            # Execute the tool
            result = await health_search_tool.ainvoke({
                "query": query,
                "location": location,
                "num": num
            })
            
            return result
            
        except Exception as e:
            return json.dumps({"error": f"Health search failed: {str(e)}"})
        finally:
            # Properly close the session and client in the same task
            try:
                await session_context.__aexit__(None, None, None)
                await client_context.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"Warning: Error during MCP client cleanup: {cleanup_error}")
    
    async def search_alternatives(self, product_name: str, dietary_restrictions: str = "", num: int = 10) -> str:
        """Search for food alternatives using a dedicated MCP client"""

        client_context, session_context, session = await self._create_client()
        
        try:
            # Load tools for this specific session
            tools = await load_mcp_tools(session)
            
            # Find the alternatives search tool
            alternatives_tool = next(
                (tool for tool in tools if tool.name == "search_food_alternatives"), 
                None
            )
            
            if not alternatives_tool:
                return json.dumps({"error": "Alternatives search tool not found"})
            
            # Execute the tool
            result = await alternatives_tool.ainvoke({
                "product_name": product_name,
                "dietary_restrictions": dietary_restrictions,
                "num": num
            })
            
            return result
            
        except Exception as e:
            return json.dumps({"error": f"Alternatives search failed: {str(e)}"})
        finally:
            # Properly close the session and client in the same task
            try:
                await session_context.__aexit__(None, None, None)
                await client_context.__aexit__(None, None, None)
            except Exception as cleanup_error:
                print(f"Warning: Error during MCP client cleanup: {cleanup_error}")

# Synchronous wrapper that creates a new async event loop per operation
class SyncMCPSearchTool:
    """
    Synchronous wrapper that creates isolated async contexts for each search operation
    """
    
    def __init__(self, server_path: str):
        self.server_path = server_path
    
    def search(self, query: str, search_type: str = "health") -> str:
        """
        Synchronous search method that creates an isolated async context
        
        Args:
            query: Search query
            search_type: Type of search ("health" or "alternatives")
        
        Returns:
            JSON string with search results
        """
        # Create a new MCPSearchTool instance for each search to ensure isolation
        mcp_tool = MCPSearchTool(self.server_path)
        
        # Create a new event loop for this specific search operation
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, run in a thread pool to avoid conflicts
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_search_in_new_loop, mcp_tool, query, search_type)
                    return future.result(timeout=30)  # 30 second timeout
            except RuntimeError:
                # No running loop, safe to create a new one
                return asyncio.run(self._perform_search(mcp_tool, query, search_type))
                
        except Exception as e:
            return json.dumps({"error": f"Search operation failed: {str(e)}"})
    
    def _run_search_in_new_loop(self, mcp_tool, query, search_type):
        """Run search in a completely new event loop"""
        return asyncio.run(self._perform_search(mcp_tool, query, search_type))
    
    async def _perform_search(self, mcp_tool, query, search_type):
        """Perform the actual search operation"""
        if search_type == "alternatives":
            return await mcp_tool.search_food_alternatives(query)
        else:
            return await mcp_tool.search_health_info(query)
