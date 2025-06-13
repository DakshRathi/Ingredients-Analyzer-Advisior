# src/tools/mcp_tools.py

from langchain.tools import tool
from pydantic import BaseModel, Field
import inspect

# Import the actual tool functions directly from your MCP server modules
from src.mcp_servers.serpapi_server import web_search as serpapi_web_search_func
from src.mcp_servers.website_content_server import fetch_website_content as scraper_fetch_content_func

# --- Pydantic Schemas for Tool Inputs ---
# This creates a strict contract for the LLM.
class WebSearchInput(BaseModel):
    query: str = Field(description="The precise and specific search query to be used for the web search.")

class FetchContentInput(BaseModel):
    url: str = Field(description="The valid URL of the website to fetch content from. Must start with http or https.")

# --- Argument Cleanup Wrapper ---
def argument_cleanup_wrapper(func):
    async def wrapper(**kwargs):
        # Check if the arguments are nested under a 'kwargs' key
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            # This handles the case: {'kwargs': {'query': '...'}}
            cleaned_args = kwargs['kwargs']
        elif 'kwargs' in kwargs and isinstance(kwargs['kwargs'], str):
            # This handles the case: {'kwargs': 'health risks...'}
            # We assume the string value should be the first argument of the target function.
            func_params = inspect.signature(func).parameters
            first_param_name = next(iter(func_params))
            cleaned_args = {first_param_name: kwargs['kwargs']}
        else:
            # The arguments are already in the correct format
            cleaned_args = kwargs
        
        # Call the original async function with the cleaned arguments
        return await func(**cleaned_args)
    return wrapper

# --- Correctly Defined LangChain Tools ---

@tool(args_schema=WebSearchInput, description="Performs a flexible web search using SerpAPI to get an overview of a topic.")
async def web_search_tool(query: str) -> str:
    """LangChain tool wrapper for the in-process SerpAPI web_search function."""
    return serpapi_web_search_func(query=query)

@tool(args_schema=FetchContentInput, description="Fetches the main text content from a given website URL.")
async def fetch_website_content_tool(url: str) -> str:
    """LangChain tool wrapper for the in-process website content fetching function."""
    return scraper_fetch_content_func(url=url)

web_search_tool.func = argument_cleanup_wrapper(web_search_tool.func)
fetch_website_content_tool.func = argument_cleanup_wrapper(fetch_website_content_tool.func)


# This list will be passed to your LangGraph agent
ALL_MCP_LANGCHAIN_TOOLS = [
    web_search_tool,
    fetch_website_content_tool,
]
