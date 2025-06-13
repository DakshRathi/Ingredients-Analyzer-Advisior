# src/mcp_servers/serpapi_server.py
import os
import json
from mcp.server.fastmcp import FastMCP
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("SerpAPI Web Search")

@mcp.tool()
def web_search(query: str, location: str = "United States", num: int = 10) -> str:
    """
    Performs a flexible web search using SerpAPI.
    The LLM provides the search query dynamically.

    Args:
        query (str): The search query provided by the LLM.
        location (str): The location for the search results (default is "United States").
        num (int): The number of results to return (default is 10).
    Returns:
        str: A JSON string containing the search results, including titles, links, and snippets.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return json.dumps({"error": "SerpAPI key not found"})
    try:
        params = {
            "engine": "google_light",
            "q": query,
            "location": location,
            "hl": "en",
            "gl": "us",
            "num": num,
            "api_key": api_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        formatted_results = []
        if "organic_results" in results:
            for result in results["organic_results"][:num]:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
        return json.dumps({
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        })
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})

if __name__ == "__main__":
    mcp.run(transport="sse")
