# src/mcp_servers/serpapi_server.py
import os
import json
from mcp.server.fastmcp import FastMCP
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("SerpAPI Health Search")

@mcp.tool()
def search_health_info(query: str, location: str = "United States", num: int = 10) -> str:
    """
    Search for health-related information using SerpAPI Google Search.
    
    Args:
        query: The search query related to health, nutrition, or food ingredients
        location: Location for search results (default: United States)
    
    Returns:
        JSON string containing search results with titles, links, and snippets
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return json.dumps({"error": "SerpAPI key not found"})
    
    try:
        # Configure search parameters for health/nutrition queries
        params = {
            "engine": "google_light",
            "q": f"{query} health nutrition benefits risks",
            "location": location,
            "hl": "en",
            "gl": "us",
            "num": num,
            "api_key": api_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract relevant information
        formatted_results = []
        if "organic_results" in results:
            for result in results["organic_results"][:num]:

                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                })
        
        return json.dumps({
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        })
        
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})

@mcp.tool()
def search_food_alternatives(product_name: str, dietary_restrictions: str = "", num: int = 10) -> str:
    """
    Search for healthy food alternatives using SerpAPI.
    
    Args:
        product_name: Name of the food product to find alternatives for
        dietary_restrictions: Any dietary restrictions to consider (e.g., "vegan", "gluten-free")
        num: Number of alternative suggestions to return (default: 3)
    
    Returns:
        JSON string containing alternative food suggestions
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return json.dumps({"error": "SerpAPI key not found"})
    
    try:
        search_query = f"healthy alternatives to {product_name} {dietary_restrictions} nutrition"
        
        params = {
            "engine": "google_light",
            "q": search_query,
            "location": "United States",
            "hl": "en",
            "gl": "us",
            "num": num,
            "api_key": api_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        alternatives = []
        if "organic_results" in results:
            for result in results["organic_results"][:num]:
                alternatives.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                })
        
        return json.dumps({
            "product": product_name,
            "alternatives_search": alternatives,
            "dietary_restrictions": dietary_restrictions
        })
        
    except Exception as e:
        return json.dumps({"error": f"Alternatives search failed: {str(e)}"})

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
