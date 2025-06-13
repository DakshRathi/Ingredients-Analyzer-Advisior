# src/mcp_servers/website_content_server.py
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WebsiteContentRetriever")

@mcp.tool()
def fetch_website_content(url: str) -> str:
    """
    Fetches the main text content from a given website URL.
    Use this tool to get detailed information from a webpage after an initial web search has provided a promising link.
    The input must be a valid URL starting with http or https.
    """
    if not url.startswith(('http://', 'https://')):
        return "Error: Invalid URL. It must start with http:// or https://."
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')

        # Remove script and style elements
        for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
            script_or_style.decompose()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        # Return a manageable chunk of the most relevant text
        return clean_text[:4000]

    except Exception as e:
        return f"Error fetching or parsing content from {url}: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="sse")
