from langchain_community.tools import DuckDuckGoSearchRun

class WebSearchTool:
    """
    A simple wrapper for the DuckDuckGo web search tool.
    """
    def __init__(self):
        self.search_engine = DuckDuckGoSearchRun()

    def search(self, query: str) -> str:
        """
        Performs a web search using DuckDuckGo.

        Args:
            query: The search query.

        Returns:
            A string containing the search results.
        """
        try:
            # DuckDuckGoSearchRun typically returns a concise summary.
            # For more control, one might need to use other libraries or APIs.
            results = self.search_engine.run(query)
            return results
        except Exception as e:
            # print(f"Error during web search for query '{query}': {e}") # Optional: log error
            return f"Search failed for query '{query}'. Error: {str(e)}"

# Example usage (optional, for testing the tool directly)
if __name__ == "__main__":
    search_tool = WebSearchTool()
    test_query = "health benefits of turmeric"
    print(f"Searching for: {test_query}")
    results = search_tool.search(test_query)
    print("\nResults:")
    print(results)
