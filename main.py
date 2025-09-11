"""
Documentation MCP Server - Searches and scrapes LlamaIndex or LangChain documentation using Serper
"""
import os
from typing import Any, List, Dict
import httpx
import asyncio
from urllib.parse import urlparse
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("documentation")

# Constants
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_SCRAPE_URL = "https://scrape.serper.dev"

# Target documentation sites
ALLOWED_DOMAINS = {
    "llamaindex": "docs.llamaindex.ai",
    "langchain": "python.langchain.com"
}

async def search_serper(query: str, library: str, num_results: int = 2) -> List[Dict[str, Any]]:
    """
    Search for documentation using Serper API.
    
    Args:
        query: Search query string
        library: The library to search for ('llamaindex' or 'langchain')
        num_results: Number of results to return (max 2)
    
    Returns:
        List of search results with title, link, and snippet
    """
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY environment variable not set")
    
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Construct search query to target specific documentation site
    site_filter = f"site:{ALLOWED_DOMAINS[library]}"
    search_query = f"({query}) AND ({site_filter})"
    
    payload = {
        "q": search_query,
        "num": min(num_results, 2)  # Serper API typically limits to 2 results
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SERPER_SEARCH_URL,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            organic_results = data.get("organic", [])
            
            for result in organic_results:
                # Verify the result is from the allowed domain
                parsed_url = urlparse(result.get("link", ""))
                if parsed_url.netloc == ALLOWED_DOMAINS[library]:
                    results.append({
                        "title": result.get("title", "No title"),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "No snippet available")
                    })
            
            return results
            
        except httpx.HTTPStatusError as e:
            raise Exception(f"Serper API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Search request failed: {str(e)}")

async def scrape_documentation(url: str) -> str:
    """
    Scrape content from a documentation page using Serper's scraping API.
    
    Args:
        url: URL of the documentation page to scrape
    
    Returns:
        Cleaned text content from the page
    """
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "url": url,
        "includeMarkdown": True
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SERPER_SCRAPE_URL,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Use markdown content if available, otherwise use plain text
            content = data.get("markdown", data.get("text", "No content available"))
            
            # Limit content length to prevent overwhelming responses
            if len(content) > 5000:
                content = content[:5000] + "\n\n... [Content truncated for length]"

            return content
            
        except httpx.HTTPStatusError as e:
            return f"Error scraping {url}: HTTP {e.response.status_code}"
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

@mcp.tool()
async def get_documentation(query: str, library: str, max_results: int = 2) -> str:
    """
    Search for and retrieve documentation from LlamaIndex or LangChain sites.
    
    This tool searches for relevant documentation using Serper API and then scrapes
    the content from the top results to provide comprehensive information.
    
    Args:
        query: Search query for documentation (e.g., "vector store", "chat models", "retrieval")
        library: The library to search for ('llamaindex' or 'langchain')
        max_results: Maximum number of documentation pages to retrieve (1-2, default 2)

    Returns:
        Formatted documentation content from the search results
    """
    try:
        # Validate inputs
        max_results = max(1, min(max_results, 2))
        
        if not query.strip():
            return "Error: Query cannot be empty"
        
        if library not in ALLOWED_DOMAINS:
            return f"Error: Invalid library. Choose either 'llamaindex' or 'langchain'"
        
        # Search for documentation
        print(f"Searching for: {query} in {library} documentation")
        search_results = await search_serper(query, library, max_results)
        
        if not search_results:
            return f"No documentation found for query: '{query}' in {library} documentation"
        
        print(f"Found {len(search_results)} results, scraping content...")
        
        # Scrape content from each result
        documentation_content = []
        
        # Use asyncio.gather to scrape multiple pages concurrently
        scrape_tasks = [scrape_documentation(result["link"]) for result in search_results]
        scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        
        for i, result in enumerate(search_results):
            scraped_content = scraped_contents[i]
            
            if isinstance(scraped_content, Exception):
                content = f"Error scraping content: {str(scraped_content)}"
            else:
                content = scraped_content
            
            doc_section = f"""
## {result['title']}
**URL:** {result['link']}
**Summary:** {result['snippet']}

**Content:**
{content}
"""
            documentation_content.append(doc_section)
        
        # Combine all documentation
        newline = '\n'
        sections_joined = ''.join(['---' + section + newline for section in documentation_content])
        
        final_content = f"""# Documentation Search Results for: "{query}" in {library.capitalize()} Documentation

Found {len(search_results)} relevant documentation pages:

{sections_joined}
---
*Search completed successfully*
"""
        
        return final_content
        
    except Exception as e:
        return f"Error retrieving documentation: {str(e)}"

if __name__ == "__main__":
    # Verify environment setup
    if not SERPER_API_KEY:
        print("WARNING: SERPER_API_KEY environment variable not set!")
        print("Please set your Serper API key")
    
    print("Starting Documentation MCP Server...")
    print("Available libraries:", ", ".join(ALLOWED_DOMAINS.keys()))
    
    # Run the server
    mcp.run(transport='stdio')