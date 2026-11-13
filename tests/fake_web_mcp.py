import os

from fastmcp import FastMCP

mcp = FastMCP('Fake Tavily MCP')


@mcp.tool
def tavily_search(query: str, max_results: int = 5) -> dict:
    """Return deterministic web evidence for the integration check."""
    return {'query': query, 'results': [{
        'title': 'Integration evidence',
        'url': 'https://example.org/evidence',
        'content': 'Remote MCP web search is connected.',
    }][:max_results]}


if __name__ == '__main__':
    mcp.run(transport='http', host='127.0.0.1',
            port=int(os.getenv('FAKE_WEB_PORT', '18003')), show_banner=False)
