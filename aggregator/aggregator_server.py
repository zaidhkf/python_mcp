import sys

from fastmcp import FastMCP
from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy

from config.settings import AGGREGATOR_HOST, AGGREGATOR_PORT, BASE_DIR, LOG_DIR
from web_search_mcp.web_search_config import create_web_search_transport, configure_web_proxy, TavilyResponseErrors

mcp = FastMCP('Data Analytics Aggregator')
mcp.add_middleware(TavilyResponseErrors())


def local_transport(module: str, log_name: str) -> StdioTransport:
    """Launch a project-local MCP server as a managed STDIO subprocess."""
    return StdioTransport(
        command=sys.executable,
        args=['-m', module],
        cwd=str(BASE_DIR),
        keep_alive=True,
        log_file=LOG_DIR / log_name,
    )


mcp.mount(create_proxy(local_transport('pandas_mcp.pandas_server', 'pandas-mcp.log'),
                       name='pandas proxy'), namespace='pandas')
mcp.mount(create_proxy(local_transport('rag_mcp.rag_server', 'rag-mcp.log'),
                       name='rag proxy'), namespace='rag')
mcp.mount(configure_web_proxy(create_proxy(create_web_search_transport(), name='web proxy')), namespace='web')


if __name__ == '__main__':
    mcp.run(transport='http', host=AGGREGATOR_HOST, port=AGGREGATOR_PORT)
