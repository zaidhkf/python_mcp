from langchain.mcp import MCPAdapter

from config.settings import AGGREGATOR_URL


def aggregator_connection() -> MCPAdapter:
    """The agent knows only this endpoint, never the downstream server URLs."""
    return MCPAdapter(AGGREGATOR_URL)


async def list_tools() -> list[str]:
    async with aggregator_connection() as adapter:
        return [tool.name for tool in await adapter.list_tools()]
