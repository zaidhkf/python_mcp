import os
import json

from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult

from config.settings import WEB_SEARCH_MCP_URL

KEY_REQUIRED_TOOLS = {'tavily_crawl', 'tavily_map', 'tavily_research'}


def configure_web_proxy(proxy):
    """Do not advertise tools that keyless Tavily cannot execute."""
    if not os.getenv('TAVILY_API_KEY', '').strip():
        proxy.disable(names=KEY_REQUIRED_TOOLS, components={'tool'})
    return proxy


def tavily_failure(data):
    if not isinstance(data, dict):
        return None
    code = data.get('code')
    if code in {'daily_cap_reached', 'rate_limit_exceeded', 'usage_limit_exceeded'}:
        delay = data.get('retry_after_seconds')
        retry = f' Retry after {delay} seconds.' if isinstance(delay, int) and delay >= 0 else ' Retry later.'
        return (f'Tavily search quota reached ({code}).' + retry +
                ' Configure TAVILY_API_KEY in .env and restart the aggregator to use account limits.'
                ' No web sources were retrieved from this call. Do not repeat it during the retry period.')
    if data.get('error') or (code and data.get('message') and 'results' not in data):
        return (f'Tavily could not complete this request ({code or "service_error"}). '
                'Check Tavily credentials, access permissions and account limits. '
                'No web sources were retrieved from this call.')
    return None


class TavilyResponseErrors(Middleware):
    """Tavily can return quota JSON with is_error=False; normalize that boundary."""
    async def on_call_tool(self, context, call_next):
        result = await call_next(context)
        if not context.message.name.startswith(('web_', 'tavily_')) or result.is_error:
            return result
        payloads = [result.structured_content]
        for block in result.content:
            try:
                payloads.append(json.loads(getattr(block, 'text', '')))
            except (ValueError, TypeError):
                pass
        for payload in payloads:
            failure = tavily_failure(payload)
            if failure:
                return ToolResult(content=failure, is_error=True)
        return result


def create_web_search_transport() -> StreamableHttpTransport:
    """Connect with a configured key or Tavily's limited keyless access."""
    key = os.getenv('TAVILY_API_KEY', '').strip()
    headers = ({'Authorization': f'Bearer {key}'} if key else
               {'X-Tavily-Access-Mode': 'keyless'})
    return StreamableHttpTransport(
        WEB_SEARCH_MCP_URL,
        headers=headers,
    )
