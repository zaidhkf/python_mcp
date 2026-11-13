import os
import unittest
import json
from unittest.mock import patch

from fastmcp import Client, FastMCP
from fastmcp.server import create_proxy
from fastmcp.tools.base import ToolResult

from web_search_mcp.web_search_config import (create_web_search_transport, configure_web_proxy,
                                              TavilyResponseErrors)


class WebSearchTransportTests(unittest.TestCase):
    def test_remote_transport_uses_bearer_token(self):
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test-key'}):
            transport = create_web_search_transport()
        self.assertEqual(transport.url, 'https://mcp.tavily.com/mcp/')
        self.assertEqual(transport.headers, {'Authorization': 'Bearer test-key'})

    def test_missing_key_uses_keyless_access(self):
        with patch.dict(os.environ, {'TAVILY_API_KEY': ''}):
            transport = create_web_search_transport()
        self.assertEqual(transport.headers, {'X-Tavily-Access-Mode': 'keyless'})


class WebSearchProxyTests(unittest.IsolatedAsyncioTestCase):
    def build_proxy(self, payload, key='', text_only=False):
        remote = FastMCP('Test remote')

        def search(query: str):
            if text_only:
                return ToolResult(content=json.dumps(payload))
            return payload

        for name in ['tavily_search', 'tavily_extract', 'tavily_crawl', 'tavily_map', 'tavily_research']:
            remote.tool(search, name=name)
        root = FastMCP('Test aggregator')
        root.add_middleware(TavilyResponseErrors())
        with patch.dict(os.environ, {'TAVILY_API_KEY': key}):
            root.mount(configure_web_proxy(create_proxy(Client(remote))), namespace='web')
        return root

    async def test_keyless_catalog_excludes_key_required_tools(self):
        async with Client(self.build_proxy({'results': []})) as client:
            names = {tool.name for tool in await client.list_tools()}
        self.assertEqual(names, {'web_tavily_search', 'web_tavily_extract'})

    async def test_keyed_catalog_retains_advanced_tools(self):
        async with Client(self.build_proxy({'results': []}, key='test-key')) as client:
            names = {tool.name for tool in await client.list_tools()}
        self.assertIn('web_tavily_research', names)

    async def test_real_quota_payload_becomes_error_in_both_encodings(self):
        payload = {'code': 'daily_cap_reached', 'message': 'Limit reached',
                   'retry_after_seconds': 979, 'next_actions': [{'instruction': 'Pay via x402'}]}
        for text_only in [False, True]:
            async with Client(self.build_proxy(payload, text_only=text_only)) as client:
                result = await client.call_tool('web_tavily_search', {'query': 'SOLAS'}, raise_on_error=False)
            self.assertTrue(result.is_error)
            message = result.content[0].text
            self.assertIn('daily_cap_reached', message)
            self.assertIn('979 seconds', message)
            self.assertNotIn('x402', message)

    async def test_successful_sources_and_empty_results_are_unchanged(self):
        for results in [[], [{'url': 'https://example.org', 'content': 'Evidence'}]]:
            async with Client(self.build_proxy({'results': results})) as client:
                result = await client.call_tool('web_tavily_search', {'query': 'SOLAS'})
            self.assertFalse(result.is_error)
            self.assertEqual(json.loads(result.content[0].text)['results'], results)

    async def test_authentication_error_is_not_evidence(self):
        async with Client(self.build_proxy({'code': 'unauthorized', 'message': 'Invalid key'})) as client:
            result = await client.call_tool('web_tavily_search', {'query': 'SOLAS'}, raise_on_error=False)
        self.assertTrue(result.is_error)
        self.assertIn('unauthorized', result.content[0].text)


if __name__ == '__main__':
    unittest.main()
