"""Exercise the real aggregator transport without paid model or Tavily requests."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update({
    'AGGREGATOR_HOST': '127.0.0.1',
    'AGGREGATOR_PORT': '18080',
    'AGGREGATOR_URL': 'http://127.0.0.1:18080/mcp',
})

from fastmcp import Client

from agent.mcp_client import aggregator_connection
from config.settings import AGGREGATOR_URL, BASE_DIR


async def check():
    async with Client(AGGREGATOR_URL) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        expected = {'pandas_list_datasets', 'rag_list_documents', 'web_tavily_search'}
        assert expected <= names, names

        datasets = await client.call_tool('pandas_list_datasets', {})
        assert not datasets.is_error and 'titanic.csv' in str(datasets), datasets

        analysis = await client.call_tool('pandas_groupby_data', {
            'filename': 'titanic.csv', 'group_by': 'pclass',
            'column': 'survived', 'metric': 'mean',
        })
        assert not analysis.is_error and '0.6296296296' in str(analysis), analysis

        rag_info = await client.call_tool('rag_get_dataset_info', {'filename': 'ProjectPlan.xlsx'})
        assert not rag_info.is_error and 'chunks' in str(rag_info), rag_info

        search = await client.call_tool('web_tavily_search', {'query': 'data analytics MCP'})
        assert not search.is_error and 'https://example.org/evidence' in str(search), search

    async with aggregator_connection() as adapter:
        tools = await adapter.list_tools()
        tool = next(tool for tool in tools if tool.name == 'pandas_list_datasets')
        result = await tool.ainvoke({})
        assert 'titanic.csv' in str(result)

    print(f'PASS: {len(names)} namespaced tools through one HTTP aggregator endpoint.')


if __name__ == '__main__':
    environment = os.environ.copy()
    environment.update({
        'AGGREGATOR_HOST': '127.0.0.1',
        'AGGREGATOR_PORT': '18080',
        'AGGREGATOR_URL': 'http://127.0.0.1:18080/mcp',
        'TAVILY_API_KEY': 'integration-test-key',
        'FAKE_WEB_PORT': '18003',
        'WEB_SEARCH_MCP_URL': 'http://127.0.0.1:18003/mcp',
    })
    processes = []
    try:
        for module in ['tests.fake_web_mcp', 'aggregator.aggregator_server']:
            processes.append(subprocess.Popen(
                [sys.executable, '-m', module], cwd=BASE_DIR, env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        deadline = time.monotonic() + 60
        while True:
            try:
                asyncio.run(asyncio.wait_for(check(), timeout=20))
                break
            except Exception:
                if time.monotonic() >= deadline or any(process.poll() is not None for process in processes):
                    raise
                time.sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
