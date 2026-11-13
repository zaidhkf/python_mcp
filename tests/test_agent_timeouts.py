import logging
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from agent.ai_agent import AgentTimeoutError, ask_agent
from agent.diagnostics import AgentProgress


class TimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_limit_keeps_completed_ranking(self):
        adapter = AsyncMock()
        adapter.list_tools.return_value = []
        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=adapter)
        connection.__aexit__ = AsyncMock(return_value=False)

        async def invoke(payload, config):
            progress = config['callbacks'][0]
            await progress.on_tool_start({'name': 'pandas_rank_dataset'}, '', run_id='rank')
            await progress.on_tool_end(ToolMessage(content='{"result":[{"title":"A","rating":9}]}',
                                                  tool_call_id='rank'), run_id='rank')
            raise GraphRecursionError('limit reached')

        agent = MagicMock()
        agent.ainvoke = AsyncMock(side_effect=invoke)
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}), \
                patch('agent.ai_agent.ChatOpenAI'), \
                patch('agent.ai_agent.aggregator_connection', return_value=connection), \
                patch('agent.ai_agent.create_agent', return_value=agent):
            result = await ask_agent('top five')
        self.assertTrue(result['partial'])
        self.assertIn('step limit', result['answer'])
        self.assertIn('rating', result['answer'])

    async def test_completed_analysis_survives_final_model_timeout(self):
        adapter = AsyncMock()
        adapter.list_tools.return_value = []
        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=adapter)
        connection.__aexit__ = AsyncMock(return_value=False)

        async def invoke(payload, config):
            progress = config['callbacks'][0]
            await progress.on_tool_start({'name': 'pandas_analyze_dataset'}, '', run_id='analysis')
            await progress.on_tool_end(ToolMessage(
                content='{"result":[{"embarked":"C","pclass":1,"row_count":85}]}',
                tool_call_id='count'), run_id='analysis')
            progress.stage = 'model_request'
            raise TimeoutError()

        agent = MagicMock()
        agent.ainvoke = AsyncMock(side_effect=invoke)
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}), \
                patch('agent.ai_agent.ChatOpenAI'), \
                patch('agent.ai_agent.aggregator_connection', return_value=connection), \
                patch('agent.ai_agent.create_agent', return_value=agent):
            result = await ask_agent('count passengers')
        self.assertTrue(result['partial'])
        self.assertIn('85', result['answer'])
        self.assertIn('may be incomplete', result['answer'])

    async def test_discovery_and_failed_tools_do_not_count_as_analysis(self):
        progress = AgentProgress(logging.getLogger('timeout-tests'))
        await progress.on_tool_start({'name': 'pandas_list_datasets'}, '', run_id='list')
        await progress.on_tool_end('titanic.csv', run_id='list')
        await progress.on_tool_start({'name': 'pandas_analyze_dataset'}, '', run_id='analysis')
        await progress.on_tool_end(ToolMessage(content='failed', status='error', tool_call_id='bad'), run_id='analysis')
        self.assertIsNone(progress.partial_response())

    async def test_provider_timeout_without_results_is_not_reported_as_success(self):
        connection = MagicMock()
        connection.__aenter__ = AsyncMock(side_effect=TimeoutError())
        connection.__aexit__ = AsyncMock(return_value=False)
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}), \
                patch('agent.ai_agent.ChatOpenAI'), \
                patch('agent.ai_agent.aggregator_connection', return_value=connection):
            with self.assertRaises(AgentTimeoutError):
                await ask_agent('count passengers')
