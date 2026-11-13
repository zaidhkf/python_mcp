import asyncio
from contextlib import AsyncExitStack, suppress
import logging
import os
import sys
from time import perf_counter

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from openai import APITimeoutError, DefaultAsyncHttpxClient

from agent.mcp_client import aggregator_connection
from agent.diagnostics import AgentProgress
from config.logging_config import LOGGER_NAME
from config.prompts import SYSTEM_PROMPT

AGENT_TIMEOUT_SECONDS = int(os.getenv('AGENT_TIMEOUT_SECONDS', '120'))


class AgentTimeoutError(TimeoutError):
    def __init__(self, stage):
        self.stage = stage
        super().__init__(f'Agent timed out during {stage}')


async def ask_agent(message: str, request_id: str = '-', active_dataset: str | None = None) -> dict:
    logger = logging.LoggerAdapter(logging.getLogger(LOGGER_NAME), {'request_id': request_id})
    started = perf_counter()
    key = os.getenv('OPENROUTER_API_KEY')
    if not key:
        raise ValueError('Set OPENROUTER_API_KEY in .env before chatting.')
    model_name = os.getenv('MODEL', 'openai/gpt-4o-mini')
    logger.info('Agent starting model=%s total_timeout=%ss model_timeout=60s max_retries=0 interpreter=%s',
                model_name, AGENT_TIMEOUT_SECONDS, sys.executable)
    progress = AgentProgress(logger)
    heartbeat = asyncio.create_task(progress.heartbeat())
    try:
      async with AsyncExitStack() as resources, asyncio.timeout(AGENT_TIMEOUT_SECONDS):
        progress.stage = 'model_initialization'
        # Flask creates a new event loop per request. Do not reuse LangChain's
        # cached async transport across loops; close this one before loop exit.
        http_client = await resources.enter_async_context(DefaultAsyncHttpxClient(timeout=60))
        extra_body = {}
        if os.getenv('OPENROUTER_REASONING_ENABLED', '').lower() == 'false':
            extra_body['reasoning'] = {'enabled': False}
        model = ChatOpenAI(model=model_name, api_key=key,
                       base_url='https://openrouter.ai/api/v1', timeout=60, max_retries=0,
                       max_tokens=2048, extra_body=extra_body, http_async_client=http_client)
        progress.stage = 'mcp_connection'
        logger.info('MCP connection opening (aggregator only)')
        async with aggregator_connection() as adapter:
            logger.info('MCP connection opened elapsed=%.2fs', perf_counter() - started)
            progress.stage = 'mcp_tool_discovery'
            discovery_started = perf_counter()
            logger.info('MCP tool discovery started')
            tools = await adapter.list_tools()
            logger.info('MCP tool discovery completed count=%d elapsed=%.2fs tools=%s',
                        len(tools), perf_counter() - discovery_started,
                        ', '.join(tool.name for tool in tools))
            progress.stage = 'agent_creation'
            dataset_context = (f'\nThe active dataset for this session is {active_dataset}. '
                               'Use this exact filename unless the user explicitly names another source.'
                               if active_dataset else
                               '\nNo active dataset is selected. Ask the user to upload or select one when needed.')
            agent = create_agent(model=model, tools=tools,
                                 system_prompt=SYSTEM_PROMPT + dataset_context)
            progress.stage = 'agent_execution'
            logger.info('Agent execution started')
            response = await agent.ainvoke({'messages': [{'role': 'user', 'content': message}]},
                                          config={'recursion_limit': 20, 'callbacks': [progress]})
            progress.stage = 'mcp_connection_cleanup'
            logger.info('Agent execution completed; closing MCP connection')
        logger.info('MCP connection closed')
    except (TimeoutError, APITimeoutError) as exc:
        logger.exception('Agent timed out elapsed=%.2fs timeout=%ss waiting_on=%s',
                         perf_counter() - started, AGENT_TIMEOUT_SECONDS, progress.describe())
        partial = progress.partial_response()
        if partial:
            logger.warning('Returning completed tool results after model timeout')
            return partial
        raise AgentTimeoutError(progress.stage) from exc
    except GraphRecursionError:
        logger.warning('Agent reached its step limit; returning any completed analysis')
        partial = progress.partial_response('The agent reached its step limit before completing the answer.')
        if partial:
            return partial
        raise
    except Exception:
        logger.exception('Agent failed elapsed=%.2fs stage=%s', perf_counter() - started, progress.stage)
        raise
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
    final = response['messages'][-1]
    content = final.content
    if isinstance(content, list):
        content = '\n'.join(block.get('text', '') if isinstance(block, dict) else str(block)
                            for block in content)
    content = progress.ranking_answer() or content
    used = [call['name'] for item in response['messages'] for call in getattr(item, 'tool_calls', [])]
    logger.info('Agent completed elapsed=%.2fs tool_calls=%d answer_characters=%d',
                perf_counter() - started, len(used), len(content))
    return {'answer': content, 'tools_used': used,
            'results': progress.structured_results(), 'active_dataset': active_dataset}
