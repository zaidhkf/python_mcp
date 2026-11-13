import asyncio
import json
from time import perf_counter

from langchain_core.callbacks import AsyncCallbackHandler


class AgentProgress(AsyncCallbackHandler):
    """Record model/tool timings without logging prompts, arguments or results."""

    def __init__(self, logger):
        self.logger = logger
        self.stage = 'initializing'
        self.started = perf_counter()
        self.active = {}
        self.completed_results = []

    def start(self, run_id, label):
        self.stage = label
        self.active[run_id] = (label, perf_counter())
        self.logger.info('%s started', label)

    def finish(self, run_id, error=None):
        label, started = self.active.pop(run_id, (self.stage, perf_counter()))
        if error is None:
            self.logger.info('%s completed elapsed=%.2fs', label, perf_counter() - started)
            self.stage = 'agent_execution'
        else:
            self.logger.warning('%s failed elapsed=%.2fs error_type=%s',
                                label, perf_counter() - started, type(error).__name__)

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.start(run_id, 'model_request')

    async def on_llm_end(self, response, *, run_id, **kwargs):
        self.finish(run_id)

    async def on_llm_error(self, error, *, run_id, **kwargs):
        self.finish(run_id, error)

    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.start(run_id, f"tool:{(serialized or {}).get('name', 'unknown')}")

    async def on_tool_end(self, output, *, run_id, **kwargs):
        label = self.active.get(run_id, ('tool:unknown', 0))[0]
        if getattr(output, 'status', None) == 'error':
            self.finish(run_id, RuntimeError('Tool returned an error result'))
            return
        if getattr(output, 'status', None) != 'error':
            content = getattr(output, 'content', output)
            if isinstance(content, list):
                content = '\n'.join(item.get('text', '') if isinstance(item, dict) else str(item)
                                    for item in content)
            elif not isinstance(content, str):
                content = json.dumps(content, default=str, ensure_ascii=False)
            self.completed_results.append((label.removeprefix('tool:'), content))
        self.finish(run_id)

    def partial_response(self, reason='The model did not finish its response.'):
        """Return actual completed work only; discovery alone cannot answer a question."""
        results = [(name, content) for name, content in self.completed_results
                   if name.startswith(('pandas_', 'rag_', 'web_'))
                   and name not in {'pandas_list_datasets', 'rag_list_documents'}]
        if not results:
            return None
        warning = (reason + ' Below are completed tool results; '
                   'the full request may be incomplete.')
        sections = []
        for name, content in results:
            try:
                content = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
            except (ValueError, TypeError):
                pass
            if len(content) > 16000:
                content = content[:16000] + '\n[Result shortened for display.]'
            sections.append(f'{name}:\n{content}')
        return {'answer': warning + '\n\n' + '\n\n'.join(sections),
                'tools_used': [name for name, _ in self.completed_results],
                'results': self.structured_results(),
                'partial': True, 'warning': warning}

    def structured_results(self):
        """Expose JSON tool results for optional table rendering in Flask."""
        results = []
        for name, content in self.completed_results:
            try:
                data = json.loads(content)
            except (ValueError, TypeError):
                continue
            if isinstance(data, (dict, list)):
                results.append({'tool': name, 'data': data})
        return results

    def ranking_answer(self):
        """Render pure ranking answers from computed rows, retaining every currency."""
        if any(name not in {'pandas_list_datasets', 'pandas_inspect_dataset', 'pandas_rank_dataset'}
               for name, _ in self.completed_results):
            return None
        sections, seen = [], set()
        for name, content in self.completed_results:
            if name != 'pandas_rank_dataset':
                continue
            try:
                data = json.loads(content)
            except (ValueError, TypeError):
                continue
            if not isinstance(data, dict) or 'sort_by' not in data:
                continue
            signature = json.dumps(data, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            groups = data.get('rankings_by_currency', {'all rows': data})
            for currency, group in groups.items():
                rows = group.get('result', [])
                heading = (f"{'Lowest' if data['ascending'] else 'Highest'} {len(rows)} by "
                           f"{', '.join(data['sort_by'])} — {currency}")
                lines = [heading, f"Source: {data['filename']}"]
                if rows:
                    fields = list(dict.fromkeys(([c for c in ['title'] if c in rows[0]]) + data['sort_by'] +
                                  [c for c in ['unit', 'currency'] if c in rows[0]]))
                    def cell(value):
                        return str(value if value is not None else 'missing').replace('|', '/').replace('\n', ' ')
                    lines += [' | '.join(fields), ' | '.join(['---'] * len(fields))]
                    lines += [' | '.join(cell(row.get(field)) for field in fields) for row in rows]
                else:
                    lines.append('No eligible rows.')
                lines.append(data['ranking_rule'])
                if data.get('note'):
                    lines.append(data['note'])
                sections.append('\n'.join(lines))
        return '\n\n'.join(sections) if sections else None

    async def on_tool_error(self, error, *, run_id, **kwargs):
        self.finish(run_id, error)

    def describe(self):
        if self.active:
            return ', '.join(f'{name} ({perf_counter() - start:.1f}s)'
                             for name, start in self.active.values())
        return self.stage

    async def heartbeat(self):
        while True:
            await asyncio.sleep(15)
            self.logger.info('Agent still running elapsed=%.1fs waiting_on=%s',
                             perf_counter() - self.started, self.describe())
