"""Exercise real async HTTP keep-alive across sequential Flask requests."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_openai import ChatOpenAI

from app import create_app


class ProviderHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        body = json.dumps({
            'id': 'test-completion', 'object': 'chat.completion', 'created': 1,
            'model': 'test-model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'Test provider response'}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, *args):
        pass


class RepeatedChatTests(unittest.TestCase):
    def test_two_requests_close_distinct_http_clients(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), ProviderHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        clients = []

        def model_factory(**kwargs):
            kwargs['base_url'] = f'http://127.0.0.1:{server.server_port}/v1'
            model = ChatOpenAI(**kwargs)
            clients.append(model.http_async_client)
            return model

        def connection_factory():
            adapter = AsyncMock()
            adapter.list_tools.return_value = []
            connection = MagicMock()
            connection.__aenter__ = AsyncMock(return_value=adapter)
            connection.__aexit__ = AsyncMock(return_value=False)
            return connection

        try:
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key', 'MODEL': 'test-model'}), \
                    patch('agent.ai_agent.ChatOpenAI', side_effect=model_factory), \
                    patch('agent.ai_agent.aggregator_connection', side_effect=connection_factory):
                flask = create_app().test_client()
                for question in ['First question', 'Second question']:
                    response = flask.post('/api/chat', json={'message': question})
                    self.assertEqual(response.status_code, 200, response.json)
                    self.assertEqual(response.json['answer'], 'Test provider response')
                    self.assertTrue(clients[-1].is_closed)
            self.assertEqual(len(clients), 2)
            self.assertIsNot(clients[0], clients[1])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
