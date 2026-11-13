import unittest
from unittest.mock import Mock, patch

from app import connection_summary, create_app


class SidebarStatusTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app({'TESTING': True, 'SECRET_KEY': 'test'}).test_client()

    def test_stale_filename_cleared_when_file_list_is_empty(self):
        with self.client.session_transaction() as session:
            session['active_dataset'] = 'titanic.csv'
        with patch('app.list_files', return_value=[]):
            result = self.client.get('/api/files').json
        self.assertEqual(result['files'], [])
        self.assertIsNone(result['active_dataset'])
        with self.client.session_transaction() as session:
            self.assertNotIn('active_dataset', session)

    def test_valid_selection_is_preserved(self):
        with self.client.session_transaction() as session:
            session['active_dataset'] = 'movies.csv'
        with patch('app.list_files', return_value=['movies.csv']):
            self.assertEqual(self.client.get('/api/files').json['active_dataset'], 'movies.csv')

    def test_chat_does_not_receive_deleted_filename(self):
        with self.client.session_transaction() as session:
            session['active_dataset'] = 'titanic.csv'
        with patch('app.list_files', return_value=[]), \
                patch('app.ask_agent', new=Mock(return_value=None)) as agent, \
                patch('app.run_async', return_value={'answer': 'ok'}):
            self.assertEqual(self.client.post('/api/chat', json={'message': 'hello'}).status_code, 200)
        self.assertIsNone(agent.call_args.kwargs['active_dataset'])

    def test_connection_count_is_services_not_tools(self):
        result = connection_summary(['pandas_a', 'pandas_b', 'rag_a', 'web_a', 'web_b'])
        self.assertEqual(result['connected'], 3)
        self.assertEqual(result['total'], 3)

    def test_partial_connection_status(self):
        with patch('app.list_tools', new=Mock(return_value=None)), \
                patch('app.asyncio.wait_for', new=Mock(return_value=None)), \
                patch('app.run_async', return_value=['pandas_a', 'rag_a']):
            response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'partial')
        self.assertEqual(response.json['connected'], 2)
        self.assertFalse(response.json['connections'][2]['connected'])
