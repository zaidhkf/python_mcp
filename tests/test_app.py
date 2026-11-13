import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from langgraph.errors import GraphRecursionError

from app import create_app
from config.settings import UPLOAD_DIR, resolve_file
from pandas_mcp.analytics import analyze, load_frame


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app({'TESTING': True, 'SECRET_KEY': 'test-secret'}).test_client()

    def test_home_and_validation(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        for body in [None, [], {'message': ''}, {'message': 42}]:
            self.assertEqual(self.client.post('/api/chat', json=body).status_code, 400)
        self.assertEqual(self.client.post('/api/upload').status_code, 400)

    def test_chat_dispatch(self):
        with patch('app.ask_agent', new=Mock(return_value=None)), \
                patch('app.run_async', return_value={'answer': 'ok', 'tools_used': []}):
            self.assertEqual(self.client.post('/api/chat', json={'message': 'hello'}).json['answer'], 'ok')

    def test_step_limit_without_results(self):
        with patch('app.ask_agent', new=Mock(return_value=None)), \
                patch('app.run_async', side_effect=GraphRecursionError('limit reached')):
            response = self.client.post('/api/chat', json={'message': 'top five'})
            self.assertEqual(response.status_code, 422)
            self.assertIn('step limit', response.json['error'])

    def test_path_confinement(self):
        for name in ['../.env', '..\\.env', 'C:/Windows/win.ini']:
            with self.assertRaises(ValueError):
                resolve_file(name, {'.csv'})

    def test_pandas_real_dataset(self):
        frame = load_frame('titanic.csv')
        result = analyze('titanic.csv', 'aggregate', 'survived', 'pclass')
        for key, value in frame.groupby('pclass')['survived'].mean().items():
            self.assertAlmostEqual(result['result'][str(key)], value, places=8)
        with self.assertRaises(ValueError):
            analyze('titanic.csv', 'preview', limit=101)

    def test_upload_metadata_and_active_dataset(self):
        response = self.client.post('/api/upload', data={'file': (io.BytesIO(b'Project milestone: launch in December.'), 'evidence.txt')})
        self.assertEqual(response.status_code, 201)
        name = response.json['filename']
        try:
            self.assertEqual(response.json['active_dataset'], name)
            self.assertEqual(response.json['metadata']['chunks'], 1)
            files = self.client.get('/api/files').json
            self.assertEqual(files['active_dataset'], name)
            self.assertIn(name, files['files'])
        finally:
            (UPLOAD_DIR / name).unlink()

    def test_select_active_dataset(self):
        response = self.client.post('/api/active', json={'filename': 'titanic.csv'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['active_dataset'], 'titanic.csv')
        self.assertEqual(self.client.get('/api/files').json['active_dataset'], 'titanic.csv')

    def test_upload_names_and_duplicate_suffixes(self):
        with TemporaryDirectory() as directory:
            uploads = Path(directory) / 'uploads'
            data = Path(directory) / 'data'
            uploads.mkdir()
            data.mkdir()
            with patch('app.UPLOAD_DIR', uploads), patch('app.DATA_DIR', data), \
                    patch('config.settings.UPLOAD_DIR', uploads), patch('config.settings.DATA_DIR', data):
                for index, expected in enumerate(['movies.csv', 'movies_dup1.csv', 'movies_dup2.csv']):
                    content = f'title,revenue\nMovie {index},{index + 1}\n'.encode()
                    response = self.client.post('/api/upload', data={'file': (io.BytesIO(content), 'movies.csv')})
                    self.assertEqual(response.status_code, 201, response.json)
                    self.assertEqual(response.json['filename'], expected)
                    self.assertEqual(response.json['active_dataset'], expected)
                    self.assertEqual((uploads / expected).read_bytes(), content)
                self.assertIn('Movie 0', (uploads / 'movies.csv').read_text())
                self.assertEqual(self.client.get('/api/files').json['active_dataset'], 'movies_dup2.csv')

    def test_upload_does_not_shadow_sample_or_existing_duplicate(self):
        with TemporaryDirectory() as directory:
            uploads = Path(directory) / 'uploads'
            data = Path(directory) / 'data'
            uploads.mkdir()
            data.mkdir()
            original = b'title,revenue\nOriginal,1\n'
            (data / 'movies.csv').write_bytes(original)
            (uploads / 'movies_dup1.csv').write_bytes(original)
            with patch('app.UPLOAD_DIR', uploads), patch('app.DATA_DIR', data), \
                    patch('config.settings.UPLOAD_DIR', uploads), patch('config.settings.DATA_DIR', data):
                response = self.client.post('/api/upload', data={'file': (io.BytesIO(b'title,revenue\nNew,2\n'), 'movies.csv')})
                self.assertEqual(response.status_code, 201, response.json)
                self.assertEqual(response.json['filename'], 'movies_dup2.csv')
                self.assertEqual((data / 'movies.csv').read_bytes(), original)
                self.assertEqual((uploads / 'movies_dup1.csv').read_bytes(), original)

    def test_passengers_by_departure_and_class(self):
        frame = load_frame('titanic.csv')
        result = analyze('titanic.csv', 'aggregate', group_by=['embarked', 'pclass'], metric='count', limit=100)
        self.assertEqual(sum(row['row_count'] for row in result['result']), len(frame))
        expected = frame.groupby(['embarked', 'pclass'], dropna=False).size().reset_index(name='row_count')
        self.assertEqual(result['result'], json.loads(expected.to_json(orient='records')))
        self.assertFalse(result['truncated'])
        limited = analyze('titanic.csv', 'aggregate', group_by=['embarked', 'pclass'], metric='count', limit=2)
        self.assertTrue(limited['truncated'])
        self.assertEqual(len(limited['result']), 2)

    def test_grouped_numeric_aggregation_and_bad_metric(self):
        result = analyze('titanic.csv', 'aggregate', column='fare', group_by=['embarked', 'pclass'])
        self.assertIn('fare_mean', result['result'][0])
        with self.assertRaisesRegex(ValueError, 'numeric column'):
            analyze('titanic.csv', 'aggregate', column='embarked')


if __name__ == '__main__':
    unittest.main()
