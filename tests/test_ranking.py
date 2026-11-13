import unittest
import json
import logging
from unittest.mock import patch

import pandas as pd

from pandas_mcp.analytics import rank_rows
from agent.diagnostics import AgentProgress


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'title': ['A', 'B', 'C', 'D'], 'revenue': [900, 2, 5000, 1],
            'unit': ['Millions', 'Billions', 'Millions', 'Millions'],
            'currency': ['USD', 'USD', 'INR', 'USD'], 'rating': [8, 9, 7, None],
        })

    def rank(self, sort_by, **kwargs):
        with patch('pandas_mcp.analytics.load_frame', return_value=self.frame.copy()):
            return rank_rows('sample.csv', sort_by, **kwargs)

    def test_monetary_units_and_currencies(self):
        result = self.rank('revenue', columns=['title'])['rankings_by_currency']
        self.assertEqual(result['USD']['result'][0]['title'], 'B')
        self.assertEqual(result['USD']['result'][0]['normalized_revenue'], 2000000000)
        self.assertEqual(result['INR']['result'][0]['title'], 'C')

    def test_answer_retains_all_currencies_and_ratings(self):
        progress = AgentProgress(logging.getLogger('ranking-tests'))
        revenue = self.rank('revenue', columns=['title'])
        ratings = self.rank('rating', columns=['title'])
        progress.completed_results = [('pandas_rank_dataset', json.dumps(data)) for data in [revenue, ratings, revenue]]
        answer = progress.ranking_answer()
        self.assertIn('USD', answer)
        self.assertIn('INR', answer)
        self.assertIn('rating', answer)
        self.assertEqual(answer.count('Monetary units normalized'), 2)

    def test_failed_ranking_is_not_rendered(self):
        progress = AgentProgress(logging.getLogger('ranking-tests'))
        progress.completed_results = [('pandas_rank_dataset', 'File not found')]
        self.assertIsNone(progress.ranking_answer())

    def test_rating_and_missing_values(self):
        result = self.rank('rating', limit=2, columns=['title'])
        self.assertEqual([row['title'] for row in result['result']], ['B', 'A'])
        self.assertEqual(result['eligible_rows'], 3)

    def test_filters_and_ascending(self):
        result = self.rank('rating', filters={'currency': 'USD'}, ascending=True)
        self.assertEqual([row['title'] for row in result['result']], ['A', 'B'])

    def test_multiple_keys_break_ties(self):
        self.frame['rating'] = [8, 8, 7, 6]
        result = self.rank(['rating', 'revenue'])['rankings_by_currency']['USD']
        self.assertEqual([row['title'] for row in result['result']], ['B', 'A', 'D'])

    def test_validation(self):
        for key, kwargs in [('unknown', {}), ('title', {}), ('rating', {'limit': 0}), ([], {})]:
            with self.assertRaises(ValueError):
                self.rank(key, **kwargs)
        self.frame.loc[0, 'unit'] = 'unrecognized'
        with self.assertRaisesRegex(ValueError, 'known unit'):
            self.rank('revenue')
