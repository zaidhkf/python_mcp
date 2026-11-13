import unittest

from pandas_mcp.analytics import column_summary, filter_rows, load_frame


class PandasAnalyticsTests(unittest.TestCase):
    def test_multiple_conditions_filter_rows(self):
        result = filter_rows(
            'titanic.csv',
            conditions=[
                {'column': 'age', 'operator': 'gt', 'value': 50},
                {'column': 'survived', 'operator': 'eq', 'value': 1},
            ],
            columns=['name', 'age', 'survived'],
        )
        expected = load_frame('titanic.csv').query('age > 50 and survived == 1')
        self.assertEqual(result['matched_rows'], len(expected))
        self.assertEqual(set(result['result'][0]), {'name', 'age', 'survived'})

    def test_column_summary(self):
        result = column_summary('titanic.csv', 'age')
        self.assertAlmostEqual(result['summary']['mean'], 29.6991176471, places=8)
        self.assertEqual(result['summary']['missing'], 177)

    def test_filter_validation(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported filter operator'):
            filter_rows('titanic.csv', [{'column': 'age', 'operator': 'execute', 'value': 1}])
        with self.assertRaisesRegex(ValueError, 'requires value to be a list'):
            filter_rows('titanic.csv', [{'column': 'pclass', 'operator': 'in', 'value': 1}])


if __name__ == '__main__':
    unittest.main()
