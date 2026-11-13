import unittest

import pandas as pd

from pandas_mcp.analytics import load_table
from data_manager.table_reader import parse_table


class TableReaderTests(unittest.TestCase):
    def test_csv_header_is_detected_on_first_row(self):
        table = load_table('titanic.csv')
        self.assertEqual(table.header.start_row, 0)
        self.assertEqual(table.header.depth, 1)
        self.assertEqual(table.frame.columns[0], 'survived')
        self.assertEqual(len(table.frame), 891)

    def test_project_plan_header_and_known_alias(self):
        table = load_table('ProjectPlan.xlsx')
        self.assertEqual(table.sheet, 'DevPlan')
        self.assertEqual(table.header.start_row, 13)
        self.assertEqual(table.header.depth, 1)
        self.assertEqual(len(table.frame), 22)
        self.assertIn('Type Detail', table.frame.columns)
        self.assertIn('Financial / Non-Financial', table.frame.columns)
        self.assertNotIn('Type_2', table.frame.columns)
        self.assertFalse(any(str(column).startswith('Unnamed:') for column in table.frame.columns))

    def test_unknown_excel_sheet_lists_available_sheets(self):
        with self.assertRaisesRegex(ValueError, 'Available sheets'):
            load_table('ProjectPlan.xlsx', 'Missing')

    def test_nested_header_is_flattened(self):
        raw = pd.DataFrame([
            ['Sales report', None, None, None],
            ['Customer', None, 'Order', None],
            ['Name', 'Region', 'Amount', 'Date'],
            ['Aida', 'North', 100, '2026-01-01'],
            ['Ben', 'South', 80, '2026-01-02'],
        ])
        table = parse_table(raw, 'nested.xlsx', 'Sheet1')
        self.assertEqual(table.header.start_row, 1)
        self.assertEqual(table.header.depth, 2)
        self.assertEqual(list(table.frame.columns), [
            'Customer / Name', 'Customer / Region', 'Order / Amount', 'Order / Date'])

    def test_incomplete_rows_are_preserved(self):
        raw = pd.DataFrame([
            ['Name', 'Age', 'Unused'],
            [None, None, None],
            ['Aida', 30, None],
            ['Ben', None, None],
            [None, None, None],
        ])
        table = parse_table(raw, 'people.csv')
        self.assertEqual(list(table.frame.columns), ['Name', 'Age'])
        self.assertEqual(len(table.frame), 2)
        self.assertTrue(pd.isna(table.frame.loc[1, 'Age']))


if __name__ == '__main__':
    unittest.main()
