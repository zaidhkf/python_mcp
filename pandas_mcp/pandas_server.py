from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from config.settings import list_files
from pandas_mcp.analytics import (EXTENSIONS, analyze, column_summary, filter_rows,
                                  load_table, rank_rows)

mcp = FastMCP('Pandas Analytics')


@mcp.tool
def list_datasets() -> list[str]:
    """List available CSV and Excel datasets."""
    return list_files(EXTENSIONS)


@mcp.tool
def inspect_dataset(filename: str, sheet: str | None = None) -> dict:
    """Inspect columns, missing values, row count and Excel sheet names."""
    table = load_table(filename, sheet)
    frame = table.frame
    return {'rows': len(frame), 'columns': {str(c): str(t) for c, t in frame.dtypes.items()},
            'missing': {str(c): int(n) for c, n in frame.isna().sum().items()},
            'sheet': table.sheet, 'sheets': table.sheets, 'header': table.header.to_dict()}


@mcp.tool
def get_dataset_info(filename: str, sheet: str | None = None) -> dict:
    """Return dataset dimensions, columns, data types, missing values and sheets."""
    return inspect_dataset(filename, sheet)


@mcp.tool
def get_column_summary(filename: str, column: str, sheet: str | None = None,
                       limit: int = 10) -> dict:
    """Return descriptive statistics and frequent values for one column."""
    try:
        return column_summary(filename, column, sheet, limit)
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def aggregate_data(filename: str, column: str | None = None,
                   metric: Literal['mean', 'sum', 'min', 'max', 'count', 'median'] = 'count',
                   sheet: str | None = None) -> dict:
    """Apply an aggregate to a column, or count all rows when column is omitted."""
    try:
        return analyze(filename, 'aggregate', column, None, metric, sheet, 100)
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def groupby_data(filename: str, group_by: str | list[str], column: str | None = None,
                 metric: Literal['mean', 'sum', 'min', 'max', 'count', 'median'] = 'count',
                 sheet: str | None = None, limit: int = 100) -> dict:
    """Group rows by one or more columns and apply a validated aggregate."""
    try:
        return analyze(filename, 'aggregate', column, group_by, metric, sheet, limit)
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def filter_dataset(filename: str, conditions: list[dict], columns: list[str] | None = None,
                   match: Literal['all', 'any'] = 'all', sheet: str | None = None,
                   limit: int = 100) -> dict:
    """Filter rows with eq/ne/gt/gte/lt/lte/contains/in/null operators."""
    try:
        return filter_rows(filename, conditions, columns, match, sheet, limit)
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def analyze_dataset(filename: str, operation: Literal['preview', 'describe', 'aggregate', 'value_counts', 'correlation'],
                    column: str | None = None, group_by: str | list[str] | None = None,
                    metric: Literal['mean', 'sum', 'min', 'max', 'count', 'median'] = 'mean',
                    sheet: str | None = None, limit: int = 20) -> dict:
    """Analyze data using preview, describe, aggregate, value_counts or correlation.

    For passenger counts by departure port and class, use operation="aggregate",
    group_by=["embarked", "pclass"], metric="count", and omit column.
    Omit column to count ALL rows, including missing values; specifying column
    counts only nonmissing values in that column. group_by accepts one column name
    or a list of names. Numeric metrics require a numeric column. Inspect column
    names first. limit caps returned groups; truncated indicates omitted groups.
    """
    try:
        return analyze(filename, operation, column, group_by, metric, sheet, limit)
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def rank_dataset(filename: str, sort_by: str | list[str], limit: int = 5,
                 ascending: bool = False, filters: dict[str, str | int | float | bool] | None = None,
                 columns: list[str] | None = None, sheet: str | None = None) -> dict:
    """Return top/bottom rows by numeric columns. Use this for top-five movies.

    For separate revenue and rating lists, call once with sort_by="revenue" and
    once with sort_by="imdb_rating", limit=5. A list means the first column is
    primary and the others only break ties, NOT a combined weighted score.
    ascending=False means highest first. filters applies exact equality, e.g.
    {"currency":"USD"}. columns selects fields to display, e.g. ["title"].
    If revenue/budget have unit and currency columns, units are normalized and
    each currency gets its own ranking. Do not compare different currencies
    without an explicit exchange-rate policy. Missing ranking values are excluded.
    """
    try:
        return rank_rows(filename, sort_by, limit, ascending, filters, columns, sheet)
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


if __name__ == '__main__':
    mcp.run(transport='stdio', show_banner=False)
