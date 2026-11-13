import json
from typing import Literal

import pandas as pd

from config.settings import resolve_file
from data_manager.models import TableResult
from data_manager.table_reader import read_table

EXTENSIONS = {'.csv', '.xlsx'}
FILTER_OPERATORS = {'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'contains',
                    'in', 'is_null', 'not_null'}


def validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for name in columns:
        if name not in frame.columns:
            raise ValueError(f'Unknown column: {name}. Available columns: {list(frame.columns)}')


def filter_rows(filename: str, conditions: list[dict], columns: list[str] | None = None,
                match: Literal['all', 'any'] = 'all', sheet: str | None = None,
                limit: int = 100) -> dict:
    """Filter rows with validated operators; never evaluate supplied code."""
    if not conditions:
        raise ValueError('Provide at least one filter condition.')
    if any(not isinstance(condition, dict) for condition in conditions):
        raise ValueError('Each filter condition must be an object.')
    if match not in {'all', 'any'}:
        raise ValueError('match must be "all" or "any".')
    if not 1 <= limit <= 500:
        raise ValueError('limit must be between 1 and 500.')
    frame = load_frame(filename, sheet)
    condition_columns = [condition.get('column') for condition in conditions]
    if any(not isinstance(column, str) or not column for column in condition_columns):
        raise ValueError('Each filter condition requires a column name.')
    validate_columns(frame, condition_columns + (columns or []))
    masks = []
    for condition in conditions:
        column = condition.get('column')
        operator = condition.get('operator')
        value = condition.get('value')
        if operator not in FILTER_OPERATORS:
            raise ValueError(f'Unsupported filter operator: {operator}. Use one of {sorted(FILTER_OPERATORS)}.')
        series = frame[column]
        if operator == 'eq':
            mask = series.eq(value)
        elif operator == 'ne':
            mask = series.ne(value)
        elif operator == 'gt':
            mask = series.gt(value)
        elif operator == 'gte':
            mask = series.ge(value)
        elif operator == 'lt':
            mask = series.lt(value)
        elif operator == 'lte':
            mask = series.le(value)
        elif operator == 'contains':
            mask = series.astype('string').str.contains(str(value), case=False, na=False, regex=False)
        elif operator == 'in':
            if not isinstance(value, list):
                raise ValueError('The "in" operator requires value to be a list.')
            mask = series.isin(value)
        elif operator == 'is_null':
            mask = series.isna()
        else:
            mask = series.notna()
        masks.append(mask.fillna(False))
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined & mask if match == 'all' else combined | mask
    filtered = frame.loc[combined, columns or list(frame.columns)]
    return {'filename': filename, 'rows': len(frame), 'matched_rows': len(filtered),
            'result': clean(filtered.head(limit), orient='records'),
            'truncated': len(filtered) > limit}


def column_summary(filename: str, column: str, sheet: str | None = None,
                   limit: int = 10) -> dict:
    """Return descriptive statistics and frequent values for one column."""
    frame = load_frame(filename, sheet)
    validate_columns(frame, [column])
    series = frame[column]
    summary = {'dtype': str(series.dtype), 'rows': len(series),
               'non_null': int(series.notna().sum()), 'missing': int(series.isna().sum()),
               'unique': int(series.nunique(dropna=True))}
    if pd.api.types.is_numeric_dtype(series):
        summary.update({key: clean(pd.Series([value]), orient='records')[0] for key, value in {
            'mean': series.mean(), 'median': series.median(), 'std': series.std(),
            'min': series.min(), 'max': series.max(),
        }.items()})
    summary['top_values'] = clean(series.value_counts(dropna=False).head(limit))
    return {'filename': filename, 'column': column, 'summary': summary}


def rank_rows(filename: str, sort_by: str | list[str], limit: int = 5,
              ascending: bool = False, filters: dict | None = None,
              columns: list[str] | None = None, sheet: str | None = None) -> dict:
    """Rank rows, preserving monetary comparability without guessing FX rates."""
    if not 1 <= limit <= 100:
        raise ValueError('limit must be between 1 and 100.')
    frame = load_frame(filename, sheet)
    keys = [sort_by] if isinstance(sort_by, str) else sort_by
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Provide distinct sort_by column names.')
    validate_columns(frame, keys + list(filters or {}) + (columns or []))
    for name, value in (filters or {}).items():
        frame = frame.loc[frame[name].eq(value)]
    frame = frame.copy()
    actual_keys = []
    monetary = False
    for key in keys:
        if not pd.api.types.is_numeric_dtype(frame[key]):
            raise ValueError(f'Ranking requires numeric values; {key} is {frame[key].dtype}.')
        if key.lower() in {'revenue', 'budget'} and {'unit', 'currency'} <= set(frame.columns):
            monetary = True
            units = frame['unit'].astype('string').str.strip().str.lower()
            multipliers = units.map({'units': 1, 'unit': 1, 'thousands': 1000,
                                     'millions': 1000000, 'billions': 1000000000})
            if multipliers.isna().any() or frame['currency'].isna().any():
                raise ValueError('Monetary ranking requires a known unit and currency for every row. Filter to comparable rows first.')
            normalized = f'normalized_{key}'
            frame[normalized] = frame[key] * multipliers
            actual_keys.append(normalized)
        else:
            actual_keys.append(key)
    selected = list(dict.fromkeys((columns or list(frame.columns)) + keys +
                    ([c for c in ['currency', 'unit'] if c in frame.columns] if monetary else []) +
                    [key for key in actual_keys if key not in keys]))

    def ranked(part):
        ordered = part.dropna(subset=actual_keys).sort_values(actual_keys, ascending=ascending, kind='stable')
        return {'eligible_rows': len(ordered), 'result': clean(ordered[selected].head(limit), orient='records'),
                'truncated': len(ordered) > limit}

    response = {'filename': filename, 'sort_by': keys, 'ascending': ascending,
                'ranking_rule': 'Lexicographic: first column is primary; later columns break ties.',
                'rows_after_filter': len(frame)}
    if monetary:
        response['note'] = 'Monetary units normalized to base currency units. Currencies ranked separately; no exchange-rate conversion applied.'
        response['rankings_by_currency'] = {str(currency): ranked(part) for currency, part in frame.groupby('currency')}
    else:
        response.update(ranked(frame))
    return response


def load_frame(filename: str, sheet: str | None = None) -> pd.DataFrame:
    return load_table(filename, sheet).frame


def load_table(filename: str, sheet: str | None = None) -> TableResult:
    """Load a dataset together with its detected header metadata."""
    return read_table(resolve_file(filename, EXTENSIONS), sheet)


def clean(value, orient=None):
    """Return protocol-safe JSON, including dates and missing values."""
    options = {'orient': orient} if orient else {}
    return json.loads(value.to_json(date_format='iso', **options))


def analyze(filename: str, operation: Literal['preview', 'describe', 'aggregate', 'value_counts', 'correlation'],
            column: str | None = None, group_by: str | list[str] | None = None,
            metric: Literal['mean', 'sum', 'min', 'max', 'count', 'median'] = 'mean',
            sheet: str | None = None, limit: int = 20) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError('limit must be between 1 and 100.')
    frame = load_frame(filename, sheet)
    groups = [group_by] if isinstance(group_by, str) else (group_by or [])
    if len(groups) != len(set(groups)):
        raise ValueError('group_by must contain distinct column names.')
    validate_columns(frame, ([column] if column is not None else []) + groups)
    if operation == 'preview':
        result = clean(frame.head(limit))
    elif operation == 'describe':
        result = clean(frame.describe(include='all'))
    elif operation == 'value_counts':
        if column is None:
            raise ValueError('column is required.')
        result = clean(frame[column].value_counts(dropna=False).head(limit))
    elif operation == 'correlation':
        result = clean(frame.select_dtypes(include='number').corr())
    elif operation == 'aggregate':
        if column is None and metric != 'count':
            raise ValueError('column is required for this metric. To count rows, use metric="count" without column.')
        if column is not None and metric in {'mean', 'sum', 'median'} and not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f'{metric} requires a numeric column; {column} is {frame[column].dtype}. Use metric="count" without column to count passengers.')
        grouped = frame.groupby(groups, dropna=False, observed=True) if groups else None
        if column is None:
            # size counts every row, including rows with missing values.
            value = grouped.size() if grouped is not None else len(frame)
        else:
            selected = grouped[column] if grouped is not None else frame[column]
            value = selected.agg(metric)
        if isinstance(value, pd.Series) and (len(groups) > 1 or column is None):
            result_name = 'row_count' if column is None else f'{column}_{metric}'
            while result_name in groups:
                result_name = '_' + result_name
            result = clean(value.rename(result_name).reset_index().head(limit), orient='records')
        else:
            result = clean(value.head(limit)) if isinstance(value, pd.Series) else clean(pd.Series([value]))
        return {'filename': filename, 'rows': len(frame), 'result': result,
                'groups_total': len(value) if isinstance(value, pd.Series) else None,
                'truncated': isinstance(value, pd.Series) and len(value) > limit}
    else:
        raise ValueError('Unsupported operation.')
    return {'filename': filename, 'rows': len(frame), 'result': result}
