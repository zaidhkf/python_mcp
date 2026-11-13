from pathlib import Path

import pandas as pd

from data_manager.dataset_registry import apply_column_aliases
from data_manager.models import HeaderDetection, TableResult

SCAN_ROWS = 20
MAX_HEADER_DEPTH = 3
EXTENSIONS = {'.csv', '.xlsx'}


def clean_cell(value) -> str | None:
    """Return a usable header label or None for blank-like values."""
    if pd.isna(value):
        return None
    text = ' '.join(str(value).split())
    return text if text and text.lower() not in {'nan', 'none', 'null'} else None


def numeric_ratio(frame: pd.DataFrame) -> float:
    """Return the share of populated cells that can be interpreted as numbers."""
    values = [value for value in frame.to_numpy().ravel() if not pd.isna(value)]
    if not values:
        return 0.0
    numeric = 0
    for value in values:
        try:
            float(str(value).replace(',', '').strip())
            numeric += 1
        except (TypeError, ValueError):
            pass
    return numeric / len(values)


def make_unique(names: list[str]) -> list[str]:
    """Create stable names for blank and duplicate columns."""
    counts = {}
    result = []
    for position, name in enumerate(names, start=1):
        name = name or f'column_{position}'
        counts[name] = counts.get(name, 0) + 1
        result.append(name if counts[name] == 1 else f'{name}_{counts[name]}')
    return result


def infer_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore numeric types after reading raw cells for header detection."""
    frame = frame.infer_objects()
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_object_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().sum() == series.notna().sum():
            frame[column] = numeric
    return frame


def header_names(header: pd.DataFrame) -> list[str]:
    """Combine one or more header rows into flattened column labels."""
    reconstructed = header.map(clean_cell).ffill(axis=1).ffill(axis=0)
    names = []
    for column in reconstructed.columns:
        parts = []
        for value in reconstructed[column]:
            text = str(value) if value else ''
            if text and text not in parts:
                parts.append(text)
        names.append(' / '.join(parts))
    return names


def nested_header_bonus(header: pd.DataFrame) -> float:
    """Reward genuine grouped headers without rewarding report titles."""
    if len(header) < 2:
        return 0.0
    bonus = 0.0
    for row_number in range(len(header) - 1):
        parent = header.iloc[row_number].map(clean_cell)
        child = header.iloc[row_number + 1].map(clean_cell)
        parent_count = parent.notna().sum()
        child_count = child.notna().sum()
        if 1 < parent_count < len(parent) and child_count > parent_count:
            bonus += 0.9
    return bonus


def detect_header(raw_frame: pd.DataFrame, scan_rows: int = SCAN_ROWS,
                  max_depth: int = MAX_HEADER_DEPTH) -> tuple[list[str], HeaderDetection]:
    """Detect a single or nested header within the first scan_rows rows."""
    if raw_frame.empty or raw_frame.dropna(how='all').empty:
        raise ValueError('The dataset is empty.')
    if scan_rows < 1 or max_depth < 1:
        raise ValueError('scan_rows and max_depth must be positive.')

    scan_limit = min(scan_rows, len(raw_frame))
    candidates = []
    for start in range(scan_limit):
        for depth in range(1, max_depth + 1):
            end = start + depth
            if end > scan_limit:
                continue
            original = raw_frame.iloc[start:end]
            names = header_names(original)
            valid_names = [name for name in names if name]
            coverage = original.notna().to_numpy().mean()
            uniqueness = len(set(valid_names)) / len(valid_names) if valid_names else 0.0
            header_numeric = numeric_ratio(original)
            following = raw_frame.iloc[end:end + 5]
            data_density = following.notna().to_numpy().mean() if not following.empty else 0.0
            data_numeric = numeric_ratio(following)
            title_rows = sum(len(original.columns) > 2 and row.notna().sum() == 1
                             for _, row in original.iterrows())
            score = (
                2.0 * coverage
                + 2.0 * uniqueness
                + 1.25 * (1 - header_numeric)
                + data_density
                + 0.75 * max(0, data_numeric - header_numeric)
                + nested_header_bonus(original)
                - 2.0 * title_rows
                - 0.15 * (depth - 1)
            )
            candidates.append((score, start, depth, names))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    score, start, depth, names = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    margin = score - second_score
    if score >= 5 and margin >= 0.30:
        confidence = 'high'
    elif score >= 4:
        confidence = 'medium'
    else:
        confidence = 'low'
    detection = HeaderDetection(start, depth, confidence, score, scan_limit)
    return make_unique(names), detection


def parse_table(raw_frame: pd.DataFrame, source_name: str,
                sheet: str | None = None) -> TableResult:
    """Apply detected headers and remove only fully empty rows and columns."""
    columns, detection = detect_header(raw_frame)
    data_start = detection.start_row + detection.depth
    frame = raw_frame.iloc[data_start:].copy()
    frame.columns = columns
    frame = frame.replace(r'^\s*$', pd.NA, regex=True)
    frame = frame.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    frame = infer_types(frame)
    frame.columns = make_unique(apply_column_aliases(
        [str(column) for column in frame.columns], source_name))
    return TableResult(frame, Path(source_name).name, sheet, [], detection)


def read_table(path: Path, sheet: str | None = None) -> TableResult:
    """Read a CSV or Excel table using the shared header detection policy."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in EXTENSIONS:
        raise ValueError('Supported dataset types are CSV and XLSX.')
    try:
        if extension == '.csv':
            raw_frame = pd.read_csv(path, header=None, dtype=object)
            result = parse_table(raw_frame, path.name)
        else:
            with pd.ExcelFile(path) as workbook:
                sheets = workbook.sheet_names
                selected_sheet = sheet if sheet is not None else sheets[0]
                if selected_sheet not in sheets:
                    raise ValueError(f'Unknown sheet: {selected_sheet}. Available sheets: {sheets}')
                raw_frame = pd.read_excel(workbook, sheet_name=selected_sheet,
                                          header=None, dtype=object)
            result = parse_table(raw_frame, path.name, selected_sheet)
            result.sheets = sheets
        return result
    except pd.errors.EmptyDataError as exc:
        raise ValueError('The dataset is empty.') from exc
