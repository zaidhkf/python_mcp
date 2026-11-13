from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from config.settings import resolve_file
from data_manager.table_reader import read_table

TABLE_EXTENSIONS = {'.csv', '.xlsx'}
DOCUMENT_EXTENSIONS = {'.txt', '.md', '.pdf'}
EXTENSIONS = TABLE_EXTENSIONS | DOCUMENT_EXTENSIONS


def table_chunks(path: Path, sheet: str | None = None) -> tuple[list[dict], dict]:
    """Represent each non-empty table row as a searchable, traceable chunk."""
    table = read_table(path, sheet)
    chunks = []
    for row_number, (_, row) in enumerate(table.frame.iterrows(), start=1):
        values = [f'{column}: {value}' for column, value in row.items() if not pd.isna(value)]
        if values:
            chunks.append({'filename': path.name, 'chunk': row_number, 'row': row_number,
                           'kind': 'table_row', 'text': ' | '.join(values)})
    return chunks, table.metadata()


def document_chunks(path: Path) -> tuple[list[dict], dict]:
    """Create overlapping text chunks for TXT, Markdown and text-based PDF files."""
    if path.suffix.lower() == '.pdf':
        pages = [page.extract_text() or '' for page in PdfReader(path).pages]
        text = '\n'.join(pages)
        details = {'pages': len(pages)}
    else:
        text = path.read_text(encoding='utf-8-sig')
        details = {}
    chunks = []
    for start in range(0, len(text), 800):
        content = text[start:start + 1000].strip()
        if content:
            chunks.append({'filename': path.name, 'chunk': start // 800 + 1,
                           'kind': 'document', 'text': content})
    return chunks, {'filename': path.name, 'characters': len(text), **details}


def chunk_source(filename: str, sheet: str | None = None) -> tuple[list[dict], dict]:
    """Load one supported source using the shared application file policy."""
    path = resolve_file(filename, EXTENSIONS)
    if path.suffix.lower() in TABLE_EXTENSIONS:
        chunks, metadata = table_chunks(path, sheet)
    else:
        if sheet is not None:
            raise ValueError('sheet is only valid for Excel datasets.')
        chunks, metadata = document_chunks(path)
    return chunks, {'filename': filename, 'type': path.suffix.lower().lstrip('.'),
                    'chunks': len(chunks), **metadata}
