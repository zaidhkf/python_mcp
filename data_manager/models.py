from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class HeaderDetection:
    """Describe how the table header was detected in the source sheet."""

    start_row: int
    depth: int
    confidence: Literal['low', 'medium', 'high']
    score: float
    scanned_rows: int

    def to_dict(self) -> dict:
        return {
            'starting_row': self.start_row + 1,
            'header_rows': self.depth,
            'confidence': self.confidence,
            'scanned_rows': self.scanned_rows,
        }


@dataclass
class TableResult:
    """Hold a parsed table together with its source and header metadata."""

    frame: pd.DataFrame
    filename: str
    sheet: str | None
    sheets: list[str]
    header: HeaderDetection

    def metadata(self) -> dict:
        return {
            'filename': self.filename,
            'sheet': self.sheet,
            'sheets': self.sheets,
            'rows': len(self.frame),
            'columns': [str(column) for column in self.frame.columns],
            'header': self.header.to_dict(),
        }
