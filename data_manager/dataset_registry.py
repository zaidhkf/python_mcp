from pathlib import Path

# Some source workbooks contain meaningful data columns with a blank or repeated
# heading. Keep those source-specific corrections explicit and reviewable here.
COLUMN_ALIASES = {
    'projectplan.xlsx': {
        'Type_2': 'Type Detail',
    },
}


def apply_column_aliases(columns: list[str], source_name: str) -> list[str]:
    """Apply known source-specific column labels without changing cell values."""
    aliases = COLUMN_ALIASES.get(Path(source_name).name.lower(), {})
    return [aliases.get(column, column) for column in columns]
