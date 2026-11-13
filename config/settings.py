import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


def configured_path(variable: str, default: str) -> Path:
    path = Path(os.getenv(variable, default))
    return (path if path.is_absolute() else BASE_DIR / path).resolve()


DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = configured_path('UPLOAD_FOLDER', 'uploads')
LOG_DIR = BASE_DIR / 'logs'
VECTOR_STORE_DIR = configured_path('VECTOR_STORE_PATH', 'vector_store')
for directory in (DATA_DIR, UPLOAD_DIR, LOG_DIR, VECTOR_STORE_DIR):
    directory.mkdir(exist_ok=True)

AGGREGATOR_HOST = os.getenv('AGGREGATOR_HOST', '127.0.0.1')
AGGREGATOR_PORT = int(os.getenv('AGGREGATOR_PORT', '8080'))
AGGREGATOR_URL = os.getenv('AGGREGATOR_URL',
                           f'http://{AGGREGATOR_HOST}:{AGGREGATOR_PORT}/mcp')
WEB_SEARCH_MCP_URL = os.getenv('WEB_SEARCH_MCP_URL', 'https://mcp.tavily.com/mcp/')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
SECRET_KEY = os.getenv('SECRET_KEY', '').strip()


def resolve_file(filename: str, extensions: set[str]) -> Path:
    """Only permit plain filenames within the application's data directories."""
    if not filename or Path(filename).name != filename or '/' in filename or '\\' in filename:
        raise ValueError('Use a filename from list_files, not a path.')
    for root in (UPLOAD_DIR, DATA_DIR):
        candidate = (root / filename).resolve()
        if candidate.parent == root.resolve() and candidate.is_file():
            if candidate.suffix.lower() not in extensions:
                raise ValueError('Unsupported file type.')
            return candidate
    raise ValueError(f'File not found: {filename}')


def list_files(extensions: set[str]) -> list[str]:
    return sorted({p.name for root in (DATA_DIR, UPLOAD_DIR) for p in root.iterdir()
                   if p.is_file() and not p.is_symlink() and p.suffix.lower() in extensions})
