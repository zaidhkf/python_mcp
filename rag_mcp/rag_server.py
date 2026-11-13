from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from config.settings import list_files
from rag_mcp.chunking import EXTENSIONS
from rag_mcp.vector_store import load_source, search_documents, search_source, source_info

mcp = FastMCP('Document Retrieval')


@mcp.tool
def list_documents() -> list[str]:
    """List CSV, XLSX, TXT, Markdown and PDF sources available for retrieval."""
    return list_files(EXTENSIONS)


@mcp.tool
def load_dataset(filename: str, sheet: str | None = None) -> dict:
    """Load and semantically index an uploaded dataset or document."""
    try:
        return load_source(filename, sheet)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def search_dataset(query: str, filename: str, top_k: int = 5,
                   sheet: str | None = None, min_score: float = 0.15) -> dict:
    """Retrieve semantically related rows or passages from one source."""
    try:
        return search_source(query, filename, top_k, sheet, min_score)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def get_dataset_info(filename: str, sheet: str | None = None) -> dict:
    """Return source type, chunks, table metadata and index status."""
    try:
        return source_info(filename, sheet)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def retrieve_documents(query: str, filename: str | None = None, top_k: int = 5) -> dict:
    """Retrieve relevant document chunks, including source filenames and chunk numbers."""
    return search_documents(query, filename, top_k)


if __name__ == '__main__':
    mcp.run(transport='stdio', show_banner=False)
