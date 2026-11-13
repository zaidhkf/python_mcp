# MCP Data Analytics AI Agent

## Project overview

This project is a Flask-based data analytics assistant built around the Model
Context Protocol (MCP). A LangChain agent connects to one aggregator endpoint and
selects from three tool groups:

- Pandas tools for exact CSV/XLSX inspection, filtering and calculations.
- RAG tools for semantic retrieval from tables and documents.
- Tavily's official remote MCP server for current web research with source URLs.

Uploaded files become the active dataset for the browser session. The page shows
upload metadata, available files, the natural-language answer, tools used and
structured tool results.

## Architecture

```text
Browser
   │ HTTP/JSON
   ▼
Flask UI (app.py)
   │
   ▼
LangChain agent + OpenRouter model
   │ Streamable HTTP
   ▼
FastMCP aggregator (127.0.0.1:8080/mcp)
   ├── Pandas MCP ── STDIO subprocess
   ├── RAG MCP ───── STDIO subprocess
   └── Tavily MCP ── remote Streamable HTTP
                      https://mcp.tavily.com/mcp/
```

Only the aggregator is exposed as a local MCP HTTP service. It starts and manages
the Pandas and RAG STDIO subprocesses. The Flask agent never connects directly to
an individual tool server.

## Requirement coverage

| Deliverable | Implementation |
| --- | --- |
| Pandas MCP | Dataset info, column summary, aggregate, GroupBy, safe filters, preview, value counts, correlation and ranking |
| RAG MCP | Load/index, semantic search and source metadata for CSV, XLSX, TXT, MD and text-based PDF |
| Web Search MCP | Tavily's official remote MCP over Streamable HTTP with optional Bearer or keyless access |
| Aggregator MCP | One HTTP endpoint with `pandas_*`, `rag_*` and `web_*` namespaced tools |
| AI agent | OpenRouter tool-calling model, MCP discovery, active-dataset context, timeouts and partial-result handling |
| Flask application | Upload, file selection, health check, chat, structured result tables and validation |
| Configuration | `.env`, central paths/settings, rotating logs and reproducible dependency files |
| Tests | Unit tests plus an integration check covering real STDIO/HTTP proxy boundaries |

## Technology stack

- Python 3.13, Flask and vanilla HTML/CSS/JavaScript
- FastMCP 4 and LangChain MCP adapter
- LangChain agent and OpenRouter-compatible chat models
- Pandas, NumPy and OpenPyXL
- Sentence Transformers with `all-MiniLM-L6-v2`
- FAISS `IndexFlatIP` with normalized embeddings for cosine similarity
- Tavily remote MCP
- PyPDF for text-based PDFs

## Main functionality

### File handling

- Accepts CSV, XLSX, TXT, MD and PDF files up to 20 MB.
- Preserves sanitized upload filenames and confines access to `data` and `uploads`.
  Name collisions use suffixes before the extension: `movies.csv`,
  `movies_dup1.csv`, `movies_dup2.csv`. Existing datasets are never overwritten.
- Scans the first 20 spreadsheet rows for the header.
- Detects one to three nested header rows and flattens them into unique labels.
- Preserves incomplete data rows and reports header, sheet, row and column metadata.
- Tracks an active dataset in the Flask session; another listed file can be selected
  without uploading it again.

### Pandas analytics

The Pandas MCP exposes these principal tools through the aggregator:

- `pandas_get_dataset_info`
- `pandas_get_column_summary`
- `pandas_aggregate_data`
- `pandas_groupby_data`
- `pandas_filter_dataset`

Compatibility tools also provide general analysis and numeric ranking. Filters
use a fixed operator allowlist (`eq`, `ne`, comparisons, `contains`, `in` and null
checks); the project does not evaluate model-generated Python or SQL.

### Semantic RAG

Table rows become traceable chunks containing their source filename and row number.
Documents use overlapping text chunks. The RAG server embeds chunks in batches,
normalizes vectors and searches with FAISS cosine similarity. Indexes and chunk
metadata are persisted in `vector_store` and invalidated when the source file or
embedding model changes.

The main exposed tools are `rag_load_dataset`, `rag_search_dataset` and
`rag_get_dataset_info`. The first semantic request downloads the configured
embedding model if it is not already in the Hugging Face cache.

### Remote web search

The aggregator proxies Tavily's official remote MCP endpoint. When
`TAVILY_API_KEY` is set, it is sent as an `Authorization: Bearer ...` header. When
the setting is empty, the client sends `X-Tavily-Access-Mode: keyless` instead.
Keyless search and extraction are rate-limited; advanced tools may still require
an API key. Web tool names are discovered remotely and exposed with the `web_`
namespace.

## Installation

Use PowerShell from the project directory:

```powershell
cd C:\Users\SBHBKFZAI\PycharmProjects\PythonProject\MCP_Data_Analytics_Project
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

For the exact verified package versions, install `requirements.lock.txt` instead.
In PyCharm, select `.venv\Scripts\python.exe` as the project interpreter and use
this project folder as the working directory.

Configure `.env`:

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
MODEL=openai/gpt-4o-mini
TAVILY_API_KEY=your_tavily_key
SECRET_KEY=replace_with_a_long_random_value
```

`TAVILY_API_KEY` is optional for keyless search/extraction. Configure it for your
account limits and the complete Tavily feature set.

Keep `.env` private. `.env.example` documents every supported setting.

## Running the application

Only two terminals are required:

```powershell
# Terminal 1 — aggregator plus its managed MCP connections
.\.venv\Scripts\python.exe -m aggregator.aggregator_server

# Terminal 2 — web application
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), then use **Check MCP
connection**. There is no `run.py`, and the Pandas/RAG modules do not need separate
terminals.

## Example questions

- Inspect `titanic.csv` and summarize missing values.
- Calculate mean survival grouped by passenger class.
- Show passengers older than 50 who survived.
- Load `ProjectPlan.xlsx` for RAG and find rows about financial milestones.
- After uploading a document, retrieve passages about the launch date and cite the
  filename and chunk.
- Search the web for current MCP developments and cite the source URLs.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Render the web interface |
| GET | `/api/files` | List files and the active dataset |
| POST | `/api/active` | Select an existing file |
| POST | `/api/upload` | Upload and inspect a supported file |
| POST | `/api/chat` | Send a question to the agent |
| GET | `/api/health` | Discover tools through the aggregator |

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests\integration_check.py
```

The integration check starts a deterministic fake remote search MCP and the real
aggregator. It verifies the aggregator's HTTP endpoint, managed Pandas/RAG STDIO
servers, remote HTTP proxy boundary and LangChain adapter without consuming Tavily
or OpenRouter credits.

## Expected outcome

With the two processes running and the OpenRouter credentials configured, the user can
upload or select a source, ask analytical questions in natural language and receive
answers grounded in computed Pandas results, semantic retrieval evidence or web
sources. The UI exposes structured results for verification instead of showing only
an untraceable model response.

## Security and limitations

Tavily keyless mode exposes only search and extract; crawl, map and research
require `TAVILY_API_KEY`. Tool discovery in `/api/health` checks connectivity,
not remaining search quota. Tavily may return `daily_cap_reached` inside an MCP
success envelope. The aggregator converts that payload into an error with retry
guidance so it cannot be treated as retrieved evidence. Wait for the reported
retry interval or configure a Tavily API key and restart the aggregator. Account
keys also have limits; this change does not bypass any quota.

- Filenames are validated and arbitrary paths are rejected.
- Tool filters are validated; no `eval`, `exec` or generated-code execution is used.
- File content and web results are treated as untrusted context, not instructions.
- Data used in a model answer is sent to OpenRouter; web queries are sent to Tavily.
- Scanned PDFs require OCR before upload.
- This is a loopback-bound, single-user development application. Add authentication,
  per-user storage and production CSRF/session controls before shared deployment.
- FastMCP and LangChain's MCP integration are evolving APIs; use the lock file for
  reproducible demonstrations.

References: [FastMCP proxy documentation](https://gofastmcp.com/servers/providers/proxy),
[LangChain MCP documentation](https://docs.langchain.com/oss/python/langchain/mcp),
and [Tavily remote MCP server](https://github.com/tavily-ai/tavily-mcp#remote-mcp-server).
