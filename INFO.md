# Running & Testing the Application

Start the app with **either** Option A or Option B below — both serve the API on
`http://localhost:8000`, so the [curl commands](#3-exercise-the-endpoints) are identical for each.

Interactive docs are available at <http://localhost:8000/docs> once the app is up.

## 1. Start the app

### Option A — Docker Compose (recommended)

Starts both the API and the Qdrant vector database. Requires only Docker.

```bash
# Only needed for /api/ask; every other endpoint works without it
echo "OPENAI_API_KEY=sk-..." > .env

docker compose up -d --build --wait
```

### Option B — Local virtual environment

Requires **Python 3.12** and Tesseract installed on the host.

> **Python 3.12 is required, not just recommended.** `pymupdf==1.24.0` and
> `spacy==3.7.5` publish no wheels beyond `cp312`, so on Python 3.13+ pip falls
> back to building MuPDF from source and the install fails.

```bash
# Tesseract: macOS
brew install tesseract
# Tesseract: Debian/Ubuntu
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# Environment and dependencies
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Qdrant — required for /api/ingest and /api/ask
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

<details>
<summary>Using <code>uv</code> instead of pip (optional)</summary>

`uv` reads `.python-version` and provisions Python 3.12 automatically:

```bash
uv sync
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```
</details>

## 2. Run the test suite

Needs neither Qdrant nor an API key — every external dependency is mocked.

```bash
pytest -v          # with uv: uv run pytest -v
```

Expected: **52 passed**.

## 3. Exercise the endpoints

Run these from the repository root; they use the checked-in `sample_pdfs/`.

| Endpoint | Purpose | Needs Qdrant | Needs `OPENAI_API_KEY` |
|---|---|:-:|:-:|
| `GET /health` | Liveness probe | no | no |
| `POST /api/extract` | Names + bounding boxes + fuzzy matching | no | no |
| `POST /api/ingest` | Chunk, embed and store a PDF | yes | no |
| `POST /api/ask` | RAG question answering | yes | yes |

### 3.1 Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### 3.2 Extract names, bounding boxes and fuzzy matches

Send the PDF plus a JSON array of names to match against it.

```bash
curl -X POST http://localhost:8000/api/extract \
  -F "pdf_file=@sample_pdfs/company_memo.pdf;type=application/pdf" \
  -F 'names=[{"first_name":"Margaret","last_name":"Thompson"},
             {"first_name":"Robert","last_name":"Chen"},
             {"first_name":"James","last_name":"Wilson"}]'
```

Returns one `extracted_names` entry per occurrence found in the document
(8 for this memo), each with its bounding box in PDF points, plus the fuzzy
matches — abridged here:

```json
{
  "extracted_names": [
    {
      "name": "Margaret Thompson",
      "bounding_box": {
        "x": 76.8, "y": 98.88,
        "width": 88.32, "height": 9.12,
        "page_number": 0
      }
    }
  ],
  "fuzzy_matches": [
    {"extracted_name": "Margaret Thompson", "matched_name": "Margaret Thompson", "score": 1.0},
    {"extracted_name": "Robert Chen", "matched_name": "Robert Chen", "score": 1.0}
  ]
}
```

Note that `James Wilson` is **absent** from `fuzzy_matches`: no name in the
document reaches the 90% similarity threshold, which is the intended behaviour.

Both validation failures return `400`:

```bash
# Not a PDF -> {"detail":"Invalid file type. File must be a PDF."}
curl -X POST http://localhost:8000/api/extract \
  -F "pdf_file=@README.md;type=text/plain" -F 'names=[]'

# Malformed names -> {"detail":"Invalid names format. Expected a JSON array of ..."}
curl -X POST http://localhost:8000/api/extract \
  -F "pdf_file=@sample_pdfs/company_memo.pdf;type=application/pdf" \
  -F 'names=not-json'
```

### 3.3 Ingest a PDF into the vector store

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "pdf_file=@sample_pdfs/company_memo.pdf;type=application/pdf"
```

```json
{"status": "success", "chunks_stored": 2}
```

Ingestion is idempotent — chunk IDs are content-derived, so re-running the same
command updates the existing points instead of duplicating them.

### 3.4 Ask a question (RAG)

Requires at least one ingested document (step 3.3) and a valid `OPENAI_API_KEY`.

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Who was promoted to Vice President of Engineering?"}'
```

```json
{
  "answer": "Robert Chen was promoted to Vice President of Engineering.",
  "sources": ["MEMORANDUM\n\nTo: All Department Heads\n\nFrom: Margaret Thompson, CEO\n..."]
}
```

When nothing clears the similarity threshold the API answers with a fallback
rather than an error, so an off-topic question is a useful negative test:

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Who is leading the Q3 migration project?"}'
```

```json
{"answer": "No relevant information found.", "sources": []}
```

A missing or invalid `OPENAI_API_KEY` surfaces as `502 {"detail":"Failed to
generate an answer."}`; an unreachable Qdrant surfaces as `503`.
