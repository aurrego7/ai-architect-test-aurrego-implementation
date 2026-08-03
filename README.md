# ML-TechTest

> **Start here: read [TASK.md](TASK.md) for full instructions.** The repo contains a partially implemented application with bugs and missing features. Your job is to fix, refactor, and complete it.
>
> **To build, run and test the finished application, see [Running & Testing the Application](#running--testing-the-application).**

## Task Description:
The task is to build a system that can extract names and last names from a scanned
PDF document, identify their bounding box locations, and provide an API endpoint to perform fuzzy
matching between the extracted names and the names provided in the request.

Create a Branch and make a pull request to this repository.

## Requirements:

1. Use an open-source OCR library to extract text from the scanned PDF document.
2. Implement a named entity recognition (NER) or a large language model (LLM) approach to
identify and extract names and last names from the extracted text.
3. Parse the extracted text and OCR results to find the bounding box coordinates of the identified
names and last names.
4. Create a RESTful API using FastAPI or Flask that accepts a scanned PDF file and a set of
name-last name pairs as input.
5. Perform fuzzy matching (with a similarity threshold of 90%) between the extracted names and
the provided names in the API request.
6. Return the extracted names, their bounding box coordinates, and the fuzzy matching results as a
JSON response.
7. Implement a vector database (e.g., Chroma, Qdrant) to store the extracted text from the PDF
documents.
8. Use an open-source embedding model (e.g., Sentence-BERT, Universal Sentence Encoder) to
convert the extracted text into vector representations and store them in the vector database.
9. Create another RESTful API endpoint that accepts a question as input and returns an answer
based on the RAG strategy.
10. Create a Docker container image for the entire application, including the text extraction, named
entity recognition, bounding box parsing, vector database, and RAG components. Ensure that
the container can be built and run using Docker commands or a Docker Compose file. Include
instructions for running the containerized application and accessing the API endpoints.
11. Include a DESIGN.md document (max 3 pages) that covers:
    a. **Component diagram** of your solution showing all services, databases, and external dependencies and how they connect (ASCII, Mermaid, or image).
    b. **Sequence diagram** showing the request flow for both API endpoints: (1) PDF upload with name extraction and fuzzy matching, and (2) RAG question-answering (ASCII, Mermaid, or image).
    c. Justify your choice of OCR library, NER approach, embedding model, and vector database.
    d. Explain tradeoffs you considered (accuracy vs speed, cost vs quality).
    e. Describe how you would scale this system to handle 1000+ PDFs per hour.
    f. Identify potential failure modes and how you would mitigate them.
    g. If you were to add LLM routing (selecting different models based on query complexity),
       how would you implement it?

## Test Steps:

1. **Set up the Development Environment:**
   - Create a new Python virtual environment and install the required libraries (e.g., PyMuPDF, Tesseract OCR, spaCy, Hugging Face Transformers, FastAPI or Flask).
   - Obtain a sample scanned PDF document containing names and last names for testing purposes.

2. **Implement Text Extraction from PDF:**
   - Use PyMuPDF or a similar library to extract text from the scanned PDF document.
   - Optionally, you can preprocess the extracted text (e.g., remove newlines, clean up formatting) for better NER or LLM performance.

3. **Implement Named Entity Recognition (NER) or LLM Approach:**
   - Use spaCy's pre-trained NER model or a custom NER model trained on a relevant dataset to identify names and last names in the extracted text.
   - Alternatively, use a pre-trained NER (e.g., BERT, RoBERTa) from the Hugging Face Transformers library for the named entity recognition task. It is easier to use a Generative LLM for zero-shot NER and return the response as a usable JSON.

4. **Parse Bounding Box Coordinates:**
   - Integrate the OCR library (e.g., Tesseract OCR) to obtain the bounding box coordinates of the recognized words in the PDF document.
   - Match the extracted names and last names with their corresponding bounding box coordinates from the OCR results.

5. **Implement the RESTful API:**
   - Use FastAPI or Flask to create a RESTful API endpoint that accepts a scanned PDF file and a set of name-last name pairs as input.
   - Perform the text extraction, named entity recognition, and bounding box parsing steps on the uploaded PDF file.
   - Implement fuzzy matching (e.g., using the python-fuzzy library) between the extracted names and the provided names in the API request, with a similarity threshold of 90%.
   - Return the extracted names, their bounding box coordinates, and the fuzzy matching results as a JSON response.

6. **Set up a Vector Database:**
   - Choose and install an open-source vector database solution (e.g., Chroma, Qdrant, Milvus) compatible with your development environment. If you are using a docker image of an open-source database, make sure to include it in the docker compose file.
   - Create a new database or collection within the vector database to store the extracted text from the PDF documents.

7. **Implement Text Embedding:**
   - Use an open-source embedding model like Sentence-BERT or Universal Sentence Encoder to convert the extracted text into vector representations.
   - Integrate the embedding model with the vector database to store the text embeddings along with their corresponding text snippets.

8. **Implement Retrieval-Augmented Generation (RAG):**
   - Create a new RESTful API endpoint that accepts a question as input.
   - Implement the RAG strategy to retrieve relevant text snippets from the vector database based on the similarity between the question embedding and the stored text embeddings.
   - Use a pre-trained language model (e.g., GPT-3.5, Gemini) to generate an answer based on the retrieved text snippets and the input question.
   - Return the generated answer as the response to the API request.

9. **Containerization:**
   - Create a new Dockerfile in the project directory with all the required dependencies to run the API server.
   - If required, generate a docker-compose file.

10. **Documentation and Submission:**
    - Document the code, and any external libraries or resources used.
    - Provide instructions for setting up and running the system, as well as any additional dependencies or requirements.

---

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

Expected: **49 passed**.

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
