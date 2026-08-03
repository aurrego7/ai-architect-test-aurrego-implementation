# Technical Assessment — PDF Name Extractor & RAG API

## Overview

You are given a **partially implemented** application that extracts names from scanned PDF documents, identifies their bounding box locations, performs fuzzy matching, and answers questions using a RAG (Retrieval-Augmented Generation) pipeline.

The codebase has **bugs, missing features, and poor engineering practices**. Your task is to fix, complete, and improve it.

## What's Already Implemented

```
app/
├── main.py
├── models/
│   └── schemas.py
├── services/
│   ├── ocr_service.py
│   ├── ner_service.py
│   ├── bbox_service.py
│   ├── fuzzy_service.py
│   ├── embedding_service.py
│   ├── vector_service.py
│   └── rag_service.py
├── api/
│   ├── extract.py
│   └── rag.py
tests/
├── conftest.py
├── test_ocr.py
├── test_ner.py
├── test_bbox.py
├── test_fuzzy.py
├── test_vector.py
├── test_rag.py
├── test_api.py
├── test_architecture.py
└── test_integration.py
```

## Your Tasks

### 1. Fix All Bugs

Run the test suite with `pytest -v`. Multiple tests are **failing due to bugs** in the implementation. Find and fix each bug.

### 2. Refactor Architecture

The current codebase has significant design issues. Think about classes, abstractions, and SOLID principles.

### 3. Implement Software Engineering Best Practices

The codebase is missing critical production practices. Think about logging, configuration (look for hardcoded values), resilience, and validation.

### 4. Complete Missing Features

Some expected functionality is not implemented. The tests will guide you.

### 5. Containerization

- Create a `Dockerfile` for the application.
- Create a `docker-compose.yml` that includes all required services.
- Ensure the container can be built and run with `docker compose up`.

### 6. Design Document

Include a `DESIGN.md` (max 3 pages) covering:
- Component diagram showing all services and dependencies.
- Sequence diagram for both API endpoints.
- Technology choice justifications and tradeoffs.
- Scaling strategy for 1000+ PDFs/hour.
- Failure modes and mitigations.
- LLM routing strategy (if you were to add model selection based on query complexity).

## Evaluation Criteria

| Criteria | Weight |
|----------|--------|
| All tests passing (bugs fixed correctly) | 20% |
| Architecture and design patterns | 20% |
| Software engineering best practices | 20% |
| Code quality and decisions | 15% |
| Design document | 10% |
| Containerization | 10% |
| Bonus: additional tests you write | 5% |

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run tests (many will fail — that's expected)
pytest -v

# Start Qdrant (required for integration tests)
docker run -p 6333:6333 qdrant/qdrant

# Run the app
uvicorn app.main:app --reload
```

## Findings Report

As part of your submission, fill in the tables below documenting every bug you found and every improvement you made. This helps us understand your debugging process and engineering judgment.

### Bugs Found & Fixed

| # | File | Bug Description | How You Fixed It |
|---|------|-----------------|------------------|
| 1 | ocr_service.py | Incorrect iteration range in `extract_text_from_pdf` | Remove starting iteration index (ie. 1) to start iteration on starting index (ie. 0) |
| 2 | ocr_service.py | Missing document closure | Implement in `try/finally` and `with` to prevent leakage |
| 3 | ocr_service.py | Missing DPI transformation in `get_word_bounding_boxes` | Compute scale_x, scale_y to transform between 2 DPIs |
| 4 | test_ocr.py | Missing assertion for completeness | Add assertion on Y coord in `test_coordinates_in_pdf_space` |
| 5 | test_ocr.py | Mock built returned blank/new mock | Set return value to be the actual mock built |
| 6 | ner_service.py | Incorrect filter values on extracted entity labels | Set ent.lable_ to be only equal to `PERSON` and simplify for loop logic to list comprehension |
| 7 | bbox_service.py | No case-insensitive matching | Added .casefold() to both words and name_parts |
| 8 | bbox_service.py | Only initial match allowed | Inverted iteration to be first on words and then on names to be able to find multiple ocurrences of the same name. Also improved effiency by gating the window in which the secondary parts of the name are looked for |
| 9 | test_bbox.py | Missing assertion for completeness | Added assertions for secondary match and check page on secondary match |
| 10 | fuzzy_service.py | Incorrect variable value | Changed `SIMILARITY_THRESHOLD` value from 70 to 90 |
| 11 | fuzzy_service.py | Incorrect fuzz ratio function | Changed `partial_ratio` to `ratio` to prevent substring matches to map to 100% |
| 12 | vector_service.py | Non unique id on PointStruct causes overwrite on upsert | Set id value to uuid |
| 13 | test_rag.py | Incorrect assertions and validation for mid word chunking | Remove `stripped[0] == stripped[0]` from assertion since it is always true and rewrite mid word chunking check |
| 14 | rag_service.py | Incorrect chunking cuts words mid word | Implement check for the last whitespace in chunk to slice chunk |
| 15 | rag_service.py | Incorrect interpolation due to double set of brackets | Remove extra set of brackets to allow proper string interpolation |
| 16 | extract.py | No file type check on uploaded file | Added a cheap and expensive file type check for early exit |
| 17 | extract.py | Import library inside of functions | Remove library import mid function and add to top level imports |
| 19 | extract.py | Missing values in response dictionary | Add page_number key-value pair to bounding_box dict |
| 20 | schemas.py | Missing values in schema structures | Add mising fuzzy_matches field to `ExtractionResponse`, page_number field to `BoundingBox`, and sources to `RAGResponse` |
| 21 | ocr_service.py | Inconsistent DPI settings causing OCR extraction issues | Added DPI settings to `get_pixmap()` in `extract_text_from_pdf` to allow for proper text extraction in testing |

(add more rows as needed)

### Architecture & Design Improvements

| # | What You Changed | Why |
|---|------------------|-----|
| 1 | Reading/creating files without a context manager | In order to prevent data leakage whenever the file is not closed or if executions on the file fail the context manager will automatically handle the teardown |
| 2 | Module level methods with aliases | In order not to breake the test suite and allow for named imports (ie. `from app.service.fuzzy_service import fuzzy_match_names`). This serves as a compatibility suite for the implementation and the tests |
| 3 | Services as classes | For NER, embedding and vector services have an expesive reload of all the resources inside them. OCR, and also vector services have multiple operations that have the same config so this way we don't have to set them each time and pass the parameters around. BBOX and rag services have other dependencies inside, so in case of making changes or swaping those dependencies it is easier. Finally, for fuzzy service in reality this is only to have consistency across services since it is in reaily a single operation. |
| 4 | Remove model imports across services and centralize in providers.py | Reduce expensive reloads of model resources and centralizing in a single place also allows for quick changes in config in a single place. This also clearly separate between the service (ie. function) and the provider (ie. engine). This way if you want to chagne the functionality of how the service work it is indepedentent from the model, and vice versa. You can change how the model works without having to change how the result is implemented. Also add lru_cache to load a single time and then from memory every later call |
| 5 | Allow dependency functions in rag and bbox services to be ingested | Since the functions are dependecies for those services, by allowing ingestion we can quickly change to a different function/logic (ie. switching from NER to LLM or different bbox algorithm) without reworking drastically the service. The function needs to mantain the same output contract. |
| 6 | Create prompts.py to hold prompts | Have a single place of reference for all prompts, and by extracting the prompt from rag_service.py the separation of responsibility is clearer since rag_service is only concerned about executing the rag, not the quality of the prompt it uses |
| 7 | Extract llm call into class function | Create a small `_call_llm` function outside of `generate_answer` makes the intention of the `generate_answer` function more as an executor rather than configuration. This also allow to modify llm interaction independently in future situations |

(add more rows as needed)

### Engineering Best Practices Added

| # | Practice | Where / How You Implemented It |
|---|----------|-------------------------------|
| 1 | Linting | Utilize ruff to dictate linting and formating standards for readability and consistency |
| 2 | Idempotency in uploads | The pdf upload to the vector store should be indempotent to save from reuploads. By using `uuid5` on the chunk helps guard this |
| 3 | Config for parameters and keys | Built config.py with pydantic-settings config to start keys, paramaters and general values |
| 4 | Logging | Created the logging infrastructure and add log messages across code for visibility, tracking and error detection |
| 5 | Error and exception handling | Create a set of errors in errors.py and catch errors around the code implementation to translate to the core errors. This allows for proper visibility when erros ocurr |

(add more rows as needed)

## Submission

Create a branch and make a pull request to this repository.
