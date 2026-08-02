# AI Image Editor 

introduces semantic image retrieval, vector indexing, metadata schema upgrades, and API-level architecture improvements on top of the Week 1 and Week 2 upload/edit platform.

##Deliverables

- Dense embedding + retrieval pipeline for images and text queries
- ChromaDB persistent vector store integrated with upload/edit lifecycle
- Structured metadata schema with migration from earlier JSON format
- Version lineage with parent-child tracking and applied transform tags
- FastAPI search/index endpoints for backend-first integration
- Streamlit semantic search UI with relevance score display
- Test coverage for search and versioning behaviors


## Project Structure

- app.py: Streamlit UI and interaction flows
- utils.py: metadata manager, versioning, Gemini edit/caption operations
- search_engine.py: embedding generation, indexing, semantic retrieval
- api.py: FastAPI backend endpoints and centralized error responses
- schemas.py: typed Pydantic models for metadata and API contracts
- errors.py: custom app exceptions
- logging_config.py: rotating file + console logging setup
- tests/test_search_engine.py: semantic retrieval tests
- tests/test_metadata_versioning.py: version lineage and transform tests

## Embedding and Retrieval Pipeline

1. Upload image through Streamlit.
2. Save original metadata record using schema v3.0.
3. Generate caption through Gemini Vision API.
4. Trigger metadata update.
5. Emit sync callback to SearchEngine.
6. Build document text from filename + caption + edit prompts.
7. Generate dense embedding using Sentence Transformers.
8. Upsert vector in ChromaDB persistent collection.
9. On query, embed text and retrieve top-K nearest vectors with cosine similarity.
10. Return image IDs with normalized relevance scores.

Notes:
- If Sentence Transformers cannot load, deterministic fallback embeddings are used so the app remains functional.
- Search indexes both original image nodes and edited version nodes.

## Metadata Schema Upgrade

Metadata now uses this container format:

```json
{
  "schema_version": "3.0",
  "last_updated": "ISO_DATE",
  "images": {
    "image_id": {
      "id": "image_id",
      "original_id": "image_id",
      "current_version": 2,
      "versions": [
        {
          "id": "image_id::v1",
          "parent_id": "image_id",
          "version_number": 1,
          "applied_transforms": ["black_and_white"]
        }
      ]
    }
  }
}
```

Migration behavior:
- Legacy Week 1/2 metadata root format is auto-migrated at runtime.
- Existing records are normalized and backfilled with original_id, current_version, version IDs, and parent lineage.

## API Endpoints

Start API server:

```bash
uvicorn api:app --reload --port 8000
```

Available endpoints:

- GET /health
- POST /api/search
- POST /api/index/rebuild
- GET /api/images/{image_id}

Search example:

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"blue sky\", \"top_k\": 5}"
```

## Streamlit Frontend Search

Start UI:

```bash
streamlit run app.py
```

UI behavior:
- Semantic search bar maps natural language queries to dense vectors.
- Results are ranked by relevance score.
- If semantic retrieval fails, keyword search fallback is used.
- Sidebar includes Rebuild Search Index action for existing datasets.

## Setup Instructions

1. Create and activate virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment file.

```bash
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash
```

4. Run Streamlit app.

```bash
streamlit run app.py
```

5. Optional: run API service in another terminal.

```bash
uvicorn api:app --reload --port 8000
```

## Index Existing Images

To rebuild vectors for all existing metadata records:

- In Streamlit sidebar, click Rebuild Search Index
- Or call API endpoint POST /api/index/rebuild

## Testing

Run tests:

```bash
pytest -q
```

Current automated coverage includes:
- Full-index rebuild and semantic query response shape
- Image version parent-child lineage
- Applied transform extraction and version metadata updates

## Logging and Error Handling

- Rotating logs are written to data/app.log
- Structured application exceptions:
  - validation_error
  - not_found
  - embedding_error
  - storage_error
- FastAPI central handlers convert exceptions to stable JSON responses
- Streamlit upload/search flows handle expected failures gracefully and continue processing

## Notes for Production Hardening

- Replace local JSON with SQLite/PostgreSQL repository layer
- Add authentication/authorization for API endpoints
- Add async task queue for caption generation and heavy embedding jobs
- Add object storage abstraction for cloud buckets
- Add CI pipeline with tests and lint checks
