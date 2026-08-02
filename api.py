"""FastAPI service exposing search/index endpoints for Week 3 architecture."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from errors import AppError, NotFoundError
from logging_config import setup_logging
from schemas import ErrorResponseModel, SearchRequestModel, SearchResponseModel
from search_engine import SearchEngine
from utils import ImageManager

logger = setup_logging()
app = FastAPI(title="AI Image Editor API", version="3.0.0")

image_manager = ImageManager()
search_engine = SearchEngine()


def _sync_callback(event: str, payload: dict) -> None:
    if event == "upsert_image":
        search_engine.index_image(payload)
    elif event == "delete_image":
        search_engine.remove_image(payload.get("id", ""))


image_manager.set_sync_callback(_sync_callback)


@app.on_event("startup")
def warm_index() -> None:
    indexed_count = search_engine.rebuild_index(image_manager.get_all_images())
    logger.info("Startup indexing complete (%d images)", indexed_count)


@app.exception_handler(AppError)
async def app_error_handler(_, exc: AppError):
    payload = ErrorResponseModel(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def generic_error_handler(_, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    payload = ErrorResponseModel(
        error_code="internal_error",
        message="Unexpected server error",
        details={},
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "ai-image-editor-api", "version": "3.0.0"}


@app.post("/api/search", response_model=SearchResponseModel)
def semantic_search(request: SearchRequestModel) -> SearchResponseModel:
    results = search_engine.search(query=request.query, top_k=request.top_k)
    return SearchResponseModel(
        query=request.query,
        total_results=len(results),
        results=results,
    )


@app.post("/api/index/rebuild")
def rebuild_index() -> dict:
    indexed_count = search_engine.rebuild_index(image_manager.get_all_images())
    return {"status": "ok", "indexed": indexed_count}


@app.get("/api/images/{image_id}")
def get_image(image_id: str) -> dict:
    image = image_manager.get_image_by_id(image_id)
    if not image:
        raise NotFoundError("Image not found", details={"image_id": image_id})
    return image
