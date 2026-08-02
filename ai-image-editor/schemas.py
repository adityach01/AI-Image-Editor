"""Typed schema models for image metadata and search payloads."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ImageVersionModel(BaseModel):
    id: str
    original_id: str
    parent_id: str
    version_number: int = Field(ge=1)
    filename: str
    path: str
    created_date: str
    edit_prompt: str
    edit_description: str = ""
    applied_transforms: List[str] = Field(default_factory=list)
    file_size: int = Field(ge=0)
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)


class ImageMetadataModel(BaseModel):
    id: str
    original_id: str
    original_name: str
    filename: str
    path: str
    upload_date: str
    caption: str = ""
    file_size: int = Field(ge=0)
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)
    current_version: int = Field(default=0, ge=0)
    versions: List[ImageVersionModel] = Field(default_factory=list)


class MetadataContainerModel(BaseModel):
    schema_version: str = "3.0"
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    images: Dict[str, ImageMetadataModel] = Field(default_factory=dict)


class SearchResultModel(BaseModel):
    image_id: str
    path: str
    caption: str
    original_name: str
    score: float
    source_type: str


class SearchRequestModel(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResponseModel(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultModel]


class ErrorResponseModel(BaseModel):
    error_code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
