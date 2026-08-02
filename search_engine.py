"""Semantic search engine backed by ChromaDB and dense text embeddings."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from errors import EmbeddingError, ValidationError
from logging_config import setup_logging


class TextEmbeddingModel:
    """Embedding adapter with sentence-transformers + deterministic fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dimension = 384
        self._logger = setup_logging()

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            sample = self._model.encode(["dimension probe"], normalize_embeddings=True)
            if sample and len(sample[0]) > 0:
                self._dimension = len(sample[0])
            self._logger.info("Loaded sentence-transformers model: %s", model_name)
        except Exception as exc:
            self._logger.warning(
                "Falling back to hash-based embeddings: %s", exc
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [vec.tolist() for vec in vectors]

        return [self._fallback_embed(text) for text in texts]

    def _fallback_embed(self, text: str) -> List[float]:
        """Deterministic dense vector fallback when model downloads are unavailable."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector: List[float] = []
        for idx in range(self._dimension):
            b = digest[idx % len(digest)]
            # Map byte [0, 255] -> [-1.0, 1.0]
            vector.append((b / 127.5) - 1.0)

        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]


class SearchEngine:
    """Handles embedding generation, indexing, and semantic retrieval."""

    def __init__(
        self,
        data_dir: str = "data",
        collection_name: str = "images",
        embedding_model: Optional[TextEmbeddingModel] = None,
    ) -> None:
        self._logger = setup_logging()
        self.data_dir = Path(data_dir)
        self.vector_store_dir = self.data_dir / "vector_store"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model or TextEmbeddingModel()

        self.client = chromadb.PersistentClient(path=str(self.vector_store_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _to_record_id(self, image_id: str, version_number: int) -> str:
        return image_id if version_number == 0 else f"{image_id}::v{version_number}"

    def _build_document(self, image_metadata: Dict[str, Any], version: Optional[Dict[str, Any]] = None) -> str:
        if version is None:
            prompt_chunks: List[str] = []
            for item in image_metadata.get("versions", []):
                edit_prompt = item.get("edit_prompt", "")
                if edit_prompt:
                    prompt_chunks.append(edit_prompt)
            prompts = " | ".join(prompt_chunks)
            return (
                f"name: {image_metadata.get('original_name', '')}; "
                f"caption: {image_metadata.get('caption', '')}; "
                f"edits: {prompts}"
            )

        return (
            f"name: {image_metadata.get('original_name', '')}; "
            f"caption: {image_metadata.get('caption', '')}; "
            f"version_prompt: {version.get('edit_prompt', '')}; "
            f"transforms: {' '.join(version.get('applied_transforms', []))}"
        )

    def _record_metadata(
        self,
        image_metadata: Dict[str, Any],
        version: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if version is None:
            return {
                "image_id": image_metadata["id"],
                "original_id": image_metadata["original_id"],
                "version_number": 0,
                "path": image_metadata.get("path", ""),
                "caption": image_metadata.get("caption", ""),
                "original_name": image_metadata.get("original_name", ""),
                "source_type": "original",
            }

        return {
            "image_id": version["id"],
            "original_id": version["original_id"],
            "version_number": int(version["version_number"]),
            "path": version.get("path", ""),
            "caption": image_metadata.get("caption", ""),
            "original_name": image_metadata.get("original_name", ""),
            "source_type": "version",
        }

    def index_image(self, image_metadata: Dict[str, Any]) -> None:
        """Upsert original image and all versions into vector store."""
        image_id = image_metadata.get("id", "")
        if not image_id:
            raise ValidationError("Missing image id for indexing")

        # Remove old entries for this original image before re-upserting.
        self.remove_image(image_id)

        records: List[Dict[str, Any]] = [
            {
                "id": self._to_record_id(image_id, 0),
                "document": self._build_document(image_metadata),
                "metadata": self._record_metadata(image_metadata, None),
            }
        ]

        for version in image_metadata.get("versions", []):
            version_number = int(version.get("version_number", 0))
            records.append(
                {
                    "id": self._to_record_id(image_id, version_number),
                    "document": self._build_document(image_metadata, version),
                    "metadata": self._record_metadata(image_metadata, version),
                }
            )

        documents = [r["document"] for r in records]
        ids = [r["id"] for r in records]
        metadatas = [r["metadata"] for r in records]

        try:
            embeddings = self.embedding_model.embed(documents)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            self._logger.info("Indexed image %s (%d vectors)", image_id, len(ids))
        except Exception as exc:
            raise EmbeddingError(
                "Failed to index image",
                details={"image_id": image_id, "reason": str(exc)},
            ) from exc

    def remove_image(self, image_id: str) -> None:
        """Delete original and version vectors for an image."""
        if not image_id:
            return

        try:
            found = self.collection.get(where={"original_id": image_id}, include=["metadatas"])
            delete_ids = found.get("ids", []) if found else []
            if delete_ids:
                self.collection.delete(ids=delete_ids)
                self._logger.info("Removed %d vectors for %s", len(delete_ids), image_id)
        except Exception as exc:
            raise EmbeddingError(
                "Failed to remove image vectors",
                details={"image_id": image_id, "reason": str(exc)},
            ) from exc

    def rebuild_index(self, all_images: List[Dict[str, Any]]) -> int:
        """Rebuild the full semantic index from metadata entries."""
        existing = self.collection.get(include=[])
        existing_ids = existing.get("ids", []) if existing else []
        if existing_ids:
            self.collection.delete(ids=existing_ids)
        indexed = 0
        for image_data in all_images:
            self.index_image(image_data)
            indexed += 1
        self._logger.info("Rebuilt semantic index for %d images", indexed)
        return indexed

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValidationError("Search query is required")

        query_text = query.strip()
        try:
            query_embedding = self.embedding_model.embed([query_text])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            raise EmbeddingError(
                "Failed to run semantic search",
                details={"query": query_text, "reason": str(exc)},
            ) from exc

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: List[Dict[str, Any]] = []
        for metadata, distance in zip(metadatas, distances):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            output.append(
                {
                    "image_id": metadata.get("image_id", ""),
                    "original_id": metadata.get("original_id", ""),
                    "version_number": int(metadata.get("version_number", 0)),
                    "path": metadata.get("path", ""),
                    "caption": metadata.get("caption", ""),
                    "original_name": metadata.get("original_name", ""),
                    "source_type": metadata.get("source_type", "original"),
                    "score": round(similarity, 4),
                }
            )

        return output
