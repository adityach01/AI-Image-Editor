from pathlib import Path

from search_engine import SearchEngine, TextEmbeddingModel


class DummyEmbeddingModel(TextEmbeddingModel):
    def __init__(self):
        self._dimension = 8

    @property
    def dimension(self):
        return self._dimension

    def embed(self, texts):
        vectors = []
        for text in texts:
            base = [0.0] * self._dimension
            value = sum(ord(ch) for ch in text) % self._dimension
            base[value] = 1.0
            vectors.append(base)
        return vectors


def test_rebuild_and_search(tmp_path: Path):
    engine = SearchEngine(data_dir=str(tmp_path), embedding_model=DummyEmbeddingModel())

    images = [
        {
            "id": "img_a",
            "original_id": "img_a",
            "original_name": "sunset.jpg",
            "path": "data/uploads/sunset.jpg",
            "caption": "orange sunset sky over mountain",
            "versions": [],
        },
        {
            "id": "img_b",
            "original_id": "img_b",
            "original_name": "city.jpg",
            "path": "data/uploads/city.jpg",
            "caption": "city street with cars",
            "versions": [],
        },
    ]

    count = engine.rebuild_index(images)
    assert count == 2

    results = engine.search("sunset", top_k=2)
    assert len(results) >= 1
    assert all("score" in row for row in results)


def test_index_image_with_versions(tmp_path: Path):
    engine = SearchEngine(data_dir=str(tmp_path), embedding_model=DummyEmbeddingModel())

    image = {
        "id": "img_x",
        "original_id": "img_x",
        "original_name": "portrait.jpg",
        "path": "data/uploads/portrait.jpg",
        "caption": "student portrait",
        "versions": [
            {
                "id": "img_x::v1",
                "original_id": "img_x",
                "version_number": 1,
                "path": "data/uploads/portrait_v1.jpg",
                "edit_prompt": "make background blur",
                "applied_transforms": ["blur"],
            }
        ],
    }

    engine.index_image(image)
    results = engine.search("background blur", top_k=5)
    assert len(results) >= 1
    assert any(item["source_type"] in {"original", "version"} for item in results)
