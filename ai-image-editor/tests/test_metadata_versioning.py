from io import BytesIO
from pathlib import Path

from PIL import Image

from utils import ImageManager


class FakeUploadFile:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self._payload = payload
        self.size = len(payload)

    def getbuffer(self):
        return self._payload


def _make_image_payload() -> bytes:
    buffer = BytesIO()
    img = Image.new("RGB", (32, 24), color=(120, 50, 200))
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_version_lineage_and_transforms(tmp_path: Path):
    manager = ImageManager(data_dir=str(tmp_path), metadata_file="metadata.json")

    upload = FakeUploadFile("sample.jpg", _make_image_payload())
    image = manager.save_image(upload, caption="sample caption")

    edited = Image.new("RGB", (32, 24), color=(10, 10, 10))
    v1 = manager.save_edited_version(
        image["id"],
        edited,
        "make this black and white",
        "convert to grayscale",
    )
    v2 = manager.save_edited_version(
        image["id"],
        edited,
        "make background blur",
        "apply gaussian blur",
    )

    assert v1["original_id"] == image["id"]
    assert v1["parent_id"] == image["id"]
    assert "black_and_white" in v1["applied_transforms"]

    assert v2["parent_id"] == f"{image['id']}::v1"
    assert "blur" in v2["applied_transforms"]

    latest = manager.get_image_by_id(image["id"])
    assert latest["current_version"] == 2
    assert len(latest["versions"]) == 2
