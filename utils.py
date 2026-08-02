"""Utility functions for image management and AI operations."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Callable, Dict, List, Optional

import google.generativeai as genai
from PIL import Image

from errors import NotFoundError, StorageError, ValidationError
from logging_config import setup_logging
from schemas import ImageMetadataModel, ImageVersionModel, MetadataContainerModel


class ImageManager:
    """Manages image storage, metadata, schema validation, and sync hooks."""

    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, data_dir: str = "data", metadata_file: str = "metadata.json"):
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.metadata_file = self.data_dir / metadata_file
        self._lock = RLock()
        self._sync_callback: Optional[Callable[[str, Dict], None]] = None
        self._logger = setup_logging()

        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        if not self.metadata_file.exists():
            self._save_metadata_container(MetadataContainerModel().model_dump())
        else:
            # Load once to migrate old formats automatically.
            self._load_metadata_container()

    def set_sync_callback(self, callback: Callable[[str, Dict], None]) -> None:
        self._sync_callback = callback

    def _emit_sync_event(self, event: str, payload: Dict) -> None:
        if not self._sync_callback:
            return
        try:
            self._sync_callback(event, payload)
        except Exception as exc:
            self._logger.exception("Sync callback failed for event=%s: %s", event, exc)

    def _load_metadata_container(self) -> Dict:
        """Load metadata with migration support from Week 1/2 format."""
        with self._lock:
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except FileNotFoundError:
                raw = {}
            except json.JSONDecodeError as exc:
                raise StorageError("Invalid metadata JSON", details={"reason": str(exc)}) from exc

            if isinstance(raw, dict) and "images" in raw and "schema_version" in raw:
                container = raw
            else:
                # Migrate legacy format where root was {image_id: image_metadata}
                container = {
                    "schema_version": "3.0",
                    "last_updated": datetime.now().isoformat(),
                    "images": raw if isinstance(raw, dict) else {},
                }

            normalized = self._normalize_container(container)
            if normalized != container:
                self._save_metadata_container(normalized)
            return normalized

    def _save_metadata_container(self, container: Dict) -> None:
        with self._lock:
            try:
                with open(self.metadata_file, "w", encoding="utf-8") as f:
                    json.dump(container, f, indent=2)
            except OSError as exc:
                raise StorageError("Failed to write metadata", details={"reason": str(exc)}) from exc

    def _normalize_container(self, container: Dict) -> Dict:
        images = container.get("images", {})
        normalized_images: Dict[str, Dict] = {}

        for image_id, image_data in images.items():
            normalized_images[image_id] = self._normalize_image_record(image_id, image_data)

        validated = MetadataContainerModel(
            schema_version="3.0",
            last_updated=datetime.now().isoformat(),
            images={k: ImageMetadataModel(**v) for k, v in normalized_images.items()},
        )
        return validated.model_dump()

    def _normalize_image_record(self, image_id: str, image_data: Dict) -> Dict:
        base = dict(image_data)
        base.setdefault("id", image_id)
        base.setdefault("original_id", image_id)
        base.setdefault("original_name", base.get("filename", image_id))
        base.setdefault("filename", image_id)
        base.setdefault("path", "")
        base.setdefault("upload_date", datetime.now().isoformat())
        base.setdefault("caption", "")
        base.setdefault("file_size", 0)
        base.setdefault("width", None)
        base.setdefault("height", None)

        versions = []
        for idx, version in enumerate(base.get("versions", []), start=1):
            versions.append(self._normalize_version_record(image_id, version, idx))

        base["versions"] = versions
        base["current_version"] = len(versions)

        return ImageMetadataModel(**base).model_dump()

    def _normalize_version_record(self, image_id: str, version: Dict, fallback_number: int) -> Dict:
        version_number = int(version.get("version_number", fallback_number))
        version_id = version.get("id") or f"{image_id}::v{version_number}"

        if version_number == 1:
            default_parent = image_id
        else:
            default_parent = f"{image_id}::v{version_number - 1}"

        normalized = {
            "id": version_id,
            "original_id": image_id,
            "parent_id": version.get("parent_id", default_parent),
            "version_number": version_number,
            "filename": version.get("filename", f"{image_id}_v{version_number}"),
            "path": version.get("path", ""),
            "created_date": version.get("created_date", datetime.now().isoformat()),
            "edit_prompt": version.get("edit_prompt", ""),
            "edit_description": version.get("edit_description", ""),
            "applied_transforms": version.get("applied_transforms", []),
            "file_size": int(version.get("file_size", 0)),
            "width": version.get("width"),
            "height": version.get("height"),
        }

        return ImageVersionModel(**normalized).model_dump()

    def _load_metadata(self) -> Dict:
        return self._load_metadata_container().get("images", {})

    def _save_images(self, images: Dict) -> None:
        container = {
            "schema_version": "3.0",
            "last_updated": datetime.now().isoformat(),
            "images": images,
        }
        self._save_metadata_container(self._normalize_container(container))

    def _validate_upload_file(self, uploaded_file) -> None:
        if uploaded_file is None:
            raise ValidationError("No upload file provided")

        file_name = getattr(uploaded_file, "name", "")
        if not file_name:
            raise ValidationError("Upload is missing file name")

        extension = Path(file_name).suffix.lower()
        if extension not in self.ALLOWED_EXTENSIONS:
            raise ValidationError(
                "Unsupported file format",
                details={"extension": extension, "allowed": sorted(self.ALLOWED_EXTENSIONS)},
            )

        size = getattr(uploaded_file, "size", 0)
        if not isinstance(size, int) or size <= 0:
            raise ValidationError("Uploaded file is empty")

    def _extract_applied_transforms(self, edit_prompt: str, edit_description: str) -> List[str]:
        source = f"{edit_prompt} {edit_description}".lower()
        transforms: List[str] = []
        keywords = {
            "black_and_white": ["black and white", "grayscale", "greyscale", "monochrome"],
            "blur": ["blur", "gaussian"],
            "sharpen": ["sharpen", "clarity"],
            "brightness": ["bright", "exposure"],
            "color": ["color", "hue", "saturation", "vibrant"],
            "background_change": ["background", "backdrop"],
            "remove_object": ["remove", "erase", "delete"],
            "add_object": ["add", "insert", "place"],
        }
        for label, words in keywords.items():
            if any(word in source for word in words):
                transforms.append(label)

        return transforms or ["general_edit"]

    def save_image(self, uploaded_file, caption: str = "") -> Dict:
        """Save uploaded image and metadata for the original image version."""
        self._validate_upload_file(uploaded_file)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = uploaded_file.name
        unique_filename = f"{timestamp}_{original_name}"

        image_path = self.uploads_dir / unique_filename
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        width: Optional[int]
        height: Optional[int]
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            width, height = None, None

        image_id = unique_filename
        metadata_entry = ImageMetadataModel(
            id=image_id,
            original_id=image_id,
            original_name=original_name,
            filename=unique_filename,
            path=str(image_path),
            upload_date=datetime.now().isoformat(),
            caption=caption,
            file_size=uploaded_file.size,
            width=width,
            height=height,
            current_version=0,
            versions=[],
        ).model_dump()

        all_metadata = self._load_metadata()
        all_metadata[image_id] = metadata_entry
        self._save_images(all_metadata)

        self._emit_sync_event("upsert_image", metadata_entry)
        return metadata_entry

    def get_all_images(self) -> List[Dict]:
        metadata = self._load_metadata()
        images = list(metadata.values())
        images.sort(key=lambda x: x.get("upload_date", ""), reverse=True)
        return images

    def get_image_by_id(self, image_id: str) -> Optional[Dict]:
        metadata = self._load_metadata()
        return metadata.get(image_id)

    def search_images(self, query: str) -> List[Dict]:
        all_images = self.get_all_images()
        query_lower = query.lower().strip()

        return [
            img
            for img in all_images
            if query_lower in img.get("caption", "").lower()
            or query_lower in img.get("original_name", "").lower()
        ]

    def update_caption(self, image_id: str, new_caption: str) -> bool:
        metadata = self._load_metadata()
        if image_id in metadata:
            metadata[image_id]["caption"] = new_caption.strip()
            self._save_images(metadata)
            self._emit_sync_event("upsert_image", metadata[image_id])
            return True
        return False

    def delete_image(self, image_id: str) -> bool:
        metadata = self._load_metadata()
        if image_id not in metadata:
            return False

        image_path = Path(metadata[image_id].get("path", ""))
        if image_path.exists():
            try:
                image_path.unlink()
            except Exception as exc:
                self._logger.warning("Error deleting file %s: %s", image_path, exc)

        for version in metadata[image_id].get("versions", []):
            version_path = Path(version.get("path", ""))
            if version_path.exists():
                try:
                    version_path.unlink()
                except Exception as exc:
                    self._logger.warning("Error deleting version file %s: %s", version_path, exc)

        del metadata[image_id]
        self._save_images(metadata)
        self._emit_sync_event("delete_image", {"id": image_id})
        return True

    def save_edited_version(
        self,
        image_id: str,
        edited_image: Image.Image,
        edit_prompt: str,
        edit_description: str = "",
    ) -> Dict:
        metadata = self._load_metadata()
        if image_id not in metadata:
            raise NotFoundError("Image not found", details={"image_id": image_id})

        image_record = metadata[image_id]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_number = len(image_record["versions"]) + 1
        original_name_stem = Path(image_record["original_name"]).stem
        file_extension = Path(image_record["original_name"]).suffix or ".jpg"
        version_filename = f"{original_name_stem}_v{version_number}_{timestamp}{file_extension}"

        version_path = self.uploads_dir / version_filename
        edited_image.save(version_path)

        file_size = version_path.stat().st_size
        width, height = edited_image.size

        if version_number == 1:
            parent_id = image_id
        else:
            parent_id = f"{image_id}::v{version_number - 1}"

        version_entry = ImageVersionModel(
            id=f"{image_id}::v{version_number}",
            original_id=image_id,
            parent_id=parent_id,
            version_number=version_number,
            filename=version_filename,
            path=str(version_path),
            created_date=datetime.now().isoformat(),
            edit_prompt=edit_prompt,
            edit_description=edit_description,
            applied_transforms=self._extract_applied_transforms(edit_prompt, edit_description),
            file_size=file_size,
            width=width,
            height=height,
        ).model_dump()

        image_record["versions"].append(version_entry)
        image_record["current_version"] = version_number
        metadata[image_id] = ImageMetadataModel(**image_record).model_dump()
        self._save_images(metadata)

        self._emit_sync_event("upsert_image", metadata[image_id])
        return version_entry

    def get_version_history(self, image_id: str) -> List[Dict]:
        metadata = self._load_metadata()
        if image_id not in metadata:
            return []
        return metadata[image_id].get("versions", [])


class GeminiVisionAPI:
    """Wrapper for Gemini Vision API operations."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)

        self.model_name = self._normalize_model_name(model_name)
        self.model = genai.GenerativeModel(self.model_name)

    def _normalize_model_name(self, model_name: str) -> str:
        raw = (model_name or "").strip().lower()

        if not raw:
            return "gemini-3.6-flash"

        if raw in {"gemini 3.6 flash", "gemini-3.6-flash"}:
            return "gemini-3.6-flash"

        # Keep explicit Gemini model selections (for account-level compatibility).
        normalized = raw.replace("_", "-").replace(" ", "-")
        if normalized.startswith("gemini"):
            return normalized

        return "gemini-3.6-flash"

    def set_model(self, model_name: str):
        self.model_name = self._normalize_model_name(model_name)
        self.model = genai.GenerativeModel(self.model_name)

    def generate_caption(self, image_path: str) -> str:
        prompt = "Write a short, clear caption for this image."
        with Image.open(image_path) as img:
            response = self.model.generate_content([prompt, img])

        text = getattr(response, "text", "") or ""
        return text.strip() if text.strip() else "No caption generated"

    def analyze_image(self, image_path: str, custom_prompt: str) -> str:
        with Image.open(image_path) as img:
            response = self.model.generate_content([custom_prompt, img])

        text = getattr(response, "text", "") or ""
        return text.strip() if text.strip() else "No response generated"

    def generate_edit_instructions(self, image_path: str, edit_prompt: str) -> str:
        """Generate concise instructions for how to perform a requested edit."""
        system_prompt = f"""You are an expert image editor. Analyze this image and the user's edit request.
Provide detailed, step-by-step instructions for how to accomplish this edit.

User's edit request: "{edit_prompt}"

Provide:
1. What specific changes need to be made
2. Which areas of the image will be affected
3. Technical approach to achieve this edit
4. Expected result

Keep it concise but detailed (2-4 sentences)."""

        with Image.open(image_path) as img:
            response = self.model.generate_content([system_prompt, img])

        text = getattr(response, "text", "") or ""
        return text.strip() if text.strip() else "Unable to generate edit instructions"

    def interpret_edit_prompt(self, edit_prompt: str) -> Dict[str, str]:
        """Interpret natural language edit prompt and categorize it."""
        prompt_lower = edit_prompt.lower()

        if any(word in prompt_lower for word in ["black and white", "grayscale", "greyscale", "monochrome"]):
            edit_type = "black_and_white"
        elif any(word in prompt_lower for word in ["remove", "delete", "erase"]):
            edit_type = "remove_object"
        elif any(word in prompt_lower for word in ["add", "insert", "place"]):
            edit_type = "add_object"
        elif any(word in prompt_lower for word in ["background", "backdrop"]) and "blur" not in prompt_lower:
            edit_type = "change_background"
        elif any(word in prompt_lower for word in ["blur"]):
            edit_type = "blur"
        elif any(word in prompt_lower for word in ["sharpen"]):
            edit_type = "sharpen"
        elif any(word in prompt_lower for word in ["bright", "dark", "exposure"]):
            edit_type = "brightness"
        elif any(word in prompt_lower for word in ["color", "hue", "saturation", "vibrant"]):
            edit_type = "adjust_color"
        elif any(word in prompt_lower for word in ["enhance"]):
            edit_type = "enhance"
        elif any(word in prompt_lower for word in ["resize", "crop", "scale"]):
            edit_type = "transform"
        else:
            edit_type = "general"

        return {
            "edit_type": edit_type,
            "original_prompt": edit_prompt,
            "complexity": "simple" if len(edit_prompt.split()) < 10 else "complex",
        }

    def simulate_edit(self, image_path: str, edit_info: Dict) -> Image.Image:
        """Apply simple PIL-based edits for demonstration purposes."""
        from PIL import ImageDraw, ImageEnhance, ImageFilter

        img = Image.open(image_path)
        edit_type = edit_info.get("edit_type", "general")
        prompt_lower = edit_info.get("original_prompt", "").lower()

        if edit_type == "black_and_white":
            img = img.convert("L").convert("RGB")

        elif edit_type == "brightness":
            enhancer = ImageEnhance.Brightness(img)
            if "bright" in prompt_lower or "increase" in prompt_lower:
                img = enhancer.enhance(1.3)
            elif "dark" in prompt_lower or "decrease" in prompt_lower:
                img = enhancer.enhance(0.7)
            else:
                img = enhancer.enhance(1.2)

        elif edit_type == "adjust_color":
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)

        elif edit_type == "enhance":
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)

        elif edit_type == "blur":
            if "background" in prompt_lower:
                img = img.filter(ImageFilter.GaussianBlur(radius=5))
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=3))

        elif edit_type == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)
            img = img.filter(ImageFilter.SHARPEN)

        elif edit_type == "transform":
            pass

        else:
            if edit_type in ["remove_object", "add_object", "change_background"]:
                img = img.convert("RGB")
                overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)
                text = f"Simulated Edit: {edit_type.replace('_', ' ').title()}"
                text_bbox = draw.textbbox((10, 10), text)
                draw.rectangle(
                    [(5, 5), (text_bbox[2] + 10, text_bbox[3] + 10)],
                    fill=(0, 0, 0, 128),
                )
                draw.text((10, 10), text, fill=(255, 255, 255, 255))
                img = img.convert("RGBA")
                img = Image.alpha_composite(img, overlay)
                img = img.convert("RGB")

        return img


PRESET_EDITS = {
    "Remove Background": "Remove the background from this image, keeping only the main subject",
    "Enhance Colors": "Enhance the colors and brightness of this image to make it more vibrant",
    "Add Blur Background": "Blur the background while keeping the main subject sharp",
    "Make Brighter": "Increase the brightness and exposure of this image",
    "Sharpen": "Sharpen this image to make details more clear",
    "Black and White": "Convert this image to black and white",
}


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_date(iso_date: str) -> str:
    """Format ISO date string to readable format."""
    try:
        dt = datetime.fromisoformat(iso_date)
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except Exception:
        return iso_date
