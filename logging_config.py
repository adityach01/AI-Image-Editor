"""Central logging configuration for Streamlit app and API."""
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_path: str = "data/app.log") -> logging.Logger:
    """Configure root logger once and return app logger."""
    logger = logging.getLogger("ai_image_editor")
    if logger.handlers:
        return logger

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger
