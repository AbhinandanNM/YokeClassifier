# -*- coding: utf-8 -*-
"""
utils/image_io.py
==================
Shared image-loading utilities used across all pipeline phases.

Provides:
  iter_images(directory)  - yield all jpg/jpeg/png files in a directory
  load_bgr(path)          - load an image as BGR (raises on failure)
"""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# Supported file extensions (lowercase only; we also check upper-case variants)
_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_images(directory: Path) -> Iterator[Path]:
    """Yield all image files (jpg, jpeg, png) in *directory*, sorted by name.

    Case-insensitive: matches both '.JPG' and '.jpg'.
    Does NOT recurse into sub-directories.
    """
    directory = Path(directory)
    seen: set = set()
    all_files: list = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in _EXTENSIONS:
            # Deduplicate in case glob returns duplicates on case-insensitive FS
            key = p.name.lower()
            if key not in seen:
                seen.add(key)
                all_files.append(p)
    yield from sorted(all_files)


def load_bgr(path: Path) -> np.ndarray:
    """Load an image as a BGR numpy array.

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError:      if cv2 cannot decode the image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Image not found: {}".format(path))
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError("cv2.imread returned None for: {}".format(path))
    return bgr
