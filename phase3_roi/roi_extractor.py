# -*- coding: utf-8 -*-
"""
phase3_roi/roi_extractor.py
============================
Phase 3 - ROI (Region of Interest) extraction from aligned images.

Responsibilities
----------------
- Reads ROI coordinates from config.yaml.
- Crops each named region (spline, top_nib, bottom_nib, hole_tl, hole_tr,
  hole_bl, hole_br) from an aligned 1080x1080 image.
- Resizes every crop to config.roi.crop_size x config.roi.crop_size.
- Converts to grayscale (as required by the training spec).
- Returns a dict mapping feature name -> numpy array.

Usage (standalone)
------------------
    from phase3_roi.roi_extractor import ROIExtractor
    extractor = ROIExtractor(config)
    crops = extractor.extract(aligned_bgr_image)
    # crops['spline'] -> 128x128 uint8 grayscale array
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml


ROOT        = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Canonical feature order — always consistent
FEATURE_NAMES = ["spline", "top_nib", "bottom_nib",
                  "hole_tl", "hole_tr", "hole_bl", "hole_br"]


class ROIExtractor:
    """Crops and resizes named ROIs from an aligned image."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = _load_cfg()
        self.cfg       = config
        roi_cfg        = config["roi"]
        self.crop_size = int(roi_cfg.get("crop_size", 128))

        # Build the ROI table: name -> (x, y, w, h)
        self.rois: Dict[str, Tuple[int, int, int, int]] = {}
        for name in FEATURE_NAMES:
            coords = roi_cfg.get(name)
            if coords is None:
                raise KeyError(
                    "ROI '{}' not found in config.yaml. "
                    "Run the calibration tool first: "
                    "python phase3_roi/calibrate_rois.py".format(name)
                )
            x, y, w, h = [int(v) for v in coords]
            self.rois[name] = (x, y, w, h)

        # Expected canvas size
        out_size = config["alignment"].get("output_size", [1080, 1080])
        self.canvas_w = int(out_size[0])
        self.canvas_h = int(out_size[1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, aligned_bgr: np.ndarray,
                grayscale: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract all 7 ROI crops from an aligned image.

        Parameters
        ----------
        aligned_bgr : H x W x 3  uint8 BGR image (must be canvas_w x canvas_h)
        grayscale   : if True, return single-channel crops (recommended for training)

        Returns
        -------
        dict  feature_name -> crop  (crop_size x crop_size, uint8)
        """
        if aligned_bgr is None:
            raise ValueError("aligned_bgr is None")

        h, w = aligned_bgr.shape[:2]
        if (w, h) != (self.canvas_w, self.canvas_h):
            # Silently resize — might happen during live capture
            aligned_bgr = cv2.resize(aligned_bgr,
                                     (self.canvas_w, self.canvas_h),
                                     interpolation=cv2.INTER_AREA)

        crops: Dict[str, np.ndarray] = {}
        for name, (x, y, rw, rh) in self.rois.items():
            # Clamp to image boundary
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(self.canvas_w, x + rw)
            y2 = min(self.canvas_h, y + rh)

            crop = aligned_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                # ROI is out of bounds — return a black patch
                crop = np.zeros((rh, rw, 3), dtype=np.uint8)

            # CLAHE normalisation on grayscale before resize
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray  = clahe.apply(gray)

            # Resize to uniform crop_size
            out = cv2.resize(gray if grayscale else crop,
                             (self.crop_size, self.crop_size),
                             interpolation=cv2.INTER_AREA)
            crops[name] = out

        return crops

    def extract_with_coords(self, aligned_bgr: np.ndarray) \
            -> Dict[str, Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Like extract(), but also returns the (x, y, w, h) of each ROI.
        Used by the visualisation overlay.
        """
        crops = self.extract(aligned_bgr, grayscale=False)
        return {name: (crops[name], self.rois[name]) for name in FEATURE_NAMES}

    def get_roi_coords(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Return the raw (x, y, w, h) dict — useful for drawing overlays."""
        return dict(self.rois)
