# -*- coding: utf-8 -*-
"""
phase4_synthetic/defect_generator.py
=====================================
Generates synthetic missing-feature defects from perfect aligned images.

It takes an aligned image and the (x, y, w, h) of a feature ROI, dilates the mask
slightly to ensure the entire feature is covered, and uses cv2.inpaint to fill
it with surrounding background colour/texture.

Usage
-----
    from phase4_synthetic.defect_generator import DefectSynthesizer
    synth = DefectSynthesizer(config)
    defect_bgr = synth.remove_feature(perfect_bgr, "spline", roi_coords)
"""

import sys
import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml

class DefectSynthesizer:
    def __init__(self, config: dict):
        self.cfg = config
        syn = config.get("synthetic", {})
        self.radius = int(syn.get("inpaint_radius", 12))
        method_str  = syn.get("inpaint_method", "telea").lower()
        self.method = cv2.INPAINT_TELEA if method_str == "telea" else cv2.INPAINT_NS
        self.dilate = int(syn.get("mask_dilate_px", 8))

    def remove_feature(self, bgr: np.ndarray, feature_name: str,
                       coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Synthetically remove a feature by inpainting over its bounding box.
        
        Parameters
        ----------
        bgr    : Aligned BGR image.
        coords : (x, y, w, h) of the feature.
        
        Returns
        -------
        inpainted : BGR image with the feature removed.
        """
        x, y, w, h = coords
        
        # Clamp to image bounds
        ih, iw = bgr.shape[:2]
        x1 = max(0, x - self.dilate)
        y1 = max(0, y - self.dilate)
        x2 = min(iw, x + w + self.dilate)
        y2 = min(ih, y + h + self.dilate)

        # Create mask for the region
        mask = np.zeros((ih, iw), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        # For holes, we might want a slightly circular mask or just rect is fine
        # since inpaint blends with edges outside the mask.
        if "hole" in feature_name:
            # refine mask to a circle for holes to look more natural
            mask[y1:y2, x1:x2] = 0
            cx = x + w // 2
            cy = y + h // 2
            radius = (max(w, h) // 2) + self.dilate
            cv2.circle(mask, (cx, cy), radius, 255, -1)

        # Inpaint
        inpainted = cv2.inpaint(bgr, mask, self.radius, self.method)

        # Add a tiny bit of Gaussian noise to the inpainted region to match sensor noise
        noise = np.random.normal(0, 3, (y2-y1, x2-x1, 3)).astype(np.float32)
        patch = inpainted[y1:y2, x1:x2].astype(np.float32) + noise
        inpainted[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)

        return inpainted
