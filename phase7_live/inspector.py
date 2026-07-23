# -*- coding: utf-8 -*-
"""
phase7_live/inspector.py
=========================
Core inspection pipeline. Brings together alignment, ROI extraction, and ML inference.
"""

import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import torch
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase2_alignment.aligner import Aligner
from phase3_roi.roi_extractor import ROIExtractor, FEATURE_NAMES
from phase5_training.model import build_model
from phase5_training.augmentations import get_val_transforms

class Inspector:
    def __init__(self, config: dict):
        self.cfg = config
        self.aligner = Aligner(config)
        self.extractor = ROIExtractor(config)
        
        t_cfg = config["training"]
        models_dir = Path(config["paths"]["models_dir"])
        
        device_str = t_cfg.get("device", "auto")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
        
        self.img_size = t_cfg.get("image_size", 128)
        self.transform = get_val_transforms(self.img_size)
        self.threshold = config["inference"].get("present_threshold", 0.50)
        self.max_reproj = config["inference"].get("max_reproj_error", 8.0)
        
        # Load the 4 models
        self.models = {}
        for group in ["spline", "top_nib", "bottom_nib", "hole"]:
            m = build_model(pretrained=False, num_classes=2)
            path = models_dir / "{}.pth".format(group)
            if not path.exists():
                print("[WARNING] Model not found: {}".format(path))
            else:
                m.load_state_dict(torch.load(str(path), map_location=self.device))
                m = m.to(self.device)
                m.eval()
                self.models[group] = m
                
        print("[Inspector] Pipeline ready on {}".format(self.device))

    def inspect(self, raw_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Run the full inspection pipeline on a raw frame.
        
        Returns a dict:
        {
            "status": "PASS" | "FAIL" | "ALIGN_ERROR",
            "aligned_bgr": np.ndarray (or raw if alignment failed),
            "features": {
                "spline": {"present": True, "prob": 0.98, "box": (x,y,w,h)},
                ...
            },
            "align_quality": dict
        }
        """
        result = {
            "status": "FAIL",
            "aligned_bgr": raw_bgr,
            "features": {},
            "align_quality": {}
        }
        
        # 1. Align
        aligned, align_q = self.aligner.align(raw_bgr)
        result["align_quality"] = align_q
        
        # Check alignment threshold (only applies to ORB)
        if not align_q["success"] or (align_q.get("method") == "orb" and align_q.get("reproj_err", 0.0) > self.max_reproj):
            result["status"] = "ALIGN_ERROR"
            return result
            
        result["aligned_bgr"] = aligned
        
        # 2. Extract ROIs (we get the raw grayscale numpy arrays and the coordinates)
        extracted = self.extractor.extract_with_coords(aligned)
        
        # 3. ML Inference
        all_present = True
        
        for feat in FEATURE_NAMES:
            crop_bgr, box = extracted[feat]
            
            # Convert crop to grayscale and add channel dim
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            # Extractor already applies CLAHE when grayscale=True, but extract_with_coords returns BGR.
            # So apply CLAHE here
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
            gray = gray[:, :, np.newaxis]
            
            tensor = self.transform(torch.from_numpy(gray).permute(2, 0, 1)).unsqueeze(0).to(self.device)
            
            group = "hole" if "hole" in feat else feat
            model = self.models.get(group)
            
            if model is None:
                # Fallback if model missing
                prob = 0.0
                present = False
            else:
                with torch.no_grad():
                    out = model(tensor)
                    prob = torch.softmax(out, dim=1)[0, 1].item()
                present = prob >= self.threshold
                
            if not present:
                all_present = False
                
            result["features"][feat] = {
                "present": present,
                "prob": prob,
                "box": box
            }
            
        result["status"] = "PASS" if all_present else "FAIL"
        
        return result
