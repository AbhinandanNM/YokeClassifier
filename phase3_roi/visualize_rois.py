# -*- coding: utf-8 -*-
"""
phase3_roi/visualize_rois.py
=============================
Phase 3 - Visualise configured ROIs on sample aligned images.

Reads the current ROI coordinates from config.yaml, crops them using
the ROIExtractor, and generates an overlay image showing bounding boxes
and the extracted crops.

Usage
-----
    python phase3_roi/visualize_rois.py

Outputs
-------
    outputs/roi_visualization_perfect.jpg
    outputs/roi_visualization_defect.jpg
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase3_roi.roi_extractor import ROIExtractor, FEATURE_NAMES

def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)

def generate_visualization(img_path: Path, extractor: ROIExtractor, output_path: Path):
    """Draw ROIs and crops on a single image and save."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print("[ERROR] Cannot load image: {}".format(img_path))
        return

    # Use the extractor to get coordinates and crops
    extracted = extractor.extract_with_coords(bgr)

    canvas = bgr.copy()
    
    # Draw boxes
    for name, (crop, (x, y, w, h)) in extracted.items():
        # Green box for visual overlay
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Make a sidebar with the crops
    sidebar_w = extractor.crop_size + 40
    sidebar = np.zeros((canvas.shape[0], sidebar_w, 3), dtype=np.uint8)
    
    y_offset = 20
    for name in FEATURE_NAMES:
        crop_bgr, _ = extracted[name]
        
        cv2.putText(sidebar, name, (10, y_offset + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 15
        
        # Draw crop
        ch, cw = crop_bgr.shape[:2]
        if ch > 0 and cw > 0:
            sidebar[y_offset:y_offset+ch, 10:10+cw] = crop_bgr
            cv2.rectangle(sidebar, (10, y_offset), (10+cw, y_offset+ch), (255, 255, 255), 1)
        
        y_offset += ch + 5

    # Combine canvas and sidebar
    combined = np.hstack([canvas, sidebar])
    
    cv2.imwrite(str(output_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("  Saved -> {}".format(output_path))


def main():
    cfg = load_config()
    aligned_dir = Path(cfg["paths"]["aligned_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  PHASE 3 - ROI VISUALIZATION")
    print("=" * 60)
    
    extractor = ROIExtractor(cfg)
    
    perfect_files = sorted(aligned_dir.glob("perfect_*_aligned.jpg"))
    defect_files = sorted(aligned_dir.glob("defect_*_aligned.jpg"))
    
    if perfect_files:
        p_img = perfect_files[0]
        p_out = outputs_dir / "roi_visualization_perfect.jpg"
        print("[1/2] Visualizing perfect image: {}".format(p_img.name))
        generate_visualization(p_img, extractor, p_out)
        
    if defect_files:
        d_img = defect_files[0]
        d_out = outputs_dir / "roi_visualization_defect.jpg"
        print("[2/2] Visualizing defect image: {}".format(d_img.name))
        generate_visualization(d_img, extractor, d_out)
        
    print("\n[PHASE 3 VISUALIZATION COMPLETE]")

if __name__ == "__main__":
    main()
