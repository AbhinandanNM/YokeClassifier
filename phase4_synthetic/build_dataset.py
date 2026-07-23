# -*- coding: utf-8 -*-
"""
phase4_synthetic/build_dataset.py
==================================
Builds the final machine-learning dataset of ROI crops.

Process:
1. Loops over all images in `aligned/` directory.
2. For PERFECT images:
   - Extracts all 7 ROIs and saves them into `<feature>/present/`.
   - Synthesizes 3 "absent" variants for each feature (using cv2.inpaint)
     and extracts the ROIs, saving them to `<feature>/absent/`.
3. For DEFECT images (which have ALL features missing):
   - Extracts all 7 ROIs and saves them into `<feature>/absent/`.

The output is a balanced dataset of `128x128` grayscale crops ready for PyTorch.

Usage
-----
    python phase4_synthetic/build_dataset.py
"""

import sys
from pathlib import Path
import random

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase3_roi.roi_extractor import ROIExtractor, FEATURE_NAMES
from phase4_synthetic.defect_generator import DefectSynthesizer
from utils.image_io import iter_images


# ---------------------------------------------------------------------------
# Label parsing — maps filename category to per-feature presence dict
# ---------------------------------------------------------------------------

# Features present in each named category (True = PRESENT, False = ABSENT)
# If a key is absent from the dict, fall back to the category-wide default.
_CATEGORY_LABELS = {
    # All 7 features PRESENT
    "perfect": {f: True  for f in FEATURE_NAMES},
    # All 7 features ABSENT
    "allmissing": {f: False for f in FEATURE_NAMES},
    # Drill holes are done (PRESENT), spline + nibs are ABSENT
    "onlydrilldone": {
        "spline":      False,
        "top_nib":     False,
        "bottom_nib":  False,
        "hole_tl":     True,
        "hole_tr":     True,
        "hole_bl":     True,
        "hole_br":     True,
    },
    # Nibs are ABSENT, everything else PRESENT
    "onlynibblingmissing": {
        "spline":      True,
        "top_nib":     False,
        "bottom_nib":  False,
        "hole_tl":     True,
        "hole_tr":     True,
        "hole_bl":     True,
        "hole_br":     True,
    },
    # Alias for older naming conventions
    "onlynibblingdone": {
        "spline":      True,
        "top_nib":     False,
        "bottom_nib":  False,
        "hole_tl":     True,
        "hole_tr":     True,
        "hole_bl":     True,
        "hole_br":     True,
    },
}


def _parse_labels(filename: str) -> dict:
    """Return per-feature presence dict for a given aligned image filename.

    Looks for a category keyword inside the filename (case-insensitive).
    Falls back to 'allmissing' (all absent) for unknown defect names, or
    'perfect' (all present) if the word 'perfect' appears.

    Example
    -------
    >>> _parse_labels("Coated-onlydrilldone_aligned.jpg")
    {'spline': False, 'top_nib': False, 'bottom_nib': False,
     'hole_tl': True, 'hole_tr': True, 'hole_bl': True, 'hole_br': True}
    """
    lower = filename.lower()
    # Check each category keyword, longest first to avoid substring confusion
    for key in sorted(_CATEGORY_LABELS, key=len, reverse=True):
        if key in lower:
            return dict(_CATEGORY_LABELS[key])
    # Fallback: treat as all-missing defect (conservative / safe)
    return {f: False for f in FEATURE_NAMES}

def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)

def build():
    cfg = load_config()
    paths = cfg["paths"]
    
    aligned_dir = Path(paths["aligned_dir"])
    roi_ds_dir = Path(paths["roi_dataset_dir"])
    
    # Ensure dataset dirs exist
    for feat in FEATURE_NAMES:
        # Group all hole corners under "hole" to share the model
        group = "hole" if "hole" in feat else feat
        (roi_ds_dir / group / "present").mkdir(parents=True, exist_ok=True)
        (roi_ds_dir / group / "absent").mkdir(parents=True, exist_ok=True)

    extractor = ROIExtractor(cfg)
    synth = DefectSynthesizer(cfg)
    augment_factor = int(cfg["synthetic"].get("augment_factor", 3))

    all_files = list(iter_images(aligned_dir))
    if not all_files:
        print("[ERROR] No aligned images found. Run Phase 2 first.")
        sys.exit(1)

    print("=" * 60)
    print("  PHASE 4 - ROI DATASET GENERATION")
    print("=" * 60)
    print("  Source : {}".format(aligned_dir))
    print("  Target : {}".format(roi_ds_dir))
    print("  Synth factor : {}x".format(augment_factor))
    print("-" * 60)

    n_present = 0
    n_absent = 0
    
    for i, fp in enumerate(all_files):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue

        # Per-feature presence map derived from the filename category tag
        label_map = _parse_labels(fp.name)
        all_present = all(label_map.values())    # True only for "perfect" parts

        # 1. Extract raw ROIs
        raw_crops = extractor.extract(bgr, grayscale=True)
        roi_coords = extractor.get_roi_coords()

        for feat in FEATURE_NAMES:
            crop = raw_crops[feat]
            group = "hole" if "hole" in feat else feat
            feature_present = label_map.get(feat, False)

            if feature_present:
                # Feature IS present in this image → save as PRESENT crop
                out_path = roi_ds_dir / group / "present" / "{}_{}.jpg".format(fp.stem, feat)
                cv2.imwrite(str(out_path), crop)
                n_present += 1

                # Only generate synthetic ABSENT from a fully-perfect image
                # (partial images are already providing real absent examples)
                if all_present:
                    for aug_idx in range(augment_factor):
                        syn_bgr = synth.remove_feature(bgr, feat, roi_coords[feat])
                        syn_crops = extractor.extract(syn_bgr, grayscale=True)
                        syn_crop = syn_crops[feat]
                        syn_path = roi_ds_dir / group / "absent" / "{}_{}_syn_{}.jpg".format(
                            fp.stem, feat, aug_idx)
                        cv2.imwrite(str(syn_path), syn_crop)
                        n_absent += 1
            else:
                # Feature IS absent in this image → save as ABSENT crop
                out_path = roi_ds_dir / group / "absent" / "{}_{}.jpg".format(fp.stem, feat)
                cv2.imwrite(str(out_path), crop)
                n_absent += 1
                
        if (i + 1) % 20 == 0:
            print("  Processed {}/{} images...".format(i + 1, len(all_files)))

    print("\n[PHASE 4 COMPLETE]")
    print("  Total PRESENT crops generated : {}".format(n_present))
    print("  Total ABSENT  crops generated : {}".format(n_absent))
    
    # Print breakdown per group
    for group in ["spline", "top_nib", "bottom_nib", "hole"]:
        p_count = len(list((roi_ds_dir / group / "present").glob("*.jpg")))
        a_count = len(list((roi_ds_dir / group / "absent").glob("*.jpg")))
        print("  - {:<12}: {:>4} present, {:>4} absent".format(group, p_count, a_count))


if __name__ == "__main__":
    build()
