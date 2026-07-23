# -*- coding: utf-8 -*-
"""
phase2_alignment/test_alignment.py
====================================
Phase 2 - Alignment validation and batch alignment runner.

What this script does
---------------------
1. Tests alignment on a sample of perfect + defect images.
2. Reports per-image alignment quality (method, n_matches, reproj_error).
3. Saves QA mosaic: reference | raw | aligned side-by-side.
4. Batch-aligns ALL perfect + defect images and saves to aligned/ directory.
5. Flags any failed alignments for manual review.

Usage
-----
    # QA test on 10 samples only:
    python phase2_alignment/test_alignment.py --mode qa

    # Batch align all 400 images:
    python phase2_alignment/test_alignment.py --mode batch

    # Both (default):
    python phase2_alignment/test_alignment.py
"""

import io
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Make sure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from phase2_alignment.aligner import Aligner
from utils.image_io import iter_images


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# QA visualisation helpers
# ---------------------------------------------------------------------------

def make_side_by_side(ref_bgr: np.ndarray,
                      raw_bgr: np.ndarray,
                      aligned_bgr: np.ndarray,
                      label: str,
                      quality: dict,
                      thumb_h: int = 360) -> np.ndarray:
    """
    Build a 3-panel thumbnail:  Reference | Raw | Aligned
    with a status bar underneath.
    """
    def thumb(img, h):
        ratio = h / img.shape[0]
        w     = int(img.shape[1] * ratio)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    ref_t = thumb(ref_bgr,     thumb_h)
    raw_t = thumb(raw_bgr,     thumb_h)
    aln_t = thumb(aligned_bgr, thumb_h)

    # Make all the same width (aligned is always out_w x out_h)
    target_w = aln_t.shape[1]
    def pad_w(img, w):
        if img.shape[1] == w:
            return img
        canvas = np.zeros((img.shape[0], w, 3), dtype=np.uint8)
        canvas[:, :img.shape[1]] = img
        return canvas

    ref_t = pad_w(ref_t, target_w)
    raw_t = pad_w(raw_t, target_w)

    # Status bar
    method   = quality.get("method", "?")
    n_match  = quality.get("n_matches", 0)
    reproj   = quality.get("reproj_err", 0.0)
    success  = quality.get("success", False)
    color    = (0, 200, 0) if success else (0, 0, 220)

    status_h = 40
    bar      = np.zeros((status_h, target_w * 3, 3), dtype=np.uint8)
    cv2.putText(bar,
                "{} | method={} matches={} reproj={:.1f}px  [{}]".format(
                    label, method, n_match, reproj,
                    "OK" if success else "FAILED"),
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                cv2.LINE_AA)

    # Labels on top panels
    def label_panel(img, text):
        out = img.copy()
        cv2.putText(out, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    ref_t = label_panel(ref_t, "REFERENCE")
    raw_t = label_panel(raw_t, "RAW INPUT")
    aln_t = label_panel(aln_t, "ALIGNED")

    top   = np.hstack([ref_t, raw_t, aln_t])
    mosaic = np.vstack([top, bar])
    return mosaic


# ---------------------------------------------------------------------------
# QA mode
# ---------------------------------------------------------------------------

def run_qa(aligner: Aligner, cfg: dict, n_samples: int = 10) -> None:
    paths       = cfg["paths"]
    outputs_dir = Path(paths["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    qa_dir      = outputs_dir / "alignment_qa"
    qa_dir.mkdir(exist_ok=True)

    perfect_dir = Path(paths["perfect_dir"])
    defect_dir  = Path(paths["defect_dir"])

    sample_files = (
        list(iter_images(perfect_dir))[:n_samples // 2] +
        list(iter_images(defect_dir))[:n_samples // 2]
    )

    print("\n[QA] Running alignment on {} sample images...".format(len(sample_files)))
    print("{:<40} {:<12} {:<10} {:<12} {:<8}".format(
        "File", "Method", "Matches", "Reproj(px)", "Status"
    ))
    print("-" * 80)

    results = []
    for fp in sample_files:
        raw_bgr          = cv2.imread(str(fp))
        aligned, quality = aligner.align(raw_bgr)
        results.append(quality)

        status = "OK" if quality["success"] else "FAILED"
        print("{:<40} {:<12} {:<10} {:<12.2f} {:<8}".format(
            fp.name[:38],
            quality.get("method", "?"),
            quality.get("n_matches", 0),
            quality.get("reproj_err", 0.0),
            status
        ))

        # Save side-by-side mosaic
        mosaic = make_side_by_side(
            aligner.ref_bgr, raw_bgr, aligned,
            fp.name, quality
        )
        out_path = qa_dir / ("qa_" + fp.stem + ".jpg")
        cv2.imwrite(str(out_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 88])

    n_ok     = sum(1 for r in results if r["success"])
    n_fail   = len(results) - n_ok
    reproj_vals = [r.get("reproj_err", 0.0) for r in results if r["success"] and r.get("method") == "orb"]
    mean_reproj = float(np.mean(reproj_vals)) if reproj_vals else 0.0

    print("\n[QA SUMMARY]")
    print("  Success    : {}/{}".format(n_ok, len(results)))
    print("  Failed     : {}".format(n_fail))
    print("  Mean reproj error : {:.2f} px".format(mean_reproj))
    if mean_reproj < 5.0:
        print("  Alignment quality : EXCELLENT (< 5 px)")
    elif mean_reproj < 10.0:
        print("  Alignment quality : GOOD (< 10 px)")
    else:
        print("  Alignment quality : POOR - check reference image")
    print("  QA mosaics saved to: {}".format(qa_dir))


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(aligner: Aligner, cfg: dict) -> None:
    paths       = cfg["paths"]
    aligned_dir = Path(paths["aligned_dir"])
    aligned_dir.mkdir(parents=True, exist_ok=True)

    perfect_dir = Path(paths["perfect_dir"])
    defect_dir  = Path(paths["defect_dir"])

    all_files = (
        [(fp, "perfect") for fp in iter_images(perfect_dir)] +
        [(fp, "defect")  for fp in iter_images(defect_dir)]
    )

    print("\n[BATCH] Aligning {} images...".format(len(all_files)))
    t0      = time.time()
    n_ok    = 0
    n_fail  = 0
    failed  = []
    reproj_errors = []

    for i, (fp, label) in enumerate(all_files):
        raw_bgr          = cv2.imread(str(fp))
        aligned, quality = aligner.align(raw_bgr)

        # Save aligned image — preserve original filename, store in aligned/
        out_name = "{}_aligned.jpg".format(fp.stem)
        out_path = aligned_dir / out_name
        cv2.imwrite(str(out_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if quality["success"]:
            n_ok += 1
            if quality.get("method") == "orb":
                reproj_errors.append(quality.get("reproj_err", 0.0))
        else:
            n_fail += 1
            failed.append(str(fp))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(all_files) - i - 1) / rate
            print("  [{}/{}]  ok={} fail={}  rate={:.1f} img/s  ETA={:.0f}s".format(
                i + 1, len(all_files), n_ok, n_fail, rate, eta
            ))

    elapsed = time.time() - t0
    mean_reproj = float(np.mean(reproj_errors)) if reproj_errors else 0.0

    print("\n[BATCH COMPLETE]")
    print("  Total images   : {}".format(len(all_files)))
    print("  Aligned OK     : {}".format(n_ok))
    print("  Failed         : {}".format(n_fail))
    print("  Mean reproj err: {:.2f} px".format(mean_reproj))
    print("  Time elapsed   : {:.1f} s  ({:.2f} img/s)".format(
        elapsed, len(all_files) / elapsed))
    print("  Output dir     : {}".format(aligned_dir))

    if failed:
        fail_log = Path(paths["outputs_dir"]) / "alignment_failures.txt"
        with open(str(fail_log), "w") as f:
            f.write("\n".join(failed))
        print("  [!] {} images failed - see: {}".format(n_fail, fail_log))

    # Write alignment metadata CSV
    import csv
    meta_path = Path(paths["outputs_dir"]) / "alignment_metadata.csv"
    with open(str(meta_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "method", "n_matches",
                         "reproj_err", "success", "inliers"])
        for fp, label in all_files:
            raw_bgr          = cv2.imread(str(fp))
            aligned, quality = aligner.align(raw_bgr)
            writer.writerow([
                fp.name, label,
                quality.get("method", ""),
                quality.get("n_matches", 0),
                quality.get("reproj_err", 0.0),
                quality.get("success", False),
                quality.get("inliers", 0),
            ])
    print("  Metadata CSV   : {}".format(meta_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 - Alignment test and batch runner"
    )
    parser.add_argument(
        "--mode", choices=["qa", "batch", "both"], default="both",
        help="qa=sample test only | batch=align all | both=qa then batch"
    )
    parser.add_argument(
        "--qa-samples", type=int, default=10,
        help="Number of images to include in QA test (default: 10)"
    )
    args = parser.parse_args()

    cfg     = load_config()
    aligner = Aligner(cfg)

    print("=" * 60)
    print("  PHASE 2 - ALIGNMENT")
    print("=" * 60)

    if args.mode in ("qa", "both"):
        run_qa(aligner, cfg, n_samples=args.qa_samples)

    if args.mode in ("batch", "both"):
        run_batch(aligner, cfg)

    print("\n[PHASE 2 COMPLETE]")


if __name__ == "__main__":
    main()
