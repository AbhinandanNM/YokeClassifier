# -*- coding: utf-8 -*-
"""
phase1_analysis/analyze_dataset.py
===================================
Phase 1 - Complete dataset analysis for the industrial visual inspection system.

What this script does
---------------------
1. Loads all images from perfect/ and defect/ directories.
2. Computes per-image statistics: resolution, brightness, contrast, edge density,
   colour channel means, Otsu threshold, part occupancy.
3. Checks for corrupt/unreadable images.
4. Compares perfect vs defect distributions.
5. Detects and reports orientation variance across images.
6. Saves a visual summary grid (PNG) and a text report.

Usage
-----
    python phase1_analysis/analyze_dataset.py

Outputs
-------
    outputs/analysis_report.txt
    outputs/dataset_grid_perfect.png
    outputs/dataset_grid_defect.png
    outputs/analysis_histograms.png
    outputs/all_image_stats.csv
"""

import os
import sys
import csv
import time
import warnings
from pathlib import Path

# Force UTF-8 output on Windows so special chars never crash the console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend - no GUI needed
import matplotlib.pyplot as plt
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-image statistics
# ---------------------------------------------------------------------------

def load_image_bgr(path: str):
    """Load image; return None if corrupt or unreadable."""
    img = cv2.imread(path)
    return img


def compute_stats(img_bgr: np.ndarray) -> dict:
    """Compute a rich set of single-image statistics."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Otsu threshold + part occupancy
    otsu_val, otsu_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    part_px    = int(np.sum(otsu_mask > 0))
    part_ratio = part_px / (h * w)

    # Edge density
    edges        = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / (h * w)

    # Channel means
    b_ch, g_ch, r_ch = (img_bgr[:, :, c] for c in range(3))

    # Laplacian variance (sharpness proxy)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Histogram entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    return {
        "height"       : h,
        "width"        : w,
        "gray_mean"    : float(gray.mean()),
        "gray_std"     : float(gray.std()),
        "gray_min"     : int(gray.min()),
        "gray_max"     : int(gray.max()),
        "b_mean"       : float(b_ch.mean()),
        "g_mean"       : float(g_ch.mean()),
        "r_mean"       : float(r_ch.mean()),
        "otsu_thresh"  : float(otsu_val),
        "part_ratio"   : part_ratio,
        "edge_density" : edge_density,
        "lap_var"      : lap_var,
        "entropy"      : entropy,
    }


def analyse_directory(directory: str, label: str) -> list:
    """Analyse every image in directory. Returns list of stat-dicts."""
    dir_path = Path(directory)
    files    = sorted(dir_path.glob("*.jpg")) \
             + sorted(dir_path.glob("*.png")) \
             + sorted(dir_path.glob("*.bmp"))

    if not files:
        print("  [WARNING] No images found in {}".format(directory))
        return []

    records = []
    corrupt = []

    for i, fp in enumerate(files):
        img = load_image_bgr(str(fp))
        if img is None:
            corrupt.append(str(fp))
            continue
        stats = compute_stats(img)
        stats["filename"] = fp.name
        stats["label"]    = label
        records.append(stats)

        if (i + 1) % 50 == 0:
            print("  [{}] processed {}/{} images...".format(label, i + 1, len(files)))

    print("  [{}] done: {} valid, {} corrupt".format(label, len(records), len(corrupt)))
    return records


def aggregate(records: list, label: str) -> dict:
    """Return mean / std / min / max for every numeric key."""
    skip = {"filename", "label", "height", "width"}
    keys = [k for k in records[0].keys() if k not in skip]
    agg  = {"label": label, "count": len(records)}
    for k in keys:
        vals           = np.array([r[k] for r in records])
        agg[k + "_mean"] = float(vals.mean())
        agg[k + "_std"]  = float(vals.std())
        agg[k + "_min"]  = float(vals.min())
        agg[k + "_max"]  = float(vals.max())
    return agg


# ---------------------------------------------------------------------------
# Orientation analysis
# ---------------------------------------------------------------------------

def estimate_orientation(img_bgr: np.ndarray):
    """Return dominant line angle in degrees, or None."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=150,
        minLineLength=80, maxLineGap=20
    )
    if lines is None:
        return None
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 != x1:
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if not angles:
        return None
    return float(np.median(angles))


def analyse_orientation(directory: str, label: str, n_samples: int = 50) -> dict:
    dir_path = Path(directory)
    files    = sorted(dir_path.glob("*.jpg"))[:n_samples]
    angles   = []
    for fp in files:
        img = load_image_bgr(str(fp))
        if img is None:
            continue
        a = estimate_orientation(img)
        if a is not None:
            angles.append(a)
    if not angles:
        return {}
    return {
        "label"      : label,
        "n_samples"  : len(angles),
        "angle_mean" : float(np.mean(angles)),
        "angle_std"  : float(np.std(angles)),
        "angle_min"  : float(np.min(angles)),
        "angle_max"  : float(np.max(angles)),
    }


# ---------------------------------------------------------------------------
# Visual grids
# ---------------------------------------------------------------------------

def save_sample_grid(directory: str, label: str, output_path: str,
                     n_cols: int = 5, n_rows: int = 4) -> None:
    """Save a grid of thumbnail images as PNG."""
    dir_path = Path(directory)
    files    = sorted(dir_path.glob("*.jpg"))
    selected = files[: n_cols * n_rows]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))
    fig.suptitle("{} samples  ({} total)".format(label, len(files)),
                 fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(selected):
            img_bgr = cv2.imread(str(selected[i]))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            thumb   = cv2.resize(img_rgb, (320, 180))
            ax.imshow(thumb)
            ax.set_title(selected[i].name[:18], fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved grid -> {}".format(output_path))


def save_histogram_comparison(perfect_records: list, defect_records: list,
                               output_path: str) -> None:
    """Plot comparative histograms of key statistics."""
    keys   = ["gray_mean", "gray_std", "edge_density", "otsu_thresh",
              "part_ratio", "lap_var", "entropy"]
    titles = ["Gray Mean", "Gray Std Dev", "Edge Density", "Otsu Threshold",
              "Part Occupancy", "Laplacian Variance", "Histogram Entropy"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Perfect vs Defect - Feature Distribution Comparison",
                 fontsize=14, fontweight="bold")

    ax_iter = iter(axes.flat)
    for k, title in zip(keys, titles):
        ax     = next(ax_iter)
        p_vals = np.array([r[k] for r in perfect_records])
        d_vals = np.array([r[k] for r in defect_records])
        lo     = min(p_vals.min(), d_vals.min())
        hi     = max(p_vals.max(), d_vals.max())
        bins   = np.linspace(lo, hi, 30)
        ax.hist(p_vals, bins=bins, alpha=0.6, color="steelblue", label="Perfect")
        ax.hist(d_vals, bins=bins, alpha=0.6, color="tomato",    label="Defect")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for ax in ax_iter:   # turn off unused subplot
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved histograms -> {}".format(output_path))


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(perfect_agg: dict, defect_agg: dict,
                 orient_p: dict, orient_d: dict,
                 output_path: str,
                 perfect_records: list, defect_records: list) -> None:

    SEP = "=" * 72

    lines = [
        SEP,
        "  INDUSTRIAL VISUAL INSPECTION - DATASET ANALYSIS REPORT",
        SEP,
        "  Generated : {}".format(time.strftime("%Y-%m-%d %H:%M:%S")),
        "",
        "SUMMARY",
        "-" * 40,
        "  Perfect images  : {}".format(perfect_agg["count"]),
        "  Defect  images  : {}".format(defect_agg["count"]),
        "",
        "RESOLUTION",
        "-" * 40,
    ]

    p_res = set((r["height"], r["width"]) for r in perfect_records)
    d_res = set((r["height"], r["width"]) for r in defect_records)
    lines.append("  Perfect resolutions : {}".format(p_res))
    lines.append("  Defect  resolutions : {}".format(d_res))
    lines.append("")

    stat_keys = [
        ("gray_mean",    "Gray Mean"),
        ("gray_std",     "Gray Std Dev"),
        ("b_mean",       "Blue Channel Mean"),
        ("g_mean",       "Green Channel Mean"),
        ("r_mean",       "Red Channel Mean"),
        ("otsu_thresh",  "Otsu Threshold"),
        ("part_ratio",   "Part Occupancy (ratio)"),
        ("edge_density", "Edge Density"),
        ("lap_var",      "Laplacian Variance"),
        ("entropy",      "Histogram Entropy"),
    ]

    lines.append("STATISTICS  (mean +/- std)  [min - max]")
    lines.append("-" * 72)
    lines.append("{:<30} {:<32}  {:<32}".format("Metric", "Perfect", "Defect"))
    lines.append("-" * 72)

    for key, name in stat_keys:
        def fmt(agg, k):
            m  = agg[k + "_mean"]
            s  = agg[k + "_std"]
            lo = agg[k + "_min"]
            hi = agg[k + "_max"]
            return "{:7.3f} +/- {:5.3f}  [{:.3f} - {:.3f}]".format(m, s, lo, hi)

        lines.append("{:<30} {:<32}  {:<32}".format(
            name, fmt(perfect_agg, key), fmt(defect_agg, key)
        ))

    lines.append("")
    lines.append("KEY DIFFERENCES")
    lines.append("-" * 40)
    diff_edge = perfect_agg["edge_density_mean"] - defect_agg["edge_density_mean"]
    diff_grey = perfect_agg["gray_mean_mean"]     - defect_agg["gray_mean_mean"]
    edge_note = "Perfect has MORE edges (features present)" if diff_edge > 0 else "WARNING: unexpected"
    lines.append("  Edge density delta (P - D) : {:+.4f}  ({})".format(diff_edge, edge_note))
    lines.append("  Gray mean   delta (P - D)  : {:+.2f}".format(diff_grey))
    lines.append("")

    lines.append("ORIENTATION VARIANCE")
    lines.append("-" * 40)
    if orient_p:
        lines.append("  Perfect : mean={:.1f} deg  std={:.1f} deg  "
                     "range=[{:.1f}, {:.1f}]  (n={})".format(
                         orient_p["angle_mean"], orient_p["angle_std"],
                         orient_p["angle_min"],  orient_p["angle_max"],
                         orient_p["n_samples"]))
    if orient_d:
        lines.append("  Defect  : mean={:.1f} deg  std={:.1f} deg  "
                     "range=[{:.1f}, {:.1f}]  (n={})".format(
                         orient_d["angle_mean"], orient_d["angle_std"],
                         orient_d["angle_min"],  orient_d["angle_max"],
                         orient_d["n_samples"]))

    lines.append("")
    lines.append("ALIGNMENT REQUIREMENT")
    lines.append("-" * 40)
    if orient_p and orient_p["angle_std"] > 5:
        lines.append("  [!] High orientation variance detected (std > 5 deg)")
        lines.append("      ORB+homography alignment is REQUIRED before ROI extraction.")
    else:
        lines.append("  Orientation variance is acceptable. Alignment still recommended.")

    lines.append("")
    lines.append("AUGMENTATION RECOMMENDATIONS")
    lines.append("-" * 40)
    lines.append("  1. RandomBrightnessContrast - handles lighting variability")
    lines.append("  2. GaussNoise               - adds sensor noise robustness")
    lines.append("  3. MotionBlur               - handles camera shake")
    angle_std = orient_p.get("angle_std", 10) if orient_p else 10
    lines.append("  4. Rotate(limit={:.0f} deg)     - handles orientation tilt".format(angle_std + 5))
    lines.append("  5. HorizontalFlip           - valid symmetry for corner holes")
    lines.append("  6. CLAHE                    - normalises local contrast")
    lines.append("")
    lines.append(SEP)
    lines.append("  END OF REPORT")
    lines.append(SEP)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\n  Report saved -> {}".format(output_path))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(records: list, output_path: str) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print("  CSV exported -> {}".format(output_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg         = load_config()
    paths       = cfg["paths"]
    outputs_dir = Path(paths["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    perfect_dir = paths["perfect_dir"]
    defect_dir  = paths["defect_dir"]

    print("=" * 60)
    print("  PHASE 1 - DATASET ANALYSIS")
    print("=" * 60)

    print("\n[1/6] Analysing perfect images...")
    perfect_records = analyse_directory(perfect_dir, "perfect")

    print("\n[2/6] Analysing defect images...")
    defect_records  = analyse_directory(defect_dir, "defect")

    if not perfect_records or not defect_records:
        print("[ERROR] One or both directories returned no valid images. Aborting.")
        sys.exit(1)

    print("\n[3/6] Aggregating statistics...")
    perfect_agg = aggregate(perfect_records, "perfect")
    defect_agg  = aggregate(defect_records,  "defect")

    print("\n[4/6] Analysing orientation variance...")
    orient_p = analyse_orientation(perfect_dir, "perfect", n_samples=50)
    orient_d = analyse_orientation(defect_dir,  "defect",  n_samples=50)

    print("\n[5/6] Saving visual sample grids...")
    save_sample_grid(
        perfect_dir, "Perfect",
        str(outputs_dir / "dataset_grid_perfect.png")
    )
    save_sample_grid(
        defect_dir, "Defect (All Features Missing)",
        str(outputs_dir / "dataset_grid_defect.png")
    )
    save_histogram_comparison(
        perfect_records, defect_records,
        str(outputs_dir / "analysis_histograms.png")
    )

    print("\n[6/6] Writing report...")
    write_report(
        perfect_agg, defect_agg,
        orient_p, orient_d,
        str(outputs_dir / "analysis_report.txt"),
        perfect_records, defect_records
    )
    export_csv(
        perfect_records + defect_records,
        str(outputs_dir / "all_image_stats.csv")
    )

    print("\n[PHASE 1 COMPLETE]")
    print("  Outputs saved to: {}".format(outputs_dir))
    print("  Files created:")
    for f in sorted(outputs_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print("    {:<42} ({:.1f} KB)".format(f.name, size_kb))


if __name__ == "__main__":
    main()
