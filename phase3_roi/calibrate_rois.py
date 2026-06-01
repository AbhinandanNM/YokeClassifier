# -*- coding: utf-8 -*-
"""
phase3_roi/calibrate_rois.py
=============================
Phase 3 - Interactive ROI calibration tool.

Opens a reference aligned image in an OpenCV window.
For each feature, you draw a rectangle by clicking two corners.
The result is written back to config/config.yaml under the [roi] section.

Controls
--------
  Left-click + drag : draw ROI rectangle for the current feature
  SPACE / ENTER     : confirm current ROI and move to next feature
  R                 : redo / redraw current ROI
  Q / ESC           : quit without saving

Usage
-----
    python phase3_roi/calibrate_rois.py

    Optionally specify a different reference image:
    python phase3_roi/calibrate_rois.py --image path/to/aligned_image.jpg
"""

import io
import sys
import argparse
from pathlib import Path
import copy

import cv2
import numpy as np
import yaml

ROOT        = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"

FEATURE_NAMES = ["spline", "top_nib", "bottom_nib",
                  "hole_tl", "hole_tr", "hole_bl", "hole_br"]

FEATURE_DESCRIPTIONS = {
    "spline"    : "CENTER SPLINE   (tall vertical strip in the middle)",
    "top_nib"   : "TOP NIB         (small protrusion at top centre)",
    "bottom_nib": "BOTTOM NIB      (small protrusion at bottom centre)",
    "hole_tl"   : "TOP-LEFT HOLE   (corner hole, top left)",
    "hole_tr"   : "TOP-RIGHT HOLE  (corner hole, top right)",
    "hole_bl"   : "BOTTOM-LEFT HOLE (corner hole, bottom left)",
    "hole_br"   : "BOTTOM-RIGHT HOLE (corner hole, bottom right)",
}

COLOR_ORDER = [
    (255, 100,   0),   # spline      - blue
    (  0, 200, 255),   # top_nib     - yellow
    (  0, 140, 255),   # bottom_nib  - orange
    (  0, 200,   0),   # hole_tl     - green
    (  0, 200,   0),   # hole_tr     - green
    (  0, 200,   0),   # hole_bl     - green
    (  0, 200,   0),   # hole_br     - green
]


# ---------------------------------------------------------------------------
# Mouse callback state
# ---------------------------------------------------------------------------

class DrawState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.drawing   = False
        self.pt1       = None
        self.pt2       = None
        self.confirmed = False

    @property
    def has_rect(self):
        return self.pt1 is not None and self.pt2 is not None


state = DrawState()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.pt1     = (x, y)
        state.pt2     = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        state.pt2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        state.pt2     = (x, y)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_rect(pt1, pt2):
    """Return (x, y, w, h) with x,y at top-left, w/h positive."""
    x1, y1 = pt1
    x2, y2 = pt2
    x  = min(x1, x2)
    y  = min(y1, y2)
    w  = abs(x2 - x1)
    h  = abs(y2 - y1)
    return x, y, w, h


def draw_overlay(canvas, rois_done, current_idx, current_rect, color, scale_x, scale_y):
    """Draw all confirmed ROIs and the live rectangle being drawn."""
    out = canvas.copy()

    # Draw confirmed ROIs
    for i, (name, (ox, oy, ow, oh)) in enumerate(rois_done.items()):
        c = COLOR_ORDER[i]
        x = int(ox / scale_x)
        y = int(oy / scale_y)
        w = int(ow / scale_x)
        h = int(oh / scale_y)
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        cv2.putText(out, name, (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)

    # Draw live rectangle
    if current_rect is not None:
        x, y, w, h = current_rect
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    # Instruction banner at bottom
    feat = FEATURE_NAMES[current_idx]
    desc = FEATURE_DESCRIPTIONS[feat]
    banner_h = 60
    cv2.rectangle(out,
                  (0, out.shape[0] - banner_h),
                  (out.shape[1], out.shape[0]),
                  (30, 30, 30), -1)
    cv2.putText(out,
                "Draw ROI for: {}".format(desc),
                (10, out.shape[0] - banner_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out,
                "SPACE/ENTER=confirm  R=redo  Q/ESC=quit  [{}/{}]".format(
                    current_idx + 1, len(FEATURE_NAMES)),
                (10, out.shape[0] - banner_h + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------

def run_calibration(image_path: str) -> dict | None:
    """
    Interactive calibration.
    Returns a dict  { feature_name: [x, y, w, h] }  or None if user quit.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Cannot load image: {}".format(image_path))
        return None

    # Resize for display if image is very large
    disp_h = 800
    scale  = disp_h / img.shape[0]
    disp_w = int(img.shape[1] * scale)
    canvas = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    scale_x = img.shape[1] / disp_w
    scale_y = img.shape[0] / disp_h

    win_name = "ROI Calibration"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    cv2.setMouseCallback(win_name, mouse_callback)

    rois_done  = {}   # feature_name -> (x, y, w, h)  in ORIGINAL image coords
    feat_idx   = 0
    state.reset()

    print("\n[CALIBRATION] Starting interactive ROI definition.")
    print("  Image  : {}".format(image_path))
    print("  Display: {}x{}  (scale x={:.2f}, y={:.2f})".format(
        disp_w, disp_h, scale_x, scale_y))
    print("  Draw a rectangle around each feature when prompted.\n")

    while feat_idx < len(FEATURE_NAMES):
        feat  = FEATURE_NAMES[feat_idx]
        color = COLOR_ORDER[feat_idx]

        # Build current rect for display
        cur_rect = None
        if state.has_rect:
            cur_rect = normalise_rect(state.pt1, state.pt2)

        frame = draw_overlay(canvas, rois_done, feat_idx, cur_rect, color, scale_x, scale_y)
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):   # Q or ESC
            print("[CALIBRATION] Aborted by user.")
            cv2.destroyAllWindows()
            return None

        elif key in (ord(" "), 13):  # SPACE or ENTER — confirm
            if state.has_rect and cur_rect[2] > 5 and cur_rect[3] > 5:
                # Scale rect back to original image coordinates
                x, y, w, h = cur_rect
                ox = int(x * scale_x)
                oy = int(y * scale_y)
                ow = int(w * scale_x)
                oh = int(h * scale_y)
                rois_done[feat] = (ox, oy, ow, oh)
                print("  [{}] ROI set: x={} y={} w={} h={}".format(
                    feat, ox, oy, ow, oh))
                feat_idx += 1
                state.reset()
            else:
                print("  [!] Draw a rectangle first (it must have area > 0)")

        elif key == ord("r"):       # R — redo
            state.reset()
            print("  [{}] Redo: draw again".format(feat))

    cv2.destroyAllWindows()

    result = {name: list(coords) for name, coords in rois_done.items()}
    return result


# ---------------------------------------------------------------------------
# Config update
# ---------------------------------------------------------------------------

def update_config(roi_dict: dict) -> None:
    """Write the calibrated ROI coordinates back into config.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    for name, coords in roi_dict.items():
        cfg["roi"][name] = [int(v) for v in coords]

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print("\n[CALIBRATION] config.yaml updated with new ROI coordinates:")
    for name, coords in roi_dict.items():
        print("  {:<12}: {}".format(name, coords))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Interactive ROI calibration tool"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a reference aligned image (default: first image in aligned/)"
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if args.image:
        image_path = args.image
    else:
        aligned_dir = Path(cfg["paths"]["aligned_dir"])
        # Prefer a perfect image as reference (has all features visible)
        perfect_files = sorted(aligned_dir.glob("perfect_*_aligned.jpg"))
        if not perfect_files:
            perfect_files = sorted(aligned_dir.glob("*.jpg"))
        if not perfect_files:
            print("[ERROR] No aligned images found in: {}".format(aligned_dir))
            sys.exit(1)
        image_path = str(perfect_files[0])

    print("=" * 60)
    print("  PHASE 3 - ROI CALIBRATION")
    print("=" * 60)
    print("  Reference image: {}".format(image_path))

    roi_dict = run_calibration(image_path)
    if roi_dict is None:
        print("[CALIBRATION] No changes saved.")
        sys.exit(0)

    update_config(roi_dict)
    print("\n[CALIBRATION COMPLETE] Run visualize_rois.py to verify.")


if __name__ == "__main__":
    main()
