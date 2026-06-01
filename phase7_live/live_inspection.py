# -*- coding: utf-8 -*-
"""
phase7_live/live_inspection.py
===============================
Phase 7 - Live Inspection UI.

Runs the camera loop, passes frames to the Inspector, and visualizes the
result (boxes and PASS/FAIL text) on an OpenCV window.

Usage
-----
    python phase7_live/live_inspection.py
"""

import sys
from pathlib import Path

import cv2
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase7_live.camera import Camera
from phase7_live.inspector import Inspector, FEATURE_NAMES

def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)

def run():
    cfg = load_config()
    
    print("=" * 60)
    print("  PHASE 7 - LIVE INSPECTION")
    print("=" * 60)
    
    cam = Camera(cfg)
    inspector = Inspector(cfg)
    
    v_cfg = cfg["visualisation"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = v_cfg.get("font_scale", 0.7)
    thickness = v_cfg.get("thickness", 2)
    c_pass = tuple(v_cfg.get("present_color", [0, 200, 0]))
    c_fail = tuple(v_cfg.get("absent_color", [0, 0, 220]))
    c_pass_bg = tuple(v_cfg.get("pass_banner_color", [0, 180, 0]))
    c_fail_bg = tuple(v_cfg.get("fail_banner_color", [0, 0, 200]))
    panel_w = v_cfg.get("panel_width", 340)

    win_name = "Industrial Visual Inspection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    print("\n[INFO] Starting live loop. Press 'Q' or 'ESC' to exit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[INFO] End of camera stream.")
            break
            
        # Inspect
        res = inspector.inspect(frame)
        display = res["aligned_bgr"].copy()
        
        h, w = display.shape[:2]
        
        # Build Side Panel
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        
        # If alignment failed, abort drawing features
        if res["status"] == "ALIGN_ERROR":
            cv2.rectangle(panel, (0, 0), (panel_w, 80), (0, 0, 255), -1)
            cv2.putText(panel, "ALIGN ERROR", (20, 50), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(panel, "Reproj Err: {:.1f}px".format(res["align_quality"].get("reproj_err", 99.9)), 
                        (20, 120), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Draw PASS/FAIL banner
            banner_color = c_pass_bg if res["status"] == "PASS" else c_fail_bg
            cv2.rectangle(panel, (0, 0), (panel_w, 80), banner_color, -1)
            cv2.putText(panel, "RESULT: {}".format(res["status"]), (20, 50), 
                        font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
                        
            # Draw feature statuses
            y_offset = 140
            for feat in FEATURE_NAMES:
                f_res = res["features"].get(feat)
                if not f_res: continue
                
                # Draw box on main image
                x, y, bw, bh = f_res["box"]
                color = c_pass if f_res["present"] else c_fail
                cv2.rectangle(display, (x, y), (x+bw, y+bh), color, thickness)
                # Label box
                cv2.putText(display, feat, (x, y-10), font, font_scale-0.2, color, thickness-1, cv2.LINE_AA)
                
                # List in panel
                text = "{:<12}: {}".format(feat.replace("_", " ").title(), "OK" if f_res["present"] else "MISSING")
                cv2.putText(panel, text, (20, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(panel, "({:.0f}%)".format(f_res["prob"]*100), (300, y_offset), 
                            font, font_scale-0.2, (180, 180, 180), 1, cv2.LINE_AA)
                
                y_offset += 40
                
            # Alignment stats at bottom
            aq = res["align_quality"]
            y_stats = h - 60
            cv2.putText(panel, "Align: {}".format(aq.get("method")), (20, y_stats), 
                        font, font_scale-0.2, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(panel, "Err: {:.1f}px".format(aq.get("reproj_err", 0.0)), (180, y_stats), 
                        font, font_scale-0.2, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Combine
        out_frame = np.hstack([display, panel])
        
        cv2.imshow(win_name, out_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
