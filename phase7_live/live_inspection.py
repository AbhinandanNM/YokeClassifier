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
import time
from pathlib import Path
import cv2
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase7_live.camera import Camera
from phase7_live.inspector import Inspector, FEATURE_NAMES
from phase7_live.excel_logger import DailyExcelLogger

def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)

def run():
    cfg = load_config()
    
    print("=" * 60)
    print("  PHASE 7 - LIVE INSPECTION & LOGGING SYSTEM")
    print("=" * 60)
    
    cam = Camera(cfg)
    inspector = Inspector(cfg)
    logger = DailyExcelLogger()
    
    v_cfg = cfg["visualisation"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = v_cfg.get("font_scale", 0.7)
    thickness = v_cfg.get("thickness", 2)
    c_pass = tuple(v_cfg.get("present_color", [0, 200, 0]))
    c_fail = tuple(v_cfg.get("absent_color", [0, 0, 220]))
    c_pass_bg = tuple(v_cfg.get("pass_banner_color", [0, 180, 0]))
    c_fail_bg = tuple(v_cfg.get("fail_banner_color", [0, 0, 200]))
    panel_w = v_cfg.get("panel_width", 400)

    win_name = "Industrial Visual Inspection System"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    device_name = "Mobile IP Cam" if "8080" in cfg.get("camera", {}).get("url", "") else "Factory CCTV"

    # Counters and logging debounce state
    counts = {
        "total_pass": logger.total_pass,
        "total_fail": logger.total_fail,
        "total_inspections": logger.total_inspections
    }
    
    # =========================================================================
    # Industrial State Machine:
    # WAITING_FOR_PART -> PART_DETECTED -> INSPECTING -> PASS/FAIL (RECORDED) -> WAITING_FOR_REMOVAL
    # =========================================================================
    STATE_WAITING_FOR_PART    = "WAITING_FOR_PART"
    STATE_PART_DETECTED       = "PART_DETECTED"
    STATE_INSPECTING          = "INSPECTING"
    STATE_RECORDED            = "RECORDED"
    STATE_WAITING_FOR_REMOVAL = "WAITING_FOR_REMOVAL"

    current_state = STATE_WAITING_FOR_PART
    current_result_display = None
    last_recorded_status = None
    inspect_candidates = []
    
    # Stability counters
    present_streak = 0
    absent_streak = 0
    REQUIRED_DETECT_FRAMES = 2   # Frames needed to confirm part arrival
    REQUIRED_REMOVE_FRAMES = 2   # Frames needed to confirm part removal

    print(f"\n[INFO] Excel Log initialized at: {logger.filepath}")
    print("[INFO] Industrial State Machine Active:")
    print("       WAITING_FOR_PART -> PART_DETECTED -> INSPECTING -> PASS/FAIL -> WAITING_FOR_REMOVAL")
    print("[INFO] Shortcuts: 'SPACE' to force-log, 'R' to reset daily counters, 'Q' or 'ESC' to exit.")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            cv2.waitKey(30)
            continue

        # Inspect current frame
        res = inspector.inspect(frame)
        is_part_valid = (res["status"] in ("PASS", "FAIL") and res["align_quality"].get("success", False))

        # Update presence / absence streaks
        if is_part_valid:
            present_streak += 1
            absent_streak = 0
        else:
            absent_streak += 1
            present_streak = 0

        # ---------------- State Machine Transitions ----------------
        if current_state == STATE_WAITING_FOR_PART:
            if present_streak >= REQUIRED_DETECT_FRAMES:
                current_state = STATE_PART_DETECTED

        elif current_state == STATE_PART_DETECTED:
            # Component placed and settling -> Trigger inspection window
            current_state = STATE_INSPECTING
            inspect_candidates = []

        elif current_state == STATE_INSPECTING:
            if is_part_valid:
                inspect_candidates.append(res)
                # Require 4 settled frames so inspection triggers quickly while ensuring part is steady
                if len(inspect_candidates) >= 4:
                    # Search all settled frames: if any frame achieved PASS, the part is verified PASS!
                    best_res = inspect_candidates[-1]
                    for cand in inspect_candidates:
                        if cand.get("status") == "PASS":
                            best_res = cand
                            break
                    else:
                        # If none were PASS, pick the frame with the maximum number of present features
                        best_res = max(
                            inspect_candidates,
                            key=lambda c: sum(1 for f in c.get("features", {}).values() if f.get("present"))
                        )

                    # Pass the aligned visual frame so defect images can be embedded in Excel on FAIL
                    counts = logger.log_inspection(device_name, best_res, display_frame=best_res.get("aligned_bgr"))
                    last_recorded_status = best_res["status"]
                    current_result_display = best_res
                    current_state = STATE_RECORDED
                    
                    print(f"\n========================================================")
                    print(f" >>> [INSPECTION RECORDED] Result: {last_recorded_status}")
                    print(f" >>> PASS COUNT : {counts['total_pass']}")
                    print(f" >>> FAIL COUNT : {counts['total_fail']}")
                    print(f" >>> TOTAL PARTS: {counts['total_inspections']}")
                    print(f"========================================================\n")
            else:
                # Part moved out of frame or lost alignment before settling
                inspect_candidates = []
                current_state = STATE_WAITING_FOR_PART

        elif current_state == STATE_RECORDED:
            # Inspection done. Part is still sitting there. Transition to waiting for removal.
            current_state = STATE_WAITING_FOR_REMOVAL

        elif current_state == STATE_WAITING_FOR_REMOVAL:
            # Maintain the result on screen, DO NOT re-inspect or increment count
            # Only reset once the part is physically removed (absent for REQUIRED_REMOVE_FRAMES)
            if absent_streak >= REQUIRED_REMOVE_FRAMES:
                current_state = STATE_WAITING_FOR_PART
                current_result_display = None
                print("[INFO] Part removed from fixture -> Ready for next component!")

        # ---------------- Display Preparation ----------------
        # If in WAITING_FOR_REMOVAL, keep displaying the inspected boxes
        active_res = res if is_part_valid else (current_result_display if current_result_display else res)
        display = active_res["aligned_bgr"].copy()
        
        if display.shape[:2] != (inspector.aligner.out_h, inspector.aligner.out_w):
            display = cv2.resize(display, (inspector.aligner.out_w, inspector.aligner.out_h), interpolation=cv2.INTER_AREA)

        h, w = display.shape[:2]
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

        # 1. Top Banner — reflects State & PASS/FAIL
        if current_state == STATE_WAITING_FOR_PART:
            cv2.rectangle(panel, (0, 0), (panel_w, 75), (80, 80, 80), -1)
            cv2.putText(panel, "WAITING FOR PART", (20, 48), font, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "Place component on fixture", (20, 115), font, 0.65, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "Camera: Ready", (20, 145), font, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        elif current_state in (STATE_PART_DETECTED, STATE_INSPECTING):
            cv2.rectangle(panel, (0, 0), (panel_w, 75), (0, 140, 255), -1)
            cv2.putText(panel, "INSPECTING...", (20, 48), font, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        elif current_state in (STATE_RECORDED, STATE_WAITING_FOR_REMOVAL):
            disp_status = last_recorded_status if last_recorded_status else active_res["status"]
            banner_color = c_pass_bg if disp_status == "PASS" else c_fail_bg
            cv2.rectangle(panel, (0, 0), (panel_w, 75), banner_color, -1)
            cv2.putText(panel, f"RESULT: {disp_status}", (20, 50), font, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

            # Draw feature status list & boxes
            y_offset = 120
            for feat in FEATURE_NAMES:
                f_res = active_res["features"].get(feat)
                if not f_res: 
                    continue
                
                # Main frame box
                x, y, bw, bh = f_res["box"]
                color = c_pass if f_res["present"] else c_fail
                cv2.rectangle(display, (x, y), (x+bw, y+bh), color, thickness)
                cv2.putText(display, feat, (x, y-10), font, font_scale-0.2, color, thickness-1, cv2.LINE_AA)

                # Panel list
                text = f"{feat.replace('_', ' ').title():<12}: {'OK' if f_res['present'] else 'MISSING'}"
                cv2.putText(panel, text, (20, y_offset), font, 0.62, color, 2, cv2.LINE_AA)
                cv2.putText(panel, f"({f_res['prob']*100:.0f}%)", (310, y_offset), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
                y_offset += 36

        # 2. Daily Counter Box (Pass / Fail / Total)
        counter_y = h - 235
        cv2.rectangle(panel, (10, counter_y), (panel_w - 10, h - 75), (30, 30, 30), -1)
        cv2.rectangle(panel, (10, counter_y), (panel_w - 10, h - 75), (75, 75, 75), 1)
        
        cv2.putText(panel, "PRODUCTION COUNTER", (20, counter_y + 25), font, 0.60, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"PASS COUNT : {counts['total_pass']}", (20, counter_y + 55), font, 0.68, (0, 240, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, f"FAIL COUNT : {counts['total_fail']}", (20, counter_y + 85), font, 0.68, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"TOTAL PARTS: {counts['total_inspections']}", (20, counter_y + 115), font, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        # Worker action guidance badge
        if current_state == STATE_WAITING_FOR_REMOVAL:
            cv2.putText(panel, "[ACTION: REMOVE PART]", (20, counter_y + 142), font, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
        elif current_state == STATE_WAITING_FOR_PART:
            cv2.putText(panel, "[ACTION: PLACE PART]", (20, counter_y + 142), font, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "[HOLD STEADY...]", (20, counter_y + 142), font, 0.52, (255, 200, 0), 1, cv2.LINE_AA)

        # 3. Bottom Device & Excel Info
        cv2.putText(panel, f"Device : {device_name}", (20, h - 45), font, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Excel  : Auto-Logged (R=Reset)", (20, h - 20), font, 0.48, (0, 200, 200), 1, cv2.LINE_AA)
        
        # Combine
        out_frame = np.hstack([display, panel])
        cv2.imshow(win_name, out_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            # Reset counters
            logger.total_pass = 0
            logger.total_fail = 0
            logger.total_inspections = 0
            counts = {"total_pass": 0, "total_fail": 0, "total_inspections": 0}
            print("\n[COUNTER RESET] Daily production counters reset to 0.")
        elif key == 32:  # SPACE bar manual force-log
            counts = logger.log_inspection(device_name, active_res)
            print(f"[MANUAL LOG] Pass: {counts['total_pass']}, Fail: {counts['total_fail']}, Total: {counts['total_inspections']}")
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
