# -*- coding: utf-8 -*-
"""
phase7_live/live_inspection.py
===============================
Phase 7 - Industrial Live Inspection & Production Counter System.

Upgrades:
1. Exact Timing Control:
   - 5.0s Inspection Settling Window while part is placed
   - 5.0s Result Display Freeze for operator verification
   - Instant Reset (< 0.2s) when part is physically removed
2. High-FPS Fluid Display on CPU:
   - Optimized SIFT matching and frame processing
3. Interactive Live ROI Box Calibration Mode (Hotkey: 'C'):
   - Real-time visual box adjustment directly over the live stream
   - Arrow keys / WASD to move and resize boxes
   - TAB/N to cycle features
   - S/ENTER to save to config.yaml and reload dynamically without restarting!
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

CONFIG_FILE = ROOT / "config" / "config.yaml"

def load_config() -> dict:
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def save_roi_config(cfg: dict, rois: dict):
    """Save calibrated ROI coordinates directly back to config.yaml."""
    cfg["roi"].update(rois)
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print("\n[SUCCESS] Calibrated ROI coordinates saved to config/config.yaml!")

def run():
    cfg = load_config()
    
    print("=" * 60)
    print("  PHASE 7 - INDUSTRIAL LIVE INSPECTION & CALIBRATION SYSTEM")
    print("=" * 60)
    
    cam = Camera(cfg)
    inspector = Inspector(cfg)
    logger = DailyExcelLogger()
    
    v_cfg = cfg.get("visualisation", {})
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

    # Counters
    counts = {
        "total_pass": logger.total_pass,
        "total_fail": logger.total_fail,
        "total_inspections": logger.total_inspections
    }
    
    # State Machine
    STATE_WAITING_FOR_PART    = "WAITING_FOR_PART"
    STATE_PART_DETECTED       = "PART_DETECTED"
    STATE_INSPECTING          = "INSPECTING"
    STATE_RECORDED            = "RECORDED"
    STATE_WAITING_FOR_REMOVAL = "WAITING_FOR_REMOVAL"

    current_state = STATE_WAITING_FOR_PART
    current_result_display = None
    last_recorded_status = None
    inspect_candidates = []

    # Timing parameters
    INSPECT_WINDOW_SEC = 5.0     # 5.0s inspection window
    RESULT_FREEZE_SEC  = 3.5     # 3.5s result display freeze
    
    inspect_start_time = 0.0
    recorded_time = 0.0

    # Presence streaks for instant reset
    present_streak = 0
    absent_streak = 0
    REQUIRED_DETECT_FRAMES = 2
    REQUIRED_REMOVE_FRAMES = 1   # Instant reset (<0.2s) when part lifted

    # Calibration Mode Variables
    calib_mode = False
    calib_idx = 0
    calib_rois = {}
    for feat in FEATURE_NAMES:
        calib_rois[feat] = list(cfg["roi"].get(feat, [100, 100, 150, 150]))

    print(f"\n[INFO] Excel Log active: {logger.filepath}")
    print("[INFO] Controls:")
    print("       'C'     : Enter/Exit Interactive ROI Calibration Mode")
    # Mouse Dragging Variables for ROI Calibration
    drag_state = {
        "dragging": False,
        "mode": None,  # 'move' or 'resize'
        "feat": None,
        "start_x": 0,
        "start_y": 0,
        "orig_box": None
    }

    def on_mouse(event, mx, my, flags, param):
        if not calib_mode:
            return

        nonlocal calib_idx
        active_f = FEATURE_NAMES[calib_idx]

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked on a corner handle of active box to resize
            bx, by, bw, bh = calib_rois[active_f]
            handle_size = 25
            if abs(mx - (bx + bw)) <= handle_size and abs(my - (by + bh)) <= handle_size:
                drag_state["dragging"] = True
                drag_state["mode"] = "resize"
                drag_state["feat"] = active_f
                drag_state["start_x"] = mx
                drag_state["start_y"] = my
                drag_state["orig_box"] = list(calib_rois[active_f])
                return

            # Check if clicked inside any box to select and drag it
            for idx, feat in enumerate(FEATURE_NAMES):
                x, y, w, h = calib_rois[feat]
                if x <= mx <= x + w and y <= my <= y + h:
                    calib_idx = idx
                    drag_state["dragging"] = True
                    drag_state["mode"] = "move"
                    drag_state["feat"] = feat
                    drag_state["start_x"] = mx
                    drag_state["start_y"] = my
                    drag_state["orig_box"] = list(calib_rois[feat])
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            if drag_state["dragging"]:
                f = drag_state["feat"]
                ox, oy, ow, oh = drag_state["orig_box"]
                dx = mx - drag_state["start_x"]
                dy = my - drag_state["start_y"]

                if drag_state["mode"] == "move":
                    calib_rois[f][0] = max(0, min(1080 - ow, ox + dx))
                    calib_rois[f][1] = max(0, min(1080 - oh, oy + dy))
                elif drag_state["mode"] == "resize":
                    calib_rois[f][2] = max(20, min(1080 - ox, ow + dx))
                    calib_rois[f][3] = max(20, min(1080 - oy, oh + dy))

        elif event == cv2.EVENT_LBUTTONUP:
            if drag_state["dragging"]:
                drag_state["dragging"] = False
                drag_state["mode"] = None
                drag_state["feat"] = None

    cv2.setMouseCallback(win_name, on_mouse)

    print(f"\n[INFO] Excel Log active: {logger.filepath}")
    print("[INFO] Controls:")
    print("       'C'     : Enter/Exit Interactive ROI Calibration Mode (MOUSE DRAGGABLE!)")
    print("       'SPACE' : Manual force-log")
    print("       'R'     : Reset daily production counters to 0")
    print("       'Q'/ESC : Exit cleanly\n")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            cv2.waitKey(20)
            continue

        # If in ROI calibration mode, display calibration interface
        if calib_mode:
            aligned_calib, _ = inspector.aligner.align(frame)
            disp_calib = aligned_calib.copy()
            if disp_calib.shape[:2] != (inspector.aligner.out_h, inspector.aligner.out_w):
                disp_calib = cv2.resize(disp_calib, (inspector.aligner.out_w, inspector.aligner.out_h))
            
            ch, cw = disp_calib.shape[:2]
            calib_panel = np.zeros((ch, panel_w, 3), dtype=np.uint8)

            active_feat = FEATURE_NAMES[calib_idx]
            for idx, feat in enumerate(FEATURE_NAMES):
                bx, by, bw, bh = calib_rois[feat]
                is_active = (feat == active_feat)
                box_color = (0, 255, 255) if is_active else (80, 80, 80)
                box_thick = 3 if is_active else 1
                cv2.rectangle(disp_calib, (bx, by), (bx + bw, by + bh), box_color, box_thick)
                cv2.putText(disp_calib, feat, (bx, by - 6), font, 0.55, box_color, 1, cv2.LINE_AA)
                
                # Draw a draggable bottom-right resize corner handle on the active box
                if is_active:
                    handle_sz = 14
                    cv2.rectangle(disp_calib, (bx + bw - handle_sz, by + bh - handle_sz), (bx + bw, by + bh), (0, 255, 255), -1)

            # Calibration sidebar
            cv2.rectangle(calib_panel, (0, 0), (panel_w, 75), (0, 140, 255), -1)
            cv2.putText(calib_panel, "ROI CALIBRATION", (20, 48), font, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(calib_panel, f"Active: {active_feat.upper()}", (20, 115), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            bx, by, bw, bh = calib_rois[active_feat]
            cv2.putText(calib_panel, f"X: {bx}  Y: {by}", (20, 150), font, 0.60, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(calib_panel, f"W: {bw}  H: {bh}", (20, 180), font, 0.60, (200, 200, 200), 1, cv2.LINE_AA)

            y_inst = 230
            cv2.putText(calib_panel, "MOUSE CONTROLS:", (20, y_inst), font, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
            instructions = [
                "- Click & Drag inside box to MOVE",
                "- Drag Yellow Bottom-Right Corner to RESIZE",
                "- Click any other box to SELECT it",
                "- Press ENTER: Save to config.yaml",
                "- Press ESC or 'C': Exit Calibration"
            ]
            for inst in instructions:
                y_inst += 32
                cv2.putText(calib_panel, inst, (20, y_inst), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

            calib_out = np.hstack([disp_calib, calib_panel])
            cv2.imshow(win_name, calib_out)

            k = cv2.waitKey(20) & 0xFF
            step = 5
            if k in (ord('q'), ord('c'), ord('C'), 27):
                calib_mode = False
                print("[INFO] Exited ROI Calibration Mode.")
            elif k in (ord('n'), ord('N'), 9):
                calib_idx = (calib_idx + 1) % len(FEATURE_NAMES)
            elif k == 13:  # ENTER to save
                save_roi_config(cfg, calib_rois)
                inspector.extractor.rois = {name: tuple(calib_rois[name]) for name in FEATURE_NAMES}
            continue

        # ---------------- MAIN OPERATOR WORKFLOW ----------------
        # 1. WAITING_FOR_PART: Show live camera view
        if current_state == STATE_WAITING_FOR_PART:
            display = frame.copy()
            if display.shape[:2] != (inspector.aligner.out_h, inspector.aligner.out_w):
                display = cv2.resize(display, (inspector.aligner.out_w, inspector.aligner.out_h), interpolation=cv2.INTER_AREA)

            h, w = display.shape[:2]
            panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

            cv2.rectangle(panel, (0, 0), (panel_w, 75), (80, 80, 80), -1)
            cv2.putText(panel, "READY TO INSPECT", (20, 48), font, 0.90, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "Place component under cam", (20, 115), font, 0.60, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "Press ENTER to Inspect", (20, 145), font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            # Production Counter Box
            counter_y = h - 245
            cv2.rectangle(panel, (10, counter_y), (panel_w - 10, h - 65), (30, 30, 30), -1)
            cv2.rectangle(panel, (10, counter_y), (panel_w - 10, h - 65), (75, 75, 75), 1)
            cv2.putText(panel, "PRODUCTION COUNTER", (20, counter_y + 25), font, 0.60, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"PASS COUNT : {counts['total_pass']}", (20, counter_y + 55), font, 0.68, (0, 240, 0), 2, cv2.LINE_AA)
            cv2.putText(panel, f"FAIL COUNT : {counts['total_fail']}", (20, counter_y + 85), font, 0.68, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"TOTAL PARTS: {counts['total_inspections']}", (20, counter_y + 115), font, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "[ACTION: HIT ENTER TO INSPECT]", (20, counter_y + 142), font, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(panel, f"Device : {device_name}", (20, h - 42), font, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(panel, f"Press 'C': Calibrate ROIs", (20, h - 20), font, 0.48, (0, 240, 255), 1, cv2.LINE_AA)

        # 2. RECORDED: Result screen frozen with PASS/FAIL details
        elif current_state == STATE_RECORDED:
            display = current_result_display["aligned_bgr"].copy()
            if display.shape[:2] != (inspector.aligner.out_h, inspector.aligner.out_w):
                display = cv2.resize(display, (inspector.aligner.out_w, inspector.aligner.out_h), interpolation=cv2.INTER_AREA)

            h, w = display.shape[:2]
            panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

            disp_status = last_recorded_status if last_recorded_status else current_result_display["status"]
            banner_color = c_pass_bg if disp_status == "PASS" else c_fail_bg
            cv2.rectangle(panel, (0, 0), (panel_w, 80), banner_color, -1)
            cv2.putText(panel, f"RESULT: {disp_status}", (20, 42), font, 1.05, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(panel, "[PRESS ENTER TO RESET]", (20, 68), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw feature status list & boxes
            y_offset = 120
            for feat in FEATURE_NAMES:
                f_res = current_result_display["features"].get(feat)
                if not f_res:
                    continue
                x, y, bw, bh = f_res["box"]
                color = c_pass if f_res["present"] else c_fail
                cv2.rectangle(display, (x, y), (x + bw, y + bh), color, thickness)
                cv2.putText(display, feat, (x, y - 10), font, font_scale - 0.2, color, thickness - 1, cv2.LINE_AA)

                text = f"{feat.replace('_', ' ').title():<12}: {'OK' if f_res['present'] else 'MISSING'}"
                cv2.putText(panel, text, (20, y_offset), font, 0.62, color, 2, cv2.LINE_AA)
                cv2.putText(panel, f"({f_res['prob']*100:.0f}%)", (310, y_offset), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
                y_offset += 36

            # Production Counter Box
            counter_y = h - 245
            cv2.rectangle(panel, (10, counter_y), (panel_w - 10, h - 65), (30, 30, 30), -1)
            cv2.rectangle(panel, (10, counter_y), (panel_w - 10, h - 65), (75, 75, 75), 1)
            cv2.putText(panel, "PRODUCTION COUNTER", (20, counter_y + 25), font, 0.60, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"PASS COUNT : {counts['total_pass']}", (20, counter_y + 55), font, 0.68, (0, 240, 0), 2, cv2.LINE_AA)
            cv2.putText(panel, f"FAIL COUNT : {counts['total_fail']}", (20, counter_y + 85), font, 0.68, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"TOTAL PARTS: {counts['total_inspections']}", (20, counter_y + 115), font, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "[HIT ENTER TO RESET / NEXT]", (20, counter_y + 142), font, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(panel, f"Device : {device_name}", (20, h - 42), font, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(panel, f"Press 'C': Calibrate ROIs", (20, h - 20), font, 0.48, (0, 240, 255), 1, cv2.LINE_AA)

        # Combine
        out_frame = np.hstack([display, panel])
        cv2.imshow(win_name, out_frame)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (10, 13):  # ENTER key hit
            if current_state == STATE_WAITING_FOR_PART:
                # 1. Perform Inspection
                res = inspector.inspect(frame)
                counts = logger.log_inspection(device_name, res, display_frame=res.get("aligned_bgr"))
                last_recorded_status = res["status"]
                current_result_display = res
                current_state = STATE_RECORDED

                print(f"\n========================================================")
                print(f" >>> [INSPECTION RECORDED] Result: {last_recorded_status}")
                print(f" >>> PASS COUNT : {counts['total_pass']}")
                print(f" >>> FAIL COUNT : {counts['total_fail']}")
                print(f" >>> TOTAL PARTS: {counts['total_inspections']}")
                print(f"========================================================\n")
            elif current_state == STATE_RECORDED:
                # 2. Reset back to live view ready for next part
                current_state = STATE_WAITING_FOR_PART
                current_result_display = None
                last_recorded_status = None
                print("[INFO] Reset -> Ready for next part inspection!")

        elif key in (ord('c'), ord('C')):
            calib_mode = True
            print("\n>>> [ROI CALIBRATION MODE ACTIVATED]")
            print("    Click & drag boxes with mouse, drag corner to resize, ENTER to save.")
        elif key in (ord('r'), ord('R')):
            logger.total_pass = 0
            logger.total_fail = 0
            logger.total_inspections = 0
            counts = {"total_pass": 0, "total_fail": 0, "total_inspections": 0}
            print("\n[COUNTER RESET] Daily production counters reset to 0.")
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
