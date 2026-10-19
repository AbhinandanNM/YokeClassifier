# -*- coding: utf-8 -*-
"""
utils/excel_logger.py
=====================
Automated Industrial Shift / Inspection Logger.
Directly creates and updates real Microsoft Excel (.xlsx) workbooks per day
using the device's date and clock.

Features:
- Records exact camera source / hardware device (Phone IP Webcam, Factory BCS30HDIR50, Dahua, USB Webcam, etc.)
- Excel file: outputs/shift_logs/Inspection_Report_YYYY-MM-DD.xlsx
- Formatted headers, column widths, and cell styling (Green for PASS, Red for FAIL)
- Also writes a companion .csv for fallback compatibility
- Maintains: Total Count, Pass Count, Fail Count, and Pass Percentage.
"""

import os
import csv
from datetime import datetime
from pathlib import Path

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


def detect_camera_device_name(camera_cfg: dict) -> str:
    """Identify human-readable camera device type from config."""
    if not camera_cfg:
        return "Unknown Device"
    cam_type = str(camera_cfg.get("type", "")).lower()
    url = str(camera_cfg.get("url", "")).lower()
    
    if cam_type == "webcam":
        return f"USB/Integrated Webcam (Index {camera_cfg.get('index', 0)})"
    elif cam_type == "simulation":
        return "Simulation Test Mode (Offline Images)"
    elif cam_type in ("ip_camera", "ip_stream", "rtsp"):
        if ":8080" in url or "http://" in url:
            return "Mobile Phone IP Webcam"
        elif "192.168.1.149" in url or "snl" in url:
            return "BCS30HDIR50 Industrial 3MP IR Bullet"
        elif "192.168.0.35" in url or "192.168.1.108" in url or "dahua" in url or "realmonitor" in url:
            return "Dahua DH-IPC-HFW1230SP-S4 IP Camera"
        else:
            return f"Network IP Camera ({camera_cfg.get('url', 'RTSP')})"
    return f"Camera ({cam_type})"


class InspectionLogger:
    def __init__(self, output_dir: str | Path, camera_cfg: dict | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.camera_device = detect_camera_device_name(camera_cfg)
        
        self.xlsx_file = self.output_dir / f"Inspection_Report_{self.current_date}.xlsx"
        self.csv_file = self.output_dir / f"inspection_log_{self.current_date}.csv"
        
        self.headers = [
            "Part No.",
            "Inspection Time",
            "Camera Device",
            "Overall Result",
            "Spline",
            "Top Nib",
            "Bottom Nib",
            "Hole TL",
            "Hole TR",
            "Hole BL",
            "Hole BR",
            "Min Confidence %",
            "Reproj Error (px)"
        ]
        
        self.today_total = 0
        self.today_pass = 0
        self.today_fail = 0
        
        self._init_files()

    def _init_files(self):
        """Initialise both .xlsx and .csv log files for today's date."""
        if HAS_OPENPYXL:
            if not self.xlsx_file.exists():
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.title = "Inspection Summary"
                
                # Title block
                ws.merge_cells("A1:M1")
                ws["A1"] = f"DAILY PRODUCTION INSPECTION REPORT — {self.current_date}"
                ws["A1"].font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
                ws["A1"].fill = PatternFill(start_color="1A365D", end_color="1A365D", fill_type="solid")
                ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
                ws.row_dimensions[1].height = 32

                # Table headers
                header_fill = PatternFill(start_color="2B6CB0", end_color="2B6CB0", fill_type="solid")
                header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
                
                for col_num, header in enumerate(self.headers, 1):
                    cell = ws.cell(row=2, column=col_num, value=header)
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                ws.row_dimensions[2].height = 24
                
                # Column widths
                col_widths = [12, 20, 28, 16, 12, 12, 14, 12, 12, 12, 12, 18, 18]
                for idx, w in enumerate(col_widths, 1):
                    ws.column_dimensions[openpyxl.utils.get_column_letter(idx)].width = w

                wb.save(self.xlsx_file)
            else:
                # Read existing counts
                try:
                    wb = openpyxl.load_workbook(self.xlsx_file)
                    ws = wb.active
                    for row in ws.iter_rows(min_row=3, values_only=True):
                        if row and row[0]:
                            self.today_total += 1
                            # Overall Result is at index 3 (4th column)
                            if len(row) > 3 and str(row[3]).strip().upper() == "PASS":
                                self.today_pass += 1
                            else:
                                self.today_fail += 1
                except Exception:
                    pass

        # CSV companion
        if not self.csv_file.exists():
            with open(self.csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_inspection(self, res: dict) -> dict:
        """
        Log an inspected part record to the date-wise Excel workbook and CSV.
        """
        now_date = datetime.now().strftime("%Y-%m-%d")
        if now_date != self.current_date:
            self.current_date = now_date
            self.xlsx_file = self.output_dir / f"Inspection_Report_{self.current_date}.xlsx"
            self.csv_file = self.output_dir / f"inspection_log_{self.current_date}.csv"
            self.today_total = 0
            self.today_pass = 0
            self.today_fail = 0
            self._init_files()

        self.today_total += 1
        status = res.get("status", "FAIL")
        if status == "PASS":
            self.today_pass += 1
        else:
            self.today_fail += 1

        feats = res.get("features", {})
        min_conf = min([f.get("prob", 0.0) for f in feats.values()]) if feats else 0.0
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = [
            f"PART_{self.today_total:04d}",
            time_str,
            self.camera_device,
            status,
            "PASS" if feats.get("spline", {}).get("present") else "FAIL",
            "PASS" if feats.get("top_nib", {}).get("present") else "FAIL",
            "PASS" if feats.get("bottom_nib", {}).get("present") else "FAIL",
            "PASS" if feats.get("hole_tl", {}).get("present") else "FAIL",
            "PASS" if feats.get("hole_tr", {}).get("present") else "FAIL",
            "PASS" if feats.get("hole_bl", {}).get("present") else "FAIL",
            "PASS" if feats.get("hole_br", {}).get("present") else "FAIL",
            round(min_conf * 100, 1),
            round(res.get("align_quality", {}).get("reproj_err", 0.0), 2)
        ]

        # Write to .xlsx Excel Workbook
        if HAS_OPENPYXL:
            try:
                wb = openpyxl.load_workbook(self.xlsx_file)
                ws = wb.active
                
                target_row = ws.max_row + 1
                pass_fill = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")
                fail_fill = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
                pass_font = Font(name="Calibri", size=11, bold=True, color="155724")
                fail_font = Font(name="Calibri", size=11, bold=True, color="721C24")

                for col_idx, val in enumerate(row, 1):
                    cell = ws.cell(row=target_row, column=col_idx, value=val)
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    # Color highlight Overall Result (Column 4)
                    if col_idx == 4:
                        if val == "PASS":
                            cell.fill = pass_fill
                            cell.font = pass_font
                        else:
                            cell.fill = fail_fill
                            cell.font = fail_font

                ws.row_dimensions[target_row].height = 20
                try:
                    wb.save(self.xlsx_file)
                except PermissionError:
                    # Operator has the Excel file open in Microsoft Excel!
                    # Save to a temporary rollover copy so zero data is lost
                    backup_path = self.output_dir / f"Inspection_Report_{self.current_date}_live.xlsx"
                    try:
                        wb.save(backup_path)
                        print(f"[InspectionLogger] Notice: Primary Excel file is open in Excel. Appended to: {backup_path.name}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"[InspectionLogger] Excel write error: {e}")

        # Write to CSV
        try:
            with open(self.csv_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception:
            pass

        pass_rate = (self.today_pass / self.today_total * 100) if self.today_total > 0 else 0.0
        return {
            "total": self.today_total,
            "pass": self.today_pass,
            "fail": self.today_fail,
            "pass_rate": pass_rate,
            "camera_device": self.camera_device,
            "xlsx_path": str(self.xlsx_file),
            "csv_path": str(self.csv_file)
        }
