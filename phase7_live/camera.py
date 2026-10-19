# -*- coding: utf-8 -*-
"""
phase7_live/camera.py
======================
Camera abstraction. Handles grabbing frames from a live webcam,
an industrial USB camera, or a simulated folder of images for testing.
"""

import cv2
import time
from pathlib import Path
import threading

class Camera:
    def __init__(self, config: dict):
        self.cfg = config["camera"]
        self.type = self.cfg.get("type", "webcam").lower()
        self.cap = None
        
        if self.type == "webcam":
            index = self.cfg.get("index", 0)
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.get("width", 1920))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.get("height", 1080))
            # Disable autofocus on all webcams / USB industrial cameras
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam at index {}".format(index))
            self._start_thread()
                
        elif self.type == "ip_camera":
            url = self.cfg.get("url", "")
            import os
            # Set TCP transport for RTSP streams to eliminate packet drops/jitter
            if "rtsp://" in url.lower():
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            # Use CAP_FFMPEG for network streams if supported
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG) if hasattr(cv2, 'CAP_FFMPEG') else cv2.VideoCapture(url)
            if not self.cap.isOpened():
                # Fallback to standard open
                self.cap = cv2.VideoCapture(url)
                if not self.cap.isOpened():
                    raise RuntimeError("Could not open IP camera at {}".format(url))
            
            # Disable autofocus across all IP cameras / phone webcams
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass
            # Apply the requested resolution so RTSP decoder matches config.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.get("width",  2048))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.get("height", 1536))

            self._start_thread()
                
        elif self.type == "simulation":
            # Just loop through test images
            sim_dir = Path(config["paths"]["aligned_dir"])
            self.sim_files = sorted(sim_dir.glob("*.jpg"))
            if not self.sim_files:
                raise RuntimeError("No images found in simulation directory.")
            self.sim_idx = 0
            print("[Camera] Running in simulation mode ({} images)".format(len(self.sim_files)))
            
        else:
            raise ValueError("Unknown camera type: {}".format(self.type))

    def _start_thread(self):
        self.stopped = False
        self.ret, self.frame = self.cap.read()
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.cap.read()
            if ret:
                self.ret = True
                self.frame = frame
            else:
                self.ret = False
                if self.type == "ip_camera":
                    time.sleep(1.0)
                    url = self.cfg.get("url", "")
                    try:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(url)
                    except:
                        pass

    def read(self):
        """Returns (success, frame_bgr)"""
        if self.type in ("webcam", "ip_camera"):
            return self.ret, self.frame
            
        elif self.type == "simulation":
            if self.sim_idx >= len(self.sim_files):
                self.sim_idx = 0
            
            fp = self.sim_files[self.sim_idx]
            frame = cv2.imread(str(fp))
            self.sim_idx += 1
            # Simulate framerate
            time.sleep(1.0 / self.cfg.get("fps", 30))
            return (frame is not None, frame)

    def read_roi(self, crop_frac: float = 0.80):
        """Returns (success, frame_bgr) cropped to the central crop_frac
        of the full frame. Discards distracting edge regions caused by
        the BCS30HDIR50's wide-angle IR lens."""
        ret, frame = self.read()
        if not ret or frame is None:
            return ret, frame
        h, w = frame.shape[:2]
        margin_x = int(w * (1.0 - crop_frac) / 2)
        margin_y = int(h * (1.0 - crop_frac) / 2)
        cropped = frame[margin_y:h - margin_y, margin_x:w - margin_x]
        return True, cropped

    def release(self):
        if self.type in ("webcam", "ip_camera"):
            self.stopped = True
            if hasattr(self, 'thread'):
                self.thread.join()
        if self.cap is not None:
            self.cap.release()
