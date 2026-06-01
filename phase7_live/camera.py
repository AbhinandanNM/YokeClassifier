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
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam at index {}".format(index))
            self._start_thread()
                
        elif self.type == "ip_camera":
            url = self.cfg.get("url", "")
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open IP camera at {}".format(url))
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
            self.ret, self.frame = self.cap.read()

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

    def release(self):
        if self.type in ("webcam", "ip_camera"):
            self.stopped = True
            if hasattr(self, 'thread'):
                self.thread.join()
        if self.cap is not None:
            self.cap.release()
