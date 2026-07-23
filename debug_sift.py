import cv2
import numpy as np
import yaml
from pathlib import Path

def test_sift():
    ref_path = "perfect/perfect_full_0000.jpg"
    bgr = cv2.imread(ref_path)
    if bgr is None:
        print("Cannot load reference image")
        return

    sift = cv2.SIFT_create(5000)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    kp, desc = sift.detectAndCompute(gray, None)
    print(f"Detected {len(kp)} keypoints")
    
    if desc is not None:
        print(f"Descriptor shape: {desc.shape}, dtype: {desc.dtype}")
        
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(desc, desc, k=2)
    
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
    print(f"Good matches: {len(good)}")

test_sift()
