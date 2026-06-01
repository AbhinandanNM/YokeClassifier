# -*- coding: utf-8 -*-
"""
phase5_training/dataset.py
===========================
PyTorch Dataset for ROI feature classification.
Loads crops from `present/` (label 1) and `absent/` (label 0).
"""

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class ROIDataset(Dataset):
    def __init__(self, data_dir: str, feature_group: str, transform=None):
        """
        data_dir: e.g. "C:/Users/Abhinandan/OneDrive/Desktop/Minor/inspection_system/roi_dataset"
        feature_group: "spline", "top_nib", "bottom_nib", or "hole"
        """
        self.data_dir = Path(data_dir) / feature_group
        self.transform = transform
        
        self.samples: List[Tuple[Path, int]] = []
        
        present_dir = self.data_dir / "present"
        if present_dir.exists():
            for fp in present_dir.glob("*.jpg"):
                self.samples.append((fp, 1))
                
        absent_dir = self.data_dir / "absent"
        if absent_dir.exists():
            for fp in absent_dir.glob("*.jpg"):
                self.samples.append((fp, 0))
                
        if not self.samples:
            raise RuntimeError("No data found in {}".format(self.data_dir))
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback if corrupt
            img = np.zeros((128, 128), dtype=np.uint8)
            
        # Add channel dimension
        img = img[:, :, np.newaxis]
        
        # Convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        else:
            img_tensor = img_tensor.float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            
        return img_tensor, label
