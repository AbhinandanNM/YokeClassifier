# -*- coding: utf-8 -*-
"""
phase5_training/augmentations.py
=================================
Torchvision pipeline for training the ROI classifiers.
Applies transformations as recommended in the Phase 1 dataset analysis.
"""

from torchvision.transforms import v2
import torch

def get_train_transforms(image_size: int = 128):
    return v2.Compose([
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.RandomAffine(degrees=35, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        # Convert to float [0, 1] then normalize to [-1, 1]
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

def get_val_transforms(image_size: int = 128):
    return v2.Compose([
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5])
    ])
