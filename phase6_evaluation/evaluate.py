# -*- coding: utf-8 -*-
"""
phase6_evaluation/evaluate.py
==============================
Evaluates the trained binary classifiers.
Computes accuracy, precision, recall, F1-score, and plots confusion matrices.

Usage
-----
    python phase6_evaluation/evaluate.py
"""

import sys
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase5_training.dataset import ROIDataset
from phase5_training.model import build_model
from phase5_training.augmentations import get_val_transforms

def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)

def evaluate_model(feature_group: str, cfg: dict):
    print("\n" + "="*50)
    print(" EVALUATING MODEL: {}".format(feature_group.upper()))
    print("="*50)
    
    t_cfg = cfg["training"]
    data_dir = cfg["paths"]["roi_dataset_dir"]
    models_dir = Path(cfg["paths"]["models_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    
    model_path = models_dir / "{}.pth".format(feature_group)
    if not model_path.exists():
        print("  [ERROR] Model not found:", model_path)
        return
        
    device_str = t_cfg.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    
    # We evaluate on the test split that we created during training.
    # To be perfectly rigorous we should save the test split indices, 
    # but since we set a fixed seed (42), we can just recreate the split here.
    full_dataset = ROIDataset(data_dir, feature_group, transform=None)
    
    val_split = t_cfg.get("val_split", 0.15)
    test_split = t_cfg.get("test_split", 0.10)
    
    n_total = len(full_dataset)
    n_val = int(val_split * n_total)
    n_test = int(test_split * n_total)
    n_train = n_total - n_val - n_test
    
    generator = torch.Generator().manual_seed(t_cfg.get("seed", 42))
    _, _, test_sub = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    
    # We wrap test_sub with val_transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            import cv2
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: img = np.zeros((128, 128), dtype=np.uint8)
            img_tensor = torch.from_numpy(img[:, :, np.newaxis]).permute(2, 0, 1)
            img_tensor = self.transform(img_tensor)
            return img_tensor, label
        def __len__(self):
            return len(self.subset)
            
    img_size = t_cfg.get("image_size", 128)
    test_ds = TransformedSubset(test_sub, get_val_transforms(img_size))
    
    nw = 0 if sys.platform == "win32" else t_cfg.get("num_workers", 4)
    test_loader = DataLoader(test_ds, batch_size=t_cfg.get("batch_size", 32), shuffle=False, num_workers=nw)
    
    model = build_model(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Metrics
    report = classification_report(all_labels, all_preds, target_names=["Absent", "Present"], zero_division=0)
    print("\nClassification Report:\n")
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save Confusion Matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix: {}'.format(feature_group.upper()))
    
    cm_path = outputs_dir / "cm_{}.png".format(feature_group)
    plt.savefig(str(cm_path))
    plt.close()
    
    print("  Saved confusion matrix -> {}".format(cm_path))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="all")
    args = parser.parse_args()
    
    cfg = load_config()
    features = cfg["training"]["features"]
    
    if args.feature != "all":
        if args.feature in features:
            features = [args.feature]
        else:
            print("Unknown feature:", args.feature)
            sys.exit(1)
            
    for feat in features:
        evaluate_model(feat, cfg)

if __name__ == "__main__":
    main()
