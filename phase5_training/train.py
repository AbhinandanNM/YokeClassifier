# -*- coding: utf-8 -*-
"""
phase5_training/train.py
=========================
Trains binary classifiers for all features (spline, top_nib, bottom_nib, hole).
Uses Early Stopping and saves the best model checkpoint.

Usage
-----
    python phase5_training/train.py
"""

import sys
from pathlib import Path
import time
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase5_training.dataset import ROIDataset
from phase5_training.model import build_model
from phase5_training.augmentations import get_train_transforms, get_val_transforms

def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)

def train_model_for_feature(feature_group: str, cfg: dict):
    print("\n" + "="*50)
    print(" TRAINING MODEL: {}".format(feature_group.upper()))
    print("="*50)
    
    t_cfg = cfg["training"]
    data_dir = cfg["paths"]["roi_dataset_dir"]
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    device_str = t_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    print("  Device:", device)
    
    # 1. Dataset & Dataloaders
    batch_size = t_cfg.get("batch_size", 32)
    img_size = t_cfg.get("image_size", 128)
    
    # We load all data with val_transforms first to split, then wrap train subset
    full_dataset = ROIDataset(data_dir, feature_group, transform=None)
    
    val_split = t_cfg.get("val_split", 0.15)
    test_split = t_cfg.get("test_split", 0.10)
    
    n_total = len(full_dataset)
    n_val = int(val_split * n_total)
    n_test = int(test_split * n_total)
    n_train = n_total - n_val - n_test
    
    generator = torch.Generator().manual_seed(t_cfg.get("seed", 42))
    train_sub, val_sub, test_sub = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    
    # Wrap subsets to apply specific transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            import cv2, numpy as np
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None: img = np.zeros((img_size, img_size), dtype=np.uint8)
            img_tensor = torch.from_numpy(img[:, :, np.newaxis]).permute(2, 0, 1)
            img_tensor = self.transform(img_tensor)
            return img_tensor, label
        def __len__(self):
            return len(self.subset)
            
    train_ds = TransformedSubset(train_sub, get_train_transforms(img_size))
    val_ds = TransformedSubset(val_sub, get_val_transforms(img_size))
    
    nw = t_cfg.get("num_workers", 4)
    # On Windows, num_workers > 0 can sometimes cause broken pipe, fallback to 0 if needed
    nw = 0 if sys.platform == "win32" else nw 
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    
    print("  Data: {} train, {} val, {} test".format(n_train, n_val, n_test))
    
    # 2. Model, Loss, Optimizer
    model = build_model(pretrained=t_cfg.get("pretrained", True), num_classes=2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=t_cfg.get("learning_rate", 1e-4),
                           weight_decay=t_cfg.get("weight_decay", 1e-5))
                           
    epochs = t_cfg.get("epochs", 50)
    scheduler_type = t_cfg.get("scheduler", "cosine")
    
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None
        
    patience = t_cfg.get("early_stop_patience", 8)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
        if scheduler:
            scheduler.step()
            
        epoch_train_loss = train_loss / len(train_ds)
        epoch_train_acc = train_corrects.double() / len(train_ds)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_ds)
        epoch_val_acc = val_corrects.double() / len(val_ds)
        
        elapsed = time.time() - t0
        print("  Epoch {:2d}/{} | Train Loss: {:.4f} Acc: {:.4f} | Val Loss: {:.4f} Acc: {:.4f} | {:.1f}s".format(
            epoch+1, epochs, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc, elapsed
        ))
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  [Early Stopping] No improvement for {} epochs.".format(patience))
                break
                
    # 4. Save best model
    model.load_state_dict(best_model_wts)
    out_path = models_dir / "{}.pth".format(feature_group)
    torch.save(model.state_dict(), str(out_path))
    print("  Saved best model to: {}".format(out_path))
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, default="all",
                        help="Train specific feature model (spline, top_nib, bottom_nib, hole) or 'all'")
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
        train_model_for_feature(feat, cfg)
        
    print("\n[PHASE 5 COMPLETE] All models trained.")

if __name__ == "__main__":
    main()
