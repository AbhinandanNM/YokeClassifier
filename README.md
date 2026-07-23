# YokeClassifier: End-to-End Industrial Visual Inspection System

Welcome to the **YokeClassifier** project! This repository contains a production-ready, end-to-end industrial visual inspection pipeline. It is designed to automatically detect the presence or absence of critical micro-features (splines, top nibs, bottom nibs, and holes) on automotive metal components (yokes). 

By combining traditional Computer Vision algorithms for precise image registration with lightweight Deep Learning models for feature classification, this system achieves incredibly high accuracy while remaining fast enough to run in real-time on factory edge devices.

This README serves as the ultimate master guide for the project. If you clone this repository, this document provides everything you need to understand the architecture, configure the system, and deploy it.

---

## 🛠️ 1. Installation & Setup

### Prerequisites
- Python 3.10+
- An IP Camera or local webcam for live inspection (Phase 7)
- Raw image datasets of "perfect" parts and "defective" parts

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbhinandanNM/YokeClassifier.git
   cd YokeClassifier
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Core libraries include: OpenCV (`cv2`), PyTorch, TorchVision, NumPy, PyYAML, Scikit-Learn, Matplotlib, Seaborn)*

3. **Directory Structure Setup:**
   Ensure you have your raw dataset placed outside the main code directory (as configured in `config.yaml`). The expected structure is:
   ```text
   Minor/
   ├── perfect/               # Raw images of perfect yokes
   ├── defect/                # Raw images of defective yokes
   └── inspection_system/     # THIS REPOSITORY
       ├── config/
       ├── phase1_analysis/
       ├── phase2_alignment/
       └── ...
   ```

---

## 🧠 2. Deep Dive: System Architecture (The 7 Phases)

The pipeline is split into 7 modular phases. You can run the entire training pipeline end-to-end using the master script (`python run_full_pipeline.py`), or run them individually.

### Phase 1: Data Analysis (`phase1_analysis/`)
**Goal:** Understand the environmental lighting and pixel distribution of the dataset.
- **How it works:** Factory lighting can be highly variable. This script parses the raw `perfect` and `defect` image directories and computes global statistics (mean intensity, standard deviation) and generates color histograms.
- **Why it matters:** By understanding the baseline pixel contrast, we can configure downstream preprocessing techniques like CLAHE (Contrast Limited Adaptive Histogram Equalization) to ensure shadows or glares don't break the system.

### Phase 2: Robust Batch Alignment (`phase2_alignment/`)
**Goal:** Mathematically lock every incoming image into a fixed, predictable coordinate grid.
- **The Problem:** In a real factory, parts vibrate on conveyor belts and are never perfectly straight. If a part is rotated, our Region of Interest (ROI) crops will capture the wrong area of the metal.
- **Primary Method (SIFT + Homography):** The system first uses SIFT (Scale-Invariant Feature Transform) to find thousands of unique micro-textures on the incoming image. It compares these points to a "Golden Reference" template. Using Lowe's Ratio Test and RANSAC, it filters outliers and calculates a Homography matrix to warp the image perfectly flat.
- **Fallback Method (Contour + MinAreaRect):** If the part is completely defective (a smooth, featureless surface), SIFT will fail to find textures. The system automatically falls back to binary contour alignment. It thresholds the image to isolate the metal silhouette, uses `cv2.minAreaRect()` to compute the exact physical angle of rotation of the part, compares it to the reference angle, and applies an Affine Transform matrix to translate, scale, and rotate the part into perfect alignment.

### Phase 3: ROI Calibration & Visualization (`phase3_roi/`)
**Goal:** Define the exact pixel boundaries for the micro-features we want to inspect.
- **How it works:** Once all images are rigidly aligned to a 1080x1080 canvas, we run `calibrate_rois.py`. This opens an interactive OpenCV window. The user physically clicks and drags bounding boxes around the 7 features: `spline`, `top_nib`, `bottom_nib`, and the 4 `holes`.
- **Storage:** These fixed coordinates are saved permanently into `config/config.yaml`.
- **Extraction:** The `ROIExtractor` uses these static coordinates to slice the large 1080x1080 image into tiny, highly focused 128x128 grayscale "crops".

### Phase 4: Synthetic Defect Generation (`phase4_synthetic/`)
**Goal:** Create a massive, perfectly balanced dataset for Deep Learning, even if you lack real defective parts.
- **The Problem:** Factories produce very few defective parts, leading to heavily imbalanced datasets that cause neural networks to overfit.
- **How it works:** We take the perfectly aligned "good" images and digitally vandalize them. Using OpenCV's `cv2.inpaint` (Telea algorithm), the system isolates the ROI masks and mathematically erases the features (filling in holes, smoothing out splines). 
- **Result:** It automatically generates hundreds of synthetic "Absent" variants for every "Present" image, outputting a perfectly balanced dataset of 128x128 crops ready for PyTorch.

### Phase 5: Deep Learning Training (`phase5_training/`)
**Goal:** Train Convolutional Neural Networks to classify the 128x128 crops.
- **Architecture:** The system utilizes **MobileNetV2**. We use this specific architecture because it relies on Depthwise Separable Convolutions, making it incredibly lightweight and fast—perfect for real-time inference on edge devices (like a Raspberry Pi or industrial IPC).
- **Training Strategy:** Instead of training one massive network, we train **4 independent binary classifiers** (one for splines, one for holes, etc.). This makes the models hyper-specialized and highly accurate.
- **Hyperparameters:** The models are trained using the Adam Optimizer, Cross-Entropy Loss, and a Cosine Annealing Learning Rate scheduler. Early stopping prevents overfitting.

### Phase 6: Model Evaluation (`phase6_evaluation/`)
**Goal:** Prove the models work on unseen data.
- **How it works:** The script tests the trained `.pth` weights against a held-out test split of crops. It utilizes `scikit-learn` to compute Precision, Recall, and F1-Scores. 
- **Visualization:** It generates Seaborn Heatmap Confusion Matrices (saved in `outputs/`) so you can visually verify that there are no False Positives or False Negatives.

### Phase 7: Live Inspection Deployment (`phase7_live/`)
**Goal:** Real-time factory floor deployment.
- **How it works:** `live_inspection.py` is the grand finale. It connects to the live camera feed (configurable URL/Index). 
- **The Pipeline Loop:** For every single frame captured at 30 FPS:
  1. The frame is dynamically **aligned** (fixing rotation/vibration).
  2. The 7 ROIs are **sliced** out of the frame.
  3. The crops are pushed through the 4 **MobileNetV2 models**.
  4. The network predicts `Present` or `Absent` for each feature.
  5. The GUI overlays bounding boxes on the live feed. Green = PASS, Red = FAIL.
  6. A massive PASS/FAIL banner is flashed on screen to alert the operator.

---

## ⚙️ 3. Master Configuration (`config.yaml`)

The entire system is controlled by `config/config.yaml`. If you want to change how the system behaves, do it here. You rarely need to edit the Python code.

Key sections to modify:
- `paths`: Change where the system looks for datasets and saves outputs.
- `alignment`: Tweak the output resolution (e.g., `1080x1080`), SIFT feature count, and matching thresholds.
- `synthetic`: Change `augment_factor` to generate more or fewer synthetic defects.
- `training`: Adjust `epochs`, `batch_size`, `learning_rate`, and `num_workers`.
- `camera`: Set your IP Camera URL or local webcam index (e.g., `0`).

---

## 🚀 4. Quickstart / Command Reference

### Option A: Fully Automated Training
If you just want to align the images, generate the synthetic dataset, train the models, and evaluate them in one single shot:
```bash
python run_full_pipeline.py
```

### Option B: Running Step-by-Step
If you want to manually execute the pipeline to inspect the outputs at each stage:
```bash
# 1. Align the raw dataset
python phase2_alignment/test_alignment.py --mode batch

# 2. Visually verify the alignment
python phase3_roi/visualize_rois.py

# 3. Build the synthetic PyTorch dataset
python phase4_synthetic/build_dataset.py

# 4. Train the Neural Networks
python phase5_training/train.py

# 5. Evaluate the Neural Networks
python phase6_evaluation/evaluate.py
```

### Option C: Launch Live Inspection
Once models are trained and saved in `models/`, run the real-time GUI:
```bash
python phase7_live/live_inspection.py
```
