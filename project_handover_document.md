# Technical Architecture and Handover Documentation
**System:** YokeClassifier Machine Vision Visual Inspection
**Classification:** Internal Confidential / Technical Handover

This document details the software architecture, machine learning methodology, and maintenance procedures for the YokeClassifier visual inspection pipeline. This guide is specifically written for the engineering team assuming ownership of the project.

The system is currently 100% operational, fully trained, and achieves a `1.00` F1-score across all defect classifications on a multi-variant (Coated & Non-Coated) dataset.

---

## 1. Architectural Philosophy

YokeClassifier departs from monolithic object detection architectures (e.g., YOLO, SSD) in favor of a **Deterministic Multi-Stage Pipeline**.

Instead of tasking a single neural network with both localizing and classifying defects across a high-resolution, complex industrial background, the pipeline enforces strict spatial determinism:
1. **Mathematical Alignment:** The entire image is mathematically transformed (rotated/translated) to perfectly match a fixed coordinate system.
2. **Crop Extraction:** Tiny, isolated sub-regions of interest (ROIs) are extracted using absolute hardcoded coordinates.
3. **Micro-Classification:** The sub-crops are fed into localized, lightweight classifiers (MobileNetV2).

**Why this approach?**
- **Inference Latency:** By executing MobileNetV2 on `128x128` crops, inference times are kept within milliseconds natively on an x86 CPU. A discrete GPU is not required.
- **Traceability:** The system does not merely state "Defective Part." It explicitly highlights the exact feature (e.g., `bottom_nib`) that caused the failure logic to trigger.
- **Data Efficiency:** The models learn pure feature topologies, stripped of varying background lighting and surrounding metal gradients.

---

## 2. Pipeline Phase Breakdown

The codebase is sequentially structured into distinct "Phases." The master script `run_full_pipeline.py` orchestrates these phases sequentially.

### 2.1 Phase 2: Core Alignment (`phase2_alignment/`)
*Entry Point: `aligner.py`*

**Mechanism:** 
The pipeline utilizes **Contour-based Alignment** utilizing `cv2.minAreaRect()`. 
1. The image is converted to grayscale, smoothed with a bilateral filter to preserve edges while eliminating noise, and processed via the Canny edge detector.
2. Morphological dilation and closing operations unify the outer boundary into a single, cohesive binary mask.
3. The geometric center and rotational angle of the mask are computed and compared against the `master_reference.jpg`.
4. A similarity affine transformation (Rotation + Translation + Scale) warps the image so that the yoke is perfectly aligned.

**Historical Context (Crucial Context for New Team):**
The initial architecture utilized SIFT/ORB feature matching. This was deprecated because smooth, reflective (Non-Coated) yokes lack distinct, repeatable microscopic surface textures, causing SIFT to fail and timeout. Contour alignment relies purely on the unchangeable physical silhouette of the machined part.

### 2.2 Phase 3: ROI Extraction (`phase3_roi/`)
*Entry Point: `roi_extractor.py`*

Once aligned, the extraction is mathematically trivial. The module reads 7 bounding box vectors from `config.yaml` (`[x, y, w, h]`) and extracts the corresponding pixels.

### 2.3 Phase 4: Synthetic Defect Generation (`phase4_synthetic/`)
*Entry Point: `build_dataset.py`, `defect_generator.py`*

**The Challenge:** Obtaining a high volume of physically defective parts in a well-managed industrial facility is impossible.
**The Innovation:** The system algorithmically synthesizes its own negative training data.
- The `DefectSynthesizer` identifies the precise coordinates of a feature on a "perfect" image.
- It applies `cv2.inpaint` utilizing the Navier-Stokes / Telea algorithms to mathematically erase the feature, extrapolating the surrounding metal gradients into the void.
- The script automatically outputs thousands of synthesized defects across varying geometries, achieving perfect dataset balance.

**Handling Coated vs. Non-Coated Variations:**
The data parser (`_parse_labels`) dynamically assigns the classification truth table by reading the filename string. The presence of keyword strings (`"perfect"`, `"onlydrilldone"`) instructs the system which crops to isolate into the `present` and `absent` directories.

### 2.4 Phase 5: Deep Learning Core (`phase5_training/`)
*Entry Point: `train.py`, `model.py`*

**Topology:** PyTorch MobileNetV2.
The pipeline trains **4 independent, discrete models**:
1. `spline.pth`
2. `top_nib.pth`
3. `bottom_nib.pth`
4. `hole.pth`

*(Note: The four outer drill holes—top-left, top-right, bottom-left, bottom-right—are unified under a single `hole.pth` model, as the geometric classification logic is identical for all corners. This quadruples the available training data for that specific model).*

**Hyperparameters:**
- **Optimizer:** AdamW (`weight_decay=1e-5`)
- **Scheduler:** Cosine Annealing 
- **Loss:** CrossEntropyLoss
- **Regularization:** Random horizontal/vertical flips, minor rotations, and brightness/contrast jittering (`augmentations.py`).

---

## 3. Operational Maintenance & Knowledge Transfer

### 3.1 Retraining the Models
Should the physical geometry of the yoke undergo an engineering revision, the models must be retrained.

1. **Ingest New Data:** Collect new multi-variant `.jpg` images.
2. **Label via Filename:** Place all images flatly into the `C:\inspection_system\perfect\` and `C:\inspection_system\defect\` directories. Ensure filenames contain the correct identifier string (e.g., `Part_onlydrilldone.jpg`). *Do not use nested subfolders.* If using nested folders, utilize the internal `tools/migrate_dataset.py` to flatten and automatically append the directory name to the filename.
3. **Execute Pipeline:**
   ```powershell
   python run_full_pipeline.py
   ```
   This will systematically wipe the existing models, align the new images, synthesize defects, train, and output new `cm_*.png` confusion matrices.

### 3.2 Recalibrating the Coordinate Matrix
If the physical camera is bumped or repositioned:
1. Re-capture a new `master_reference.jpg` under the new camera position.
2. Update the `reference_image` path in `config.yaml`.
3. Run `python phase3_roi/calibrate_rois.py` to visually drag and snap the 7 ROI boxes over the new reference image to redefine the extraction coordinates.

### 3.3 Adding a New Defect Type
To add a new inspection point (e.g., a newly added center hole):
1. Add the ROI name to `FEATURE_NAMES` across the pipeline.
2. Add the default coordinates to `config.yaml`.
3. Update `_CATEGORY_LABELS` in `phase4_synthetic/build_dataset.py` to include the boolean truth value for the new feature across all existing defects.
4. Execute `run_full_pipeline.py` to train the new feature model.
