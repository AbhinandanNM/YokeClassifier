# -*- coding: utf-8 -*-
"""
run_full_pipeline.py
=====================
Runs the ENTIRE inspection pipeline end-to-end:
  Phase 2 → Batch Alignment (with improved contour fallback)
  Phase 3 → ROI Visualization (verify)
  Phase 4 → Dataset Generation (synthetic defects + real crops)
  Phase 5 → Model Training (MobileNetV2 classifiers)
  Phase 6 → Evaluation (accuracy, confusion matrices)

Usage:
    python run_full_pipeline.py
"""

import subprocess
import sys
import time
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_phase(phase_name, command, timeout=None):
    """Run a phase and print status."""
    print("\n" + "=" * 70)
    print("  >>> {} <<<".format(phase_name))
    print("=" * 70 + "\n")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + command,
        cwd=str(ROOT),
        timeout=timeout,
        capture_output=False,
    )
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print("\n  [!!!] {} FAILED (exit code {}) after {:.1f}s".format(
            phase_name, result.returncode, elapsed))
        print("  Continuing with remaining phases...\n")
        return False
    
    print("\n  [OK] {} completed in {:.1f}s".format(phase_name, elapsed))
    return True


def clean_old_data():
    """Remove old ROI dataset and models to ensure fresh pipeline run."""
    import yaml
    with open(ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    roi_dir = Path(cfg["paths"]["roi_dataset_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    
    if roi_dir.exists():
        print("  Clearing old ROI dataset: {}".format(roi_dir))
        shutil.rmtree(roi_dir)
    
    if models_dir.exists():
        print("  Clearing old models: {}".format(models_dir))
        shutil.rmtree(models_dir)


def main():
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   INDUSTRIAL VISUAL INSPECTION - FULL PIPELINE RUN" + " " * 17 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    t_total = time.time()
    
    # Step 0: Clean old data
    print("\n[STEP 0] Cleaning old datasets and models...")
    clean_old_data()
    
    # Phase 2: Batch Alignment
    ok = run_phase(
        "PHASE 2 - BATCH ALIGNMENT",
        ["phase2_alignment/test_alignment.py", "--mode", "batch"],
        timeout=None,
    )
    
    # Phase 3: ROI Visualization (verify)
    run_phase(
        "PHASE 3 - ROI VISUALIZATION",
        ["phase3_roi/visualize_rois.py"],
        timeout=60,
    )
    
    # Phase 4: Dataset Generation
    ok = run_phase(
        "PHASE 4 - ROI DATASET GENERATION",
        ["phase4_synthetic/build_dataset.py"],
        timeout=None,
    )
    
    if not ok:
        print("\n[FATAL] Phase 4 failed. Cannot continue without dataset.")
        sys.exit(1)
    
    # Phase 5: Training
    ok = run_phase(
        "PHASE 5 - MODEL TRAINING (15 epochs)",
        ["phase5_training/train.py"],
        timeout=None,
    )
    
    if not ok:
        print("\n[FATAL] Phase 5 failed. Cannot evaluate without trained models.")
        sys.exit(1)
    
    # Phase 6: Evaluation
    run_phase(
        "PHASE 6 - MODEL EVALUATION",
        ["phase6_evaluation/evaluate.py"],
        timeout=300,
    )
    
    total_elapsed = time.time() - t_total
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   FULL PIPELINE COMPLETE — Total time: {:.0f}s ({:.1f} min){}#".format(
        total_elapsed, total_elapsed / 60, " " * max(0, 20 - len("{:.1f}".format(total_elapsed / 60)))))
    print("#" + " " * 68 + "#")
    print("#" * 70)


if __name__ == "__main__":
    main()
