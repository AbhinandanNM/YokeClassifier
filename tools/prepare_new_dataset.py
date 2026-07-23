# -*- coding: utf-8 -*-
"""
tools/prepare_new_dataset.py
=============================
Prepares the new high-quality dataset for training.

What this script does
---------------------
1. Clears old images from Minor/perfect/ and Minor/defect/.
2. Copies the new high-quality JPG images into the correct directories:
     - Coated-perfect.jpg     → perfect/
     - Non-coated-perfect.jpg → perfect/
     - All other categories   → defect/   (allmissing, onlydrilldone, onlynibblingmissing)
3. Copies the new Coated-perfect.jpg as the reference image for alignment.

Run ONCE after you have converted HEIC → JPG (tools/convert_heic.py).

Usage
-----
    python tools/prepare_new_dataset.py --source C:/minorprojectimg/Sample

    # Dry-run (see what it would do without making changes):
    python tools/prepare_new_dataset.py --source C:/minorprojectimg/Sample --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Category → label mapping
# Perfect = "perfect" in category name (case-insensitive)
# Everything else is a defect
# ---------------------------------------------------------------------------
DEFECT_KEYWORDS = ["allmissing", "onlydrilldone", "onlynibblingmissing",
                   "onlydrilldone", "onlynibblingdone"]


def is_perfect(stem: str) -> bool:
    """Return True if the filename stem represents a PERFECT part."""
    lower = stem.lower()
    return lower.endswith("-perfect") or lower.endswith("_perfect")


def classify_files(source_dir: Path):
    """Return two lists: (perfect_files, defect_files) of Path objects."""
    perfect_files = []
    defect_files = []
    for fp in sorted(source_dir.iterdir()):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if is_perfect(fp.stem):
            perfect_files.append(fp)
        else:
            defect_files.append(fp)
    return perfect_files, defect_files


def clear_directory(directory: Path, dry_run: bool) -> int:
    """Delete all image files in *directory*. Returns count removed."""
    count = 0
    for fp in directory.iterdir():
        if fp.is_file() and fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if not dry_run:
                fp.unlink()
            count += 1
    return count


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    """Copy src → dst, printing the action."""
    print("  COPY  {}  ->  {}".format(src.name, dst))
    if not dry_run:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare new high-quality dataset for YokeClassifier"
    )
    parser.add_argument(
        "--source", required=True,
        help="Directory containing the new dataset JPG files (e.g. C:/minorprojectimg/Sample)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without making any changes"
    )
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    if not source_dir.exists():
        print("[ERROR] Source directory not found: {}".format(source_dir))
        sys.exit(1)

    # Locate project root (tools/ is inside inspection_system/)
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root.parent  # Minor/

    perfect_dir = dataset_root / "perfect"
    defect_dir  = dataset_root / "defect"
    config_path = project_root / "config" / "config.yaml"

    # Sanity checks
    for d in [perfect_dir, defect_dir]:
        if not d.exists():
            print("[ERROR] Directory not found: {}".format(d))
            print("        Check config.yaml paths match your folder layout.")
            sys.exit(1)

    print("=" * 65)
    print("  DATASET PREPARATION  {}".format("(DRY RUN)" if args.dry_run else ""))
    print("=" * 65)
    print("  Source       : {}".format(source_dir))
    print("  perfect/ dir : {}".format(perfect_dir))
    print("  defect/  dir : {}".format(defect_dir))
    print("-" * 65)

    # --- Step 1: Classify source files ---
    perfect_files, defect_files = classify_files(source_dir)
    print("\n[Step 1] Classified source files:")
    print("  Perfect  ({:>2}): {}".format(len(perfect_files),
          ", ".join(f.name for f in perfect_files)))
    print("  Defect   ({:>2}): {}".format(len(defect_files),
          ", ".join(f.name for f in defect_files)))

    # --- Step 2: Clear old data ---
    print("\n[Step 2] Clearing old training images...")
    n_p = clear_directory(perfect_dir, args.dry_run)
    n_d = clear_directory(defect_dir,  args.dry_run)
    print("  Removed {} images from perfect/".format(n_p))
    print("  Removed {} images from defect/".format(n_d))

    # --- Step 3: Copy new images ---
    print("\n[Step 3] Copying new images...")
    for fp in perfect_files:
        copy_file(fp, perfect_dir / fp.name, args.dry_run)
    for fp in defect_files:
        copy_file(fp, defect_dir / fp.name, args.dry_run)

    # --- Step 4: Update reference image in config.yaml ---
    # Use the first perfect file found (prefer Coated-perfect if it exists)
    ref_candidates = [f for f in perfect_files if "coated" in f.stem.lower()
                      and "non" not in f.stem.lower()]
    if not ref_candidates:
        ref_candidates = perfect_files

    if ref_candidates:
        new_ref = perfect_dir / ref_candidates[0].name
        print("\n[Step 4] Updating reference image in config.yaml...")
        print("  New reference: {}".format(new_ref))
        if not args.dry_run:
            _update_reference_image(config_path, new_ref)
        else:
            print("  (dry-run: config.yaml NOT modified)")

    print("\n" + "=" * 65)
    print("  DONE{}".format(" (dry-run — no changes made)" if args.dry_run else ""))
    print("  Next steps:")
    print("  1. python phase2_alignment/test_alignment.py")
    print("  2. python phase4_synthetic/build_dataset.py")
    print("  3. python phase5_training/train.py")
    print("=" * 65)


def _update_reference_image(config_path: Path, new_ref: Path) -> None:
    """Update the reference_image path in config.yaml (simple text substitution)."""
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    # Match the reference_image line regardless of what path is currently there
    pattern = r"(reference_image:\s*)(.+)"
    replacement = r"\g<1>{}".format(new_ref.as_posix())
    new_content, n = re.subn(pattern, replacement, content)

    if n == 0:
        print("  [WARN] Could not find 'reference_image:' key in config.yaml. "
              "Please update it manually.")
        return

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("  config.yaml updated.")


if __name__ == "__main__":
    main()
