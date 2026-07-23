# -*- coding: utf-8 -*-
"""
tools/convert_heic.py
======================
One-shot offline HEIC → JPG converter.

Run this ONCE on any folder containing *.heic / *.HEIC files before feeding
images into the pipeline.  The pipeline itself (aligner, build_dataset, etc.)
uses cv2.imread, which cannot decode HEIC on Windows natively.

Usage
-----
    python tools/convert_heic.py --input C:/minorprojectimg/Sample --output C:/minorprojectimg/Converted

    # Convert in-place (overwrites directory with JPGs alongside HEICs):
    python tools/convert_heic.py --input C:/minorprojectimg/Sample

Dependencies
------------
    pip install pillow-heif
"""

import argparse
import sys
from pathlib import Path

try:
    import pillow_heif
    from PIL import Image
    pillow_heif.register_heif_opener()   # teaches PIL to open HEIC files
except ImportError:
    print("[ERROR] pillow-heif is not installed.")
    print("        Run:  pip install pillow-heif")
    sys.exit(1)


def convert_folder(input_dir: Path, output_dir: Path, quality: int = 95) -> None:
    """Convert all HEIC files in *input_dir* to JPG in *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    heic_files = sorted(
        [p for p in input_dir.iterdir()
         if p.is_file() and p.suffix.lower() == ".heic"]
    )

    if not heic_files:
        print("[INFO] No HEIC files found in: {}".format(input_dir))
        return

    print("=" * 60)
    print("  HEIC → JPG Converter")
    print("=" * 60)
    print("  Input  : {}".format(input_dir))
    print("  Output : {}".format(output_dir))
    print("  Files  : {}".format(len(heic_files)))
    print("-" * 60)

    ok = 0
    fail = 0
    for heic_path in heic_files:
        out_path = output_dir / (heic_path.stem + ".jpg")
        try:
            img = Image.open(heic_path)
            # Ensure RGB (not RGBA or other modes that JPEG cannot store)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_path, "JPEG", quality=quality)
            print("  [OK]   {} → {}".format(heic_path.name, out_path.name))
            ok += 1
        except Exception as exc:
            print("  [FAIL] {} — {}".format(heic_path.name, exc))
            fail += 1

    print("-" * 60)
    print("  Converted : {}  |  Failed : {}".format(ok, fail))
    if fail:
        print("  [!] Check the failed files above and re-run manually.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HEIC images to JPG for use in the pipeline"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory containing *.heic files"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for JPG files (default: same as input)"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality (1-100, default: 95)"
    )
    args = parser.parse_args()

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve() if args.output else input_dir

    if not input_dir.exists():
        print("[ERROR] Input directory does not exist: {}".format(input_dir))
        sys.exit(1)

    convert_folder(input_dir, output_dir, quality=args.quality)


if __name__ == "__main__":
    main()
