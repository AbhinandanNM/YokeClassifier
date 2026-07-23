import sys
import time
from pathlib import Path
import cv2
import yaml

ROOT = Path("c:/Users/Abhinandan/OneDrive/Desktop/Minor/inspection_system")
sys.path.insert(0, str(ROOT))
from phase2_alignment.aligner import Aligner
from utils.image_io import iter_images

def main():
    print("Loading config...")
    cfg_path = ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        
    print("Instantiating Aligner (this loads the reference image)...")
    t0 = time.time()
    aligner = Aligner(cfg)
    print(f"Aligner loaded in {time.time() - t0:.2f}s")
    
    perfect_dir = Path(cfg["paths"]["perfect_dir"])
    files = list(iter_images(perfect_dir))
    if not files:
        print("No files found!")
        return
        
    test_img = files[0]
    print(f"Loading {test_img.name}...")
    t1 = time.time()
    bgr = cv2.imread(str(test_img))
    print(f"Loaded in {time.time() - t1:.2f}s, shape: {bgr.shape}")
    
    print("Aligning...")
    t2 = time.time()
    aligned, quality = aligner.align(bgr)
    print(f"Aligned in {time.time() - t2:.2f}s")
    print(quality)

if __name__ == "__main__":
    main()
