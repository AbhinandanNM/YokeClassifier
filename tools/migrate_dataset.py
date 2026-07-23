import os
import shutil
from pathlib import Path

# Define paths
ROOT = Path(r"C:\Users\Abhinandan\OneDrive\Desktop\Minor")
FINAL_DIR = ROOT / "Final"
PERFECT_DIR = ROOT / "perfect"
DEFECT_DIR = ROOT / "defect"

def migrate_datasets():
    print("Starting Dataset Migration...")
    
    if not FINAL_DIR.exists():
        print(f"[ERROR] Could not find {FINAL_DIR}")
        return

    # Ensure target directories exist
    PERFECT_DIR.mkdir(exist_ok=True)
    DEFECT_DIR.mkdir(exist_ok=True)
    
    total_copied = 0
    
    # Iterate through all subfolders in Final/
    for subfolder in FINAL_DIR.iterdir():
        if not subfolder.is_dir():
            continue
            
        folder_name = subfolder.name
        
        # Determine destination based on whether it is 'perfect'
        if "perfect" in folder_name.lower():
            target_dir = PERFECT_DIR
        else:
            target_dir = DEFECT_DIR
            
        # Iterate through images in this subfolder
        count = 0
        for img_path in subfolder.glob("*.jpg"):
            # Create a new filename that INCLUDES the folder name so the parser can read it!
            # Example: Coated-onlydrilldone_20260710_000224.jpg
            new_filename = f"{folder_name}_{img_path.name}"
            new_filepath = target_dir / new_filename
            
            # Copy the file
            shutil.copy2(img_path, new_filepath)
            count += 1
            total_copied += 1
            
        print(f"Copied {count} images from {folder_name} -> {target_dir.name}/")

    print(f"\n[SUCCESS] Successfully migrated and safely renamed {total_copied} images!")
    print("You can now run: python run_full_pipeline.py")

if __name__ == "__main__":
    migrate_datasets()
