import os
from pathlib import Path
from collections import defaultdict

FINAL_NAMES = [
    "garbage",
    "overflowing_dustbin",
    "overgrown_plants",
    "water_leakage",
    "water_stagnation",
    "dirty_bathroom",
    "crowd"
]

OUT_DIR = Path(r"c:\campus_issue\combined_dataset_7classes")

def verify_dataset():
    splits = ['train', 'valid', 'test']
    
    total_images = {'train': 0, 'valid': 0, 'test': 0}
    class_counts = defaultdict(int)
    all_filenames = set()
    duplicates = set()
    
    errors = []
    
    for split in splits:
        img_dir = OUT_DIR / split / 'images'
        lbl_dir = OUT_DIR / split / 'labels'
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        images = list(img_dir.glob('*.*'))
        total_images[split] = len(images)
        
        for img_path in images:
            if img_path.name in all_filenames:
                duplicates.add(img_path.name)
            all_filenames.add(img_path.name)
            
            label_path = lbl_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                errors.append(f"Missing label for {img_path}")
                continue
                
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                errors.append(f"Corrupted label file {label_path}: {e}")
                continue
                
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                    
                if len(parts) != 5:
                    errors.append(f"Invalid annotation format in {label_path} line {line_idx+1}: {line.strip()}")
                    continue
                    
                try:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    if cls_id < 0 or cls_id >= len(FINAL_NAMES):
                        errors.append(f"Class ID out of range in {label_path} line {line_idx+1}: {cls_id}")
                    
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        errors.append(f"Bounding box not normalized in {label_path} line {line_idx+1}: {line.strip()}")
                        
                    class_counts[cls_id] += 1
                except ValueError:
                    errors.append(f"Non-numeric values in {label_path} line {line_idx+1}: {line.strip()}")

    print("=== Verification Results (7-class) ===")
    print(f"Images in train: {total_images['train']}")
    print(f"Images in valid: {total_images['valid']}")
    print(f"Images in test:  {total_images['test']}")
    
    print("\nAnnotation counts per class:")
    for i in range(len(FINAL_NAMES)):
        print(f"Class {i:2d} ({FINAL_NAMES[i]}): {class_counts[i]}")
        
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate filenames (across all splits):")
        for d in list(duplicates)[:10]:
            print(f" - {d}")
        if len(duplicates) > 10:
            print(" ...")
    else:
        print("\nNo duplicate filenames found.")
        
    if errors:
        print(f"\nFound {len(errors)} errors:")
        for e in errors[:20]:
            print(f" - {e}")
        if len(errors) > 20:
            print(" ...")
    else:
        print("\nNo errors found! Dataset is valid.")

if __name__ == '__main__':
    verify_dataset()
