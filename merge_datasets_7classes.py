import os
import shutil
import random
from pathlib import Path
import yaml

# NEW 7-CLASS MAPPING
# 0: garbage
# 1: overflowing_dustbin
# 2: overgrown_plants
# 3: water_leakage
# 4: water_stagnation
# 5: dirty_bathroom
# 6: crowd

CLASS_MAPPING = {
    'GARBAGE CLASSIFICATION': {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0
    },
    'dirty_bathroom': {
        11: 0,
        6: 5, 7: 5, 8: 5, 9: 5, 10: 5,
        12: 5, 13: 5, 14: 5, 15: 5, 16: 5
        # 0-5 are ignored
    },
    'overflowing_dustbin': {
        1: 1
        # 0 is ignored
    },
    'overgrown_plants': {
        0: 2, 1: 2
        # 2, 3, 4 are ignored
    },
    'water_leakage': {
        0: 3
    },
    'water_stagnation': {
        0: 4, 1: 4
    },
    'crowd': {
        0: 6
    }
}

FINAL_NAMES = [
    "garbage",
    "overflowing_dustbin",
    "overgrown_plants",
    "water_leakage",
    "water_stagnation",
    "dirty_bathroom",
    "crowd"
]

BASE_DIR = Path(r"c:\campus_issue\dataset")
OUT_DIR = Path(r"c:\campus_issue\combined_dataset_7classes")

def setup_directories():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    
    for split in ['train', 'valid', 'test']:
        (OUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

def create_data_yaml():
    yaml_content = {
        'train': './train/images',
        'val': './valid/images',
        'test': './test/images',
        'nc': len(FINAL_NAMES),
        'names': FINAL_NAMES
    }
    with open(OUT_DIR / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

def process_file_pair(img_path, label_path, dataset_name, out_split, counter):
    new_base_name = f"{dataset_name.replace(' ', '_').lower()}_{counter:06d}"
    
    out_img_path = OUT_DIR / out_split / 'images' / (new_base_name + img_path.suffix)
    out_label_path = OUT_DIR / out_split / 'labels' / (new_base_name + '.txt')

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return False

    valid_lines = []
    mapping = CLASS_MAPPING.get(dataset_name, {})

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                class_id = int(parts[0])
                if class_id in mapping:
                    new_id = mapping[class_id]
                    valid_lines.append(f"{new_id} {' '.join(parts[1:5])}\n")
            except ValueError:
                pass
    
    if not valid_lines:
        return False # No valid annotations found (or all were ignored)
    
    # Write new labels
    with open(out_label_path, 'w') as f:
        f.writelines(valid_lines)
        
    # Copy image
    shutil.copy2(img_path, out_img_path)
    return True

def process_standard_dataset(dataset_name):
    print(f"Processing standard dataset: {dataset_name}")
    dataset_dir = BASE_DIR / dataset_name
    counter = 1
    for split in ['train', 'valid', 'test', 'val']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        out_split = 'valid' if split == 'val' else split
        
        img_dir = split_dir / 'images'
        lbl_dir = split_dir / 'labels'
        
        if not img_dir.exists() or not lbl_dir.exists():
            # If structure is flat inside split
            imgs = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.jpeg')) + list(split_dir.glob('*.png'))
            for img_path in imgs:
                label_path = split_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    if process_file_pair(img_path, label_path, dataset_name, out_split, counter):
                        counter += 1
        else:
            imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
            for img_path in imgs:
                label_path = lbl_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    if process_file_pair(img_path, label_path, dataset_name, out_split, counter):
                        counter += 1
    print(f"  -> Added {counter-1} images from {dataset_name}")

def process_water_stagnation():
    print("Processing water_stagnation (custom layout, random seed 42, 80/10/10)")
    dataset_name = 'water_stagnation'
    dataset_dir = BASE_DIR / dataset_name
    
    all_pairs = []
    
    for root, dirs, files in os.walk(dataset_dir):
        if "Raw data" in root or "Raw Data" in root:
            continue
            
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                label_path = root_path / (file_path.stem + '.txt')
                if label_path.exists():
                    all_pairs.append((file_path, label_path))
                    
    random.seed(42)
    random.shuffle(all_pairs)
    
    n = len(all_pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    counter = 1
    for idx, (img_path, label_path) in enumerate(all_pairs):
        if idx < train_end:
            split = 'train'
        elif idx < val_end:
            split = 'valid'
        else:
            split = 'test'
            
        if process_file_pair(img_path, label_path, dataset_name, split, counter):
            counter += 1
            
    print(f"  -> Added {counter-1} images from {dataset_name}")

def main():
    print("Setting up 7-class combined dataset directories...")
    setup_directories()
    create_data_yaml()
    
    for dataset_name in CLASS_MAPPING.keys():
        if dataset_name == 'water_stagnation':
            process_water_stagnation()
        else:
            process_standard_dataset(dataset_name)
            
    print("Done merging datasets into 7-class format.")

if __name__ == '__main__':
    main()
