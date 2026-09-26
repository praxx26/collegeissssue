import os
import shutil
from pathlib import Path

# New dataset classes: ['garbage', 'garbage_bin', 'garbage_overflow', 'overflow']
# Final 7 classes: 
# 0: garbage
# 1: overflowing_dustbin
# ...

EXTRA_MAPPING = {
    0: 0, # garbage -> garbage (0)
    # 1 is skipped (garbage_bin is not overflowing)
    2: 1, # garbage_overflow -> overflowing_dustbin (1)
    3: 1  # overflow -> overflowing_dustbin (1)
}

EXTRA_DIR = Path(r"c:\campus_issue\to_add_datasforlessdata\extraOverflowBinimages")
OUT_DIR = Path(r"c:\campus_issue\combined_dataset_7classes")

def add_extra_dataset():
    counter = 1
    added_images = 0
    
    for split in ['train', 'valid']:
        img_dir = EXTRA_DIR / split / 'images'
        lbl_dir = EXTRA_DIR / split / 'labels'
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        imgs = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
        
        for img_path in imgs:
            label_path = lbl_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                continue
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        if class_id in EXTRA_MAPPING:
                            new_id = EXTRA_MAPPING[class_id]
                            valid_lines.append(f"{new_id} {' '.join(parts[1:5])}\n")
                    except ValueError:
                        pass
                        
            if valid_lines:
                new_base_name = f"extra_overflow_{split}_{counter:06d}"
                out_img_path = OUT_DIR / split / 'images' / (new_base_name + img_path.suffix)
                out_label_path = OUT_DIR / split / 'labels' / (new_base_name + '.txt')
                
                with open(out_label_path, 'w') as f:
                    f.writelines(valid_lines)
                
                shutil.copy2(img_path, out_img_path)
                added_images += 1
                counter += 1
                
    print(f"Added {added_images} images from extra dataset.")

if __name__ == '__main__':
    add_extra_dataset()
