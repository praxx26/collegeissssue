import os
import shutil

src_base = r"c:\campus_issue\to_add_datasforlessdata"
dst_base = r"c:\campus_issue\combined_dataset_7classes"

datasets_to_add = [
    {"name": "extratrashforgarbage", "target_class_id": 0},
    {"name": "extrawaterstagnation", "target_class_id": 4}
]

splits = ["train", "valid", "test"]

for ds in datasets_to_add:
    ds_name = ds["name"]
    target_id = ds["target_class_id"]
    
    for split in splits:
        src_images = os.path.join(src_base, ds_name, split, "images")
        src_labels = os.path.join(src_base, ds_name, split, "labels")
        
        if not os.path.exists(src_images) or not os.path.exists(src_labels):
            print(f"Skipping {split} for {ds_name} as it doesn't exist.")
            continue
            
        dst_images = os.path.join(dst_base, split, "images")
        dst_labels = os.path.join(dst_base, split, "labels")
        
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)
        
        for img_name in os.listdir(src_images):
            # Create a unique name to avoid collisions
            base, ext = os.path.splitext(img_name)
            new_base = f"{ds_name}_{base}"
            new_img_name = new_base + ext
            new_lbl_name = new_base + ".txt"
            
            src_img_path = os.path.join(src_images, img_name)
            dst_img_path = os.path.join(dst_images, new_img_name)
            
            # Copy image
            shutil.copy2(src_img_path, dst_img_path)
            
            # Process label
            src_lbl_path = os.path.join(src_labels, base + ".txt")
            dst_lbl_path = os.path.join(dst_labels, new_lbl_name)
            
            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, "r") as f_in, open(dst_lbl_path, "w") as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if parts:
                            parts[0] = str(target_id)
                            f_out.write(" ".join(parts) + "\n")
                            
        print(f"Processed {ds_name} -> {split}")

print("Merge completed!")
