import os

dst_base = r"c:\campus_issue\combined_dataset_7classes"
datasets_to_check = ["extratrashforgarbage", "extrawaterstagnation"]

for ds_name in datasets_to_check:
    print(f"--- Checking dataset: {ds_name} ---")
    total_images = 0
    class_counts = {}
    
    for split in ["train", "valid", "test"]:
        dst_images = os.path.join(dst_base, split, "images")
        dst_labels = os.path.join(dst_base, split, "labels")
        
        if not os.path.exists(dst_images) or not os.path.exists(dst_labels):
            continue
            
        images_found = [f for f in os.listdir(dst_images) if f.startswith(ds_name + "_")]
        total_images += len(images_found)
        
        # Count classes across all added labels
        for img in images_found:
            base, ext = os.path.splitext(img)
            lbl_file = os.path.join(dst_labels, base + ".txt")
            if os.path.exists(lbl_file):
                with open(lbl_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        cls_id = line.split()[0]
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
    
    print(f"Total images added: {total_images}")
    print(f"Total bounding boxes found in label files: {sum(class_counts.values())}")
    print(f"Mapped classes (class_id: count): {class_counts}")
    print()
