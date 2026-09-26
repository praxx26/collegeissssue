import sys
from ultralytics import YOLO

FINAL_NAMES = [
    "Garbage",
    "Overflowing Dustbin",
    "Overgrown Plants",
    "Water Leakage",
    "Water Stagnation",
    "Dirty Bathroom",
    "Crowd"
]

def format_class_name(cls_id):
    if 0 <= cls_id < len(FINAL_NAMES):
        return FINAL_NAMES[cls_id]
    return f"Unknown Class ({cls_id})"

def test_image(model_path, image_path):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Running inference on {image_path}...")
    # conf=0.25 to catch all reasonable detections. iou=0.45 for NMS
    results = model(image_path, conf=0.25, iou=0.45)[0]
    
    print("\nDetected Issues:")
    if len(results.boxes) == 0:
        print("- No issues detected.")
    
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        
        # xyxy format (xmin, ymin, xmax, ymax)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        class_name = format_class_name(cls_id)
        print(f"- {class_name} — {int(conf * 100)}% (Box: [{x1}, {y1}, {x2}, {y2}])")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_inference_7classes.py <path_to_model.pt> <path_to_image.jpg>")
        sys.exit(1)
        
    test_image(sys.argv[1], sys.argv[2])
