import os
import json
from inference import CampusAI

def main():
    print("Initializing Campus AI Pipeline (YOLO + CLIP Fallback + BLIP)...")
    pipeline = CampusAI(yolo_model_path="models/best.pt")
    
    test_dir = r"C:\campus_issue\combined_dataset_7classes\test\images"
    if not os.path.exists(test_dir):
        print(f"Test directory not found at {test_dir}")
        return
        
    images = os.listdir(test_dir)
    if not images:
        print("No images found in test directory.")
        return
        
    test_image = os.path.join(test_dir, images[0])
    print(f"\nRunning inference on: {test_image}")
    
    result = pipeline.predict(test_image)
    
    # Strip base64 image for clean console output
    if "annotated_image" in result:
        result["annotated_image"] = "<BASE64_STRING_OMITTED_FOR_CONSOLE>"
    
    print("\nFINAL PRODUCTION OUTPUT:")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
