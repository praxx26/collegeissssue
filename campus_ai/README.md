# Campus AI Inference Module

This module provides a production-ready AI inference pipeline for detecting campus issues and generating natural-language descriptions of images. 

It combines three intelligent layers:
1. **YOLO11**: Primary Object Detector for precise bounding-box detection of campus issues.
2. **Hugging Face CLIP Fallback**: Zero-shot classifier (`openai/clip-vit-base-patch32`) serving as a robust fallback/safety net if YOLO misses an issue.
3. **Salesforce BLIP**: Image Captioner for generating contextual, natural-language descriptions based on the final detected issues and the visual imagery.

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Ensure that the YOLO model weights are located at `models/best.pt`.

## Option 1: FastAPI Production Server

We have provided a ready-to-deploy FastAPI server for your backend team to use immediately. It handles file uploads and returns the exact JSON you need.

Start the server from the `campus_ai` directory:
```bash
python api.py
```
*(Or use `uvicorn api:app --host 0.0.0.0 --port 8000`)*

**Test the API (cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
```

## Option 2: Python Import

Models are optimized to load into memory only once upon instantiation of the `CampusAI` class. Subsequent calls to `.predict()` will reuse these models, providing fast inference.

```python
from campus_ai.inference import CampusAI

# Initialize the pipeline (models are loaded here)
# You can customize thresholds: conf_threshold=0.5, crowd_threshold=5, fallback_threshold=0.25
ai_pipeline = CampusAI()

# Run inference on an image path or a PIL Image object
result = ai_pipeline.predict("path/to/campus_image.jpg")

print(result)
```

## Output Structure

The output is a lightweight, simplified Python dictionary tailored exactly for the production team.

### Issue Detected Example
```json
{
    "predicted_classes": [
        "garbage",
        "water_stagnation"
    ],
    "caption": "An image showing garbage and water stagnation."
}
```

### Normal Example
```json
{
    "predicted_classes": [
        "no issues detected"
    ],
    "caption": "The campus area appears clean and free of the specified issues."
}
```
