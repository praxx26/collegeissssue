import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import sys
import os

# Ensure the script can import local modules
sys.path.append(os.path.dirname(__file__))
from inference import CampusAI

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Campus AI API",
    description="Production API for Campus Issue Detection (YOLO + CLIP Fallback + BLIP Captioning)",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests (React, Angular, Mobile, Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the AI pipeline globally when the server starts
print("Loading Campus AI Pipeline. This may take a moment...")
# Initialize the global pipeline instance
try:
    pipeline = CampusAI(
        yolo_model_path=os.path.join(os.path.dirname(__file__), "models/best.pt"),
        conf_threshold=0.5,
        fallback_threshold=0.40,  # Increased to prevent noisy false positives on chaotic images
        crowd_threshold=5
    )
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

@app.get("/health")
async def health_check():
    """Check if the API and models are loaded and ready."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models failed to load.")
    return {"status": "healthy"}

@app.post("/predict")
def predict_issue(file: UploadFile = File(...)):
    """
    Upload a campus image to detect issues.
    Returns the predicted classes and a natural language caption.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models failed to load.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image bytes synchronously
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
    # Run inference synchronously in FastAPI threadpool
    try:
        result = pipeline.predict(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("api:app", host=host, port=port, reload=False)

