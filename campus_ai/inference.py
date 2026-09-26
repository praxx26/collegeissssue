import os
import io
import base64
import torch
from PIL import Image
from ultralytics import YOLO
try:
    from .caption_generator import CaptionGenerator
    from .hf_fallback import FallbackClassifier
except ImportError:
    from caption_generator import CaptionGenerator
    from hf_fallback import FallbackClassifier

class CampusAI:
    def __init__(self, yolo_model_path="models/best.pt", conf_threshold=0.5, fallback_threshold=0.40, crowd_threshold=5):
        self.conf_threshold = conf_threshold
        self.crowd_threshold = crowd_threshold
        self.fallback_threshold = fallback_threshold
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load YOLO model
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found at {yolo_model_path}")
            
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to(self.device)
        
        self.class_names = [
            "garbage", "overflowing_dustbin", "overgrown_plants", 
            "water_leakage", "water_stagnation", "dirty_bathroom", "crowd"
        ]
        
        # 2. Load Hugging Face Captioning Model
        self.captioner = CaptionGenerator()
        
        # 3. Load Hugging Face Fallback Model (CLIP)
        self.fallback = FallbackClassifier()
        
    def _encode_image(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
        
    def predict(self, image_source):
        """
        Runs the full inference pipeline (YOLO -> Fallback -> Caption).
        image_source can be a path string or a PIL Image.
        Returns a structured dictionary.
        """
        try:
            if isinstance(image_source, str):
                image = Image.open(image_source).convert("RGB")
            elif isinstance(image_source, Image.Image):
                image = image_source.convert("RGB")
            else:
                raise ValueError("Unsupported image type.")
        except Exception as e:
            return {"error": f"Invalid image: {str(e)}"}
            
        # ============================================================
        # 1. YOLO INFERENCE
        # ============================================================
        with torch.no_grad():
            results = self.yolo(image, conf=self.conf_threshold, verbose=False)[0]
            
        yolo_detections = []
        crowd_count = 0
        yolo_issue_classes = set()
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"unknown_{cls_id}"
            
            if class_name == "crowd":
                crowd_count += 1
                
            yolo_detections.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "source": "yolo",
                "bbox": [x1, y1, x2, y2]
            })
            
            if class_name != "crowd":
                yolo_issue_classes.add(class_name)
                
        # Crowd Logic
        crowd_status = "detected" if crowd_count >= self.crowd_threshold else "not_detected"
        
        # ============================================================
        # 2. HUGGING FACE FALLBACK & COMBINATION LOGIC
        # ============================================================
        final_classes = list(yolo_issue_classes)
        
        # Always run fallback to catch ANY issues YOLO might have missed
        # Since fallback now returns a list of all predictions above threshold
        fallback_preds = self.fallback.classify(image, threshold=self.fallback_threshold)
        
        for pred in fallback_preds:
            if pred["used"] and pred["class"] not in final_classes and pred["class"] != "crowd":
                final_classes.append(pred["class"])
            
        # Add crowd to final classes if threshold was met
        if crowd_status == "detected" and "crowd" not in final_classes:
            final_classes.append("crowd")
            
        # Status determination
        status = "issue_detected" if len(final_classes) > 0 else "normal"
        
        # ============================================================
        # 3. CAPTION GENERATION
        # ============================================================
        context = final_classes if status == "issue_detected" else []
        caption = self.captioner.generate_caption(image, context_classes=context)
        
        # ============================================================
        # 4. JSON OUTPUT
        # ============================================================
        if len(final_classes) == 0:
            predicted = ["no issues detected"]
        else:
            predicted = final_classes

        result = {
            "predicted_classes": predicted,
            "caption": caption
        }
        
        return result
