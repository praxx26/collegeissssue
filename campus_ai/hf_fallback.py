import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class FallbackClassifier:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Use descriptive prompts to help CLIP distinguish between classes
        self.class_prompts = {
            "garbage": "a photo of loose garbage and trash scattered on the ground or street",
            "overflowing dustbin": "a close up photo of a physical dustbin or trash can that is overflowing with garbage",
            "overgrown plants": "a photo of overgrown plants, weeds, and unkempt vegetation",
            "water leakage": "a photo showing active water leakage dripping from a broken pipe, faucet, ceiling, or wall",
            "water stagnation": "a photo of a large puddle of dirty, stagnant water pooling heavily on the ground or street",
            "dirty bathroom": "a photo of a dirty and unsanitary bathroom or toilet",
            "crowd": "a photo of a dense crowd of people",
            "normal": "a photo of a clean and normal outdoor campus area with no issues",
            "unrelated": "a photo of an indoor room, gym equipment, social media post, text, news footage, protests, police, water cannons, or completely unrelated scene",
            "people": "a photo of a few people posing indoors or outdoors, unrelated to campus issues",
            "vehicles": "a photo of cars, trucks, motorcycles, or traffic on a road",
            "animals": "a photo of animals, dogs, cats, or pets",
            "blurry": "a blurry, completely dark, or unrecognizable image"
        }
        self.allowed_classes = list(self.class_prompts.keys())
        
    def classify(self, image: Image.Image, threshold: float = 0.25):
        """
        Runs zero-shot classification on the image against the allowed classes.
        Returns the top class and its confidence. If the confidence is below the
        threshold, or the top class is 'normal' or a sink class, it returns 'normal'.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Formulate prompts for CLIP zero-shot classification using the detailed descriptions
        prompts = list(self.class_prompts.values())
        
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
        
        predictions = []
        for idx, score_tensor in enumerate(probs):
            score = score_tensor.item()
            best_class = self.allowed_classes[idx]
            best_class_normalized = best_class.replace(" ", "_")
            
            # Skip sink classes
            if best_class_normalized in ["normal", "unrelated", "people", "vehicles", "animals", "blurry"]:
                continue
                
            if score >= threshold:
                predictions.append({"class": best_class_normalized, "confidence": score, "used": True})
                
        # Sort by confidence descending
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        if not predictions:
            return [{"class": "normal", "confidence": probs.max().item(), "used": False}]
            
        return predictions
