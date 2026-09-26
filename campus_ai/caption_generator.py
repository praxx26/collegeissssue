import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class CaptionGenerator:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
    def generate_caption(self, image: Image.Image, context_classes: list = None) -> str:
        """
        Generates a natural language description of the image.
        Uses YOLO-detected classes as context to guide the BLIP model, 
        ensuring it describes visually supported information.
        """
        if not context_classes or len(context_classes) == 0:
            return "The campus area appears clean and free of the specified issues."
            
        # Ensure image is in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        clean_classes = [c.replace("_", " ") for c in context_classes]
        issues_str = " and ".join(clean_classes)
        text = f"An image showing {issues_str}"
            
        inputs = self.processor(image, text, return_tensors="pt").to(self.device)
            
        # Generate caption with strict penalties to prevent "slumum" hallucination
        with torch.no_grad():
            out = self.model.generate(
                **inputs, 
                max_new_tokens=20,
                repetition_penalty=1.5,
                no_repeat_ngram_size=2
            )
            
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Capitalize first letter and add period if missing
        if caption:
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith('.'):
                caption += '.'
                
        return caption
