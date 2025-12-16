import io
import torch
from PIL import Image
from torchvision import transforms
from fastapi import HTTPException, status
from app.core.model_loader import ml_models
from app.schemas.image_dto import CelestialBodyPredictionOutput

class ImageService:
    def __init__(self):
        # Configuration.
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Classes.
        self.classes = ['asteroid', 'black hole', 'comet', 'constellation', 'galaxy', 'nebula', 'planet', 'star']

    def predict(self, file_bytes: bytes, filename: str, content_type: str, version_id: str = None) -> CelestialBodyPredictionOutput:
        selected_version = version_id or ml_models.default_image_version
        
        if not selected_version or selected_version not in ml_models.image_models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="No image model loaded or version not found."
            )
        
        model = ml_models.image_models[selected_version]

        # Pretreatment of image.
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Invalid image file: {e}"
            )

        # Prediction.
        with torch.no_grad(): # No need to calculate gradients.
            output_normal = model(input_batch)

            input_flip_h = torch.flip(input_batch, dims=[3])
            output_flip_h = model(input_flip_h)

            input_flip_v = torch.flip(input_batch, dims=[2])
            output_flip_v = model(input_flip_v)

            output_avg = (output_normal + output_flip_h + output_flip_v) / 3.0
            
        # Interpretation (Softmax to get %).
        probabilities = torch.nn.functional.softmax(output_avg[0], dim=0)
        
        # We take the class with the highest probability.
        top_prob, top_catid = torch.topk(probabilities, 1)
        
        predicted_index = top_catid[0].item()
        confidence = top_prob[0].item()
        
        # If the model predicts something outside the list.
        predicted_label = self.classes[predicted_index] if predicted_index < len(self.classes) else "Unknown"

        return CelestialBodyPredictionOutput(
            filename=filename,
            content_type=content_type,
            predicted_class=predicted_label,
            confidence_score=confidence,
            model_version=selected_version
        )