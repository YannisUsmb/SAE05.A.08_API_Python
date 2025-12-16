import os
import glob
import lightgbm as lgb
import torch

class ModelLoader:
    def __init__(self):
        # Asteroids.
        self.asteroid_models = {}
        self.default_version = None

        # Images.
        self.image_models = {}
        self.default_image_version = None

        self.models_path = os.getenv("MODEL_PATH", "/app/models")

    def load_models(self):
        """
        Loads all AI models into memory at startup.
        """
        print(f"Loading models from {self.models_path}...")

        # Load Asteroid Prediction Model (LightGBM).
        search_path = os.path.join(self.models_path, "asteroid_prediction_model_*.txt")
        model_files = sorted(glob.glob(search_path))

        if not model_files:
            print("No asteroid prediction models found.")
            self.asteroid_models = {}
            self.default_version = None
        else:
            self.asteroid_models = {}
            for file_path in model_files:
                try:
                    filename = os.path.basename(file_path)
                    version = filename.replace("asteroid_prediction_model_", "").replace(".txt", "")

                    self.asteroid_models[version] = lgb.Booster(model_file=file_path)
                    print(f"Asteroid Prediction Model {version} loaded successfully.")
                except Exception as e:
                    print(f"Error loading Asteroid Prediction Model {file_path}: {e}")
        
        if self.asteroid_models:
            self.default_version = list(self.asteroid_models.keys())[-1]
            print(f"Default model version set to: {self.default_version}.")

        # Load Celestial Body Classification Model (PyTorch).
        search_path_img = os.path.join(self.models_path, "image_classification_model_*.pt")
        img_files = sorted(glob.glob(search_path_img))

        if not img_files:
            print("No image classification models found.")
        else:
            self.image_models = {}
            for file_path in img_files:
                try:
                    filename = os.path.basename(file_path)
                    version = filename.replace("image_classification_model_", "").replace(".pt", "")
                    
                    model = torch.jit.load(file_path, map_location=torch.device('cpu'))
                    model.eval()
                    
                    self.image_models[version] = model
                    print(f"Loaded Image Classification Model: {version}.")
                except Exception as e:
                    print(f"Error loading image classification model {filename}: {e}.")

            if self.image_models:
                self.default_image_version = list(self.image_models.keys())[-1]
                print(f"Default Image Classification Model Version: {self.default_image_version}.")

# Singleton.
ml_models = ModelLoader()