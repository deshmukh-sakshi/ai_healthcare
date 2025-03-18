import os
import kaggle
from pathlib import Path

def download_model():
    """
    Downloads the brain tumor model from Kaggle if it doesn't exist locally.
    Returns the path to the model file.
    """
    model_path = 'brain_tumor_mri.keras'
    
    # If model doesn't exist locally, download it
    if not os.path.exists(model_path):
        print("Downloading model from Kaggle...")
        
        # Configure Kaggle credentials
        os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
        
        try:
            # Download the model from Kaggle
            kaggle.api.authenticate()
            kaggle.api.model_download(
                'esfiam/cnn-brain-tumor-detector',
                path='.'
            )
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise
    
    return model_path 