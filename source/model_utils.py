import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Add RoMa path
sys.path.append(os.path.join(PROJECT_ROOT, "submodules", "RoMa"))

def load_roma_model_with_custom_cache(model_type="indoor", device="cuda"):
    """
    Load RoMa model with custom model cache directory instead of torch hub cache
    
    Args:
        model_type: "indoor" or "outdoor"
        device: Device to load the model on
    
    Returns:
        RoMa model instance
    """
    # Import RoMa functions
    from romatch import roma_indoor, roma_outdoor
    
    # Define URLs for models
    weight_urls = {
        "romatch": {
            "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
            "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
        },
        "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    }
    
    def download_model_to_custom_dir(url, filename):
        """Download model to custom model directory instead of torch hub cache"""
        model_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(model_path):
            print(f"Loading cached model from {model_path}")
            return torch.load(model_path, map_location='cpu')
        else:
            print(f"Downloading model to {model_path}")
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            torch.save(state_dict, model_path)
            return state_dict
    
    # Download models to custom directory
    roma_weights = download_model_to_custom_dir(
        weight_urls["romatch"][model_type], 
        f"roma_{model_type}.pth"
    )
    dinov2_weights = download_model_to_custom_dir(
        weight_urls["dinov2"], 
        "dinov2_vitl14_pretrain.pth"
    )
    
    # Load model with custom weights
    if model_type == "indoor":
        model = roma_indoor(device=device, weights=roma_weights, dinov2_weights=dinov2_weights)
    else:
        model = roma_outdoor(device=device, weights=roma_weights, dinov2_weights=dinov2_weights)
    
    return model