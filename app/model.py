import wandb
import torch
import os

from torchvision import models, resnet18, transforms
from PIL import Image

from dotenv import load_dotenv

load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
wand_api_key = os.getenv("WANDB_API_KEY")

def load_model_from_wandb(
    run_path: str, 
    model_filename: str, 
    local_dir: str = "./models"
    ):
    """
    Downloads a model artifact from Weights & Biases (wandb) and returns the local file path.
    :param run_path: e.g. "user/project/run_id"
    :param model_filename: e.g. "model.pt"
    :param local_dir: Target directory for the model
    :return: Local path to the model file
    """
    api = wandb.Api()
    run = api.run(run_path)
    artifact = run.use_artifact(f"{model_filename}:latest")
    artifact_dir = artifact.download(local_dir)
    return f"{artifact_dir}/{model_filename}"

def load_model(model_path: str):
    """
    Loads a PyTorch model from a given path.
    :param model_path: Path to the model file
    :return: Loaded PyTorch model
    """
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


