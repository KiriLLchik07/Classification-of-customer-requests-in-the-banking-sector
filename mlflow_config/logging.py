import random
import subprocess
import mlflow
import numpy as np
import torch
import transformers
import platform

def log_environment(device: str):
    mlflow.log_params({
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device
    })

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode("utf-8").strip()

def log_git_commit():
    try:
        commit = get_git_commit()
        mlflow.log_param("git_commit", commit)
    except Exception:
        mlflow.log_param("git_commit", "unknown")
