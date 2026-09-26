"""Loading registered models and running a prediction.

Preprocessing is a copy of A1's eval_transform - if it drifted from what the
models were trained with, every prediction here would silently be wrong.
"""

from __future__ import annotations

import time

import mlflow.pytorch
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from registry import MODEL_NAME

THRESHOLD = 0.5
IMG_SIZE = 224
EVAL_TRANSFORM = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

torch.set_num_threads(2)


def load_model(version: int) -> torch.nn.Module:
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{version}", map_location="cpu")
    return model.eval()


@torch.no_grad()
def predict(model: torch.nn.Module, img: Image.Image) -> tuple[float, float]:
    """Returns (P(pneumonia), milliseconds)."""
    t0 = time.perf_counter()
    x = EVAL_TRANSFORM(img.convert("L")).unsqueeze(0)
    prob = torch.sigmoid(model(x)).item()
    return prob, (time.perf_counter() - t0) * 1000


def image_stats(img: Image.Image) -> dict:
    """Same statistics as Step 1, so live images are comparable to the dataset versions."""
    small = img.convert("L")
    small.thumbnail((512, 512))
    arr = np.asarray(small, dtype=np.float32)
    return {"mean_intensity": float(arr.mean()), "std_intensity": float(arr.std())}
