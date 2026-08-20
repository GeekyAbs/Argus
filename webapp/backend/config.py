from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_ROOT = Path(__file__).resolve().parents[1]

# model2.pt -> 14-class vehicle/pedestrian RT-DETR-l
# model1.pt -> 3-class rider safety RT-DETR-l (With Helmet / Without Helmet / licence)
VEHICLE_WEIGHTS = PROJECT_ROOT / "models" / "model2.pt"
SAFETY_WEIGHTS = PROJECT_ROOT / "models" / "model1.pt"

UPLOADS_DIR = WEBAPP_ROOT / "uploads"
OUTPUTS_DIR = WEBAPP_ROOT / "outputs"
FRONTEND_DIR = WEBAPP_ROOT / "frontend"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.60
IMG_SIZE = 640
STREAM_FPS = 20
FRAME_SKIP = 3  # process every Nth frame; raise if the live feed lags behind

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

VEHICLE_COLOR = (0, 220, 80)   # BGR, green
SAFETY_COLOR = (0, 165, 255)   # BGR, orange
