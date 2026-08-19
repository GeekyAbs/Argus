from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_ROOT = Path(__file__).resolve().parents[1]

VEHICLE_WEIGHTS = PROJECT_ROOT / "argus-other-full-v1" / "best.pt"
SAFETY_WEIGHTS = PROJECT_ROOT / "argus-i-pytorch-default-v1" / "best.pt"

UPLOADS_DIR = WEBAPP_ROOT / "uploads"
OUTPUTS_DIR = WEBAPP_ROOT / "outputs"
FRONTEND_DIR = WEBAPP_ROOT / "frontend"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.60
IMG_SIZE = 640
STREAM_FPS = 20
FRAME_SKIP = 1  # process every Nth frame; raise on slow (CPU-only) hardware

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

VEHICLE_COLOR = (0, 220, 80)   # BGR, green
SAFETY_COLOR = (0, 165, 255)   # BGR, orange
