from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import RTDETR

from . import config

_EXECUTOR = ThreadPoolExecutor(max_workers=2)


@dataclass
class DualModels:
    vehicle: RTDETR
    safety: RTDETR


def load_models() -> DualModels:
    vehicle = RTDETR(str(config.VEHICLE_WEIGHTS))
    safety = RTDETR(str(config.SAFETY_WEIGHTS))
    vehicle.to(config.DEVICE)
    safety.to(config.DEVICE)
    return DualModels(vehicle=vehicle, safety=safety)


def track_frame(model: RTDETR, frame: np.ndarray):
    """Run one persistent ByteTrack step on a single frame, streamed."""
    results = model.track(
        source=frame,
        persist=True,
        stream=True,
        tracker="bytetrack.yaml",
        conf=config.CONFIDENCE,
        imgsz=config.IMG_SIZE,
        device=config.DEVICE,
        verbose=False,
    )
    return next(iter(results))


def draw_detections(frame: np.ndarray, result, color: tuple[int, int, int], prefix: str, counts: Counter) -> None:
    boxes = result.boxes
    if boxes is None or boxes.xyxy.numel() == 0:
        return

    coordinates = boxes.xyxy.cpu().numpy().astype(int)
    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(coordinates)

    for (x1, y1, x2, y2), class_id, confidence, track_id in zip(coordinates, classes, confidences, track_ids):
        name = result.names[class_id]
        counts[name] += 1
        identity = f" #{track_id}" if track_id is not None else ""
        label = f"{prefix}:{name}{identity} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1, height + 6)
        cv2.rectangle(frame, (x1, label_y - height - 6), (x1 + width + 4, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_legend(frame: np.ndarray, counts: Counter) -> None:
    if not counts:
        return
    lines = [f"{name}: {count}" for name, count in sorted(counts.items(), key=lambda kv: -kv[1])]
    x, y = 8, 20
    line_height = 18
    box_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in lines) + 16
    box_height = line_height * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, y - 16), (x - 6 + box_width, y - 16 + box_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_height


def run_dual_model(models: DualModels, frame: np.ndarray) -> tuple[np.ndarray, Counter]:
    """Run both RT-DETR models on one frame concurrently and draw combined results."""
    vehicle_future = _EXECUTOR.submit(track_frame, models.vehicle, frame)
    safety_future = _EXECUTOR.submit(track_frame, models.safety, frame)
    vehicle_result = vehicle_future.result()
    safety_result = safety_future.result()

    counts: Counter = Counter()
    draw_detections(frame, vehicle_result, config.VEHICLE_COLOR, "vehicle", counts)
    draw_detections(frame, safety_result, config.SAFETY_COLOR, "safety", counts)
    draw_legend(frame, counts)
    return frame, counts
