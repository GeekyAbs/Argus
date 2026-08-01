from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import kagglehub
from ultralytics import RTDETR

# Set this to the video you want to process.
VIDEO_PATH = Path("/path/to/video.mp4")
OUTPUT_PATH = Path("output/tracked_video.mp4")
CONFIDENCE = 0.60


def model_weights(handle: str) -> Path:
    """Download a Kaggle model and return its single RT-DETR .pt weight file."""
    files = sorted(Path(kagglehub.model_download(handle)).rglob("*.pt"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one .pt file from {handle}, found {len(files)}")
    return files[0]


def track_frame(model, frame, confidence: float):
    """Run one persistent ByteTrack instance."""
    return model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=confidence,
        verbose=False,
    )[0]


def draw_detections(frame, result, color: tuple[int, int, int], prefix: str) -> None:
    boxes = result.boxes
    if boxes is None or boxes.xyxy.numel() == 0:
        return

    coordinates = boxes.xyxy.cpu().numpy().astype(int)
    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(coordinates)

    for (x1, y1, x2, y2), class_id, confidence, track_id in zip(coordinates, classes, confidences, track_ids):
        name = result.names[class_id]
        identity = f" #{track_id}" if track_id is not None else ""
        label = f"{prefix}:{name}{identity} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(y1, height + 6)
        cv2.rectangle(frame, (x1, label_y - height - 6), (x1 + width + 4, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


async def process_video(vehicle_model, safety_model) -> None:
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            vehicle_result, safety_result = await asyncio.gather(
                asyncio.to_thread(track_frame, vehicle_model, frame, CONFIDENCE),
                asyncio.to_thread(track_frame, safety_model, frame, CONFIDENCE),
            )
            draw_detections(frame, vehicle_result, (0, 220, 80), "vehicle")
            draw_detections(frame, safety_result, (0, 165, 255), "safety")
            writer.write(frame)

            cv2.imshow("ARGUS: green=vehicle | orange=safety", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video does not exist: {VIDEO_PATH}")

    safety_weights = model_weights("abeshchakraborty/argus-i/pyTorch/default")
    vehicle_weights = model_weights("abeshchakraborty/argus/other/full")
    print(f"Vehicle weights: {vehicle_weights}")
    print(f"Safety weights: {safety_weights}")
    asyncio.run(process_video(RTDETR(str(vehicle_weights)), RTDETR(str(safety_weights))))
    print(f"Saved tracked video to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
