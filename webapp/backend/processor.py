from __future__ import annotations

import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .inference import DualModels, annotate, infer, run_dual_model


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return buffer.tobytes()


def _placeholder_frame(text: str) -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return _encode_jpeg(frame)


IDLE_FRAME = _placeholder_frame("ARGUS: waiting for upload...")


class JobProcessor:
    """Runs dual-model inference over an uploaded image/video on a background thread."""

    def __init__(self, models: DualModels, file_path: Path, media_type: str):
        self.id = uuid.uuid4().hex
        self.models = models
        self.file_path = file_path
        self.media_type = media_type  # "image" | "video"
        self.state = "processing"  # processing | done | error
        self.error: Optional[str] = None
        self.counts: Counter = Counter()
        self.output_path: Optional[Path] = None

        self._lock = threading.Lock()
        self._latest_jpeg: bytes = IDLE_FRAME
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def _set_latest(self, frame: np.ndarray) -> None:
        jpeg = _encode_jpeg(frame)
        with self._lock:
            self._latest_jpeg = jpeg

    def status(self) -> dict:
        return {
            "job_id": self.id,
            "state": self.state,
            "media_type": self.media_type,
            "error": self.error,
            "counts": dict(self.counts),
            "output_url": f"/outputs/{self.output_path.name}" if self.output_path else None,
        }

    def _run(self) -> None:
        try:
            if self.media_type == "image":
                self._run_image()
            else:
                self._run_video()
            if self.state != "error":
                self.state = "done"
        except Exception as exc:  # surfaced via /api/status
            self.state = "error"
            self.error = str(exc)

    def _run_image(self) -> None:
        frame = cv2.imread(str(self.file_path))
        if frame is None:
            raise ValueError(f"Could not read image: {self.file_path}")
        annotated, counts = run_dual_model(self.models, frame)
        self.counts = counts
        self._set_latest(annotated)

    def _run_video(self) -> None:
        capture = cv2.VideoCapture(str(self.file_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {self.file_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.output_path = config.OUTPUTS_DIR / f"{self.id}.mp4"
        writer = cv2.VideoWriter(str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {self.output_path}")

        frame_index = 0
        last_results: Optional[tuple] = None
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break

                # Only every FRAME_SKIP-th frame is run through the models, but every
                # frame gets boxes drawn on it -- skipped frames reuse the most recent
                # results so the output video does not flicker between annotated and
                # bare frames. Boxes on reused frames lag by up to FRAME_SKIP-1 frames.
                if frame_index % config.FRAME_SKIP == 0:
                    last_results = infer(self.models, frame)

                if last_results is not None:
                    self.counts = annotate(frame, *last_results)

                frame_index += 1
                writer.write(frame)
                self._set_latest(frame)
        finally:
            capture.release()
            writer.release()

        if self._stop_event.is_set():
            self.state = "error"
            self.error = "cancelled: replaced by a new upload"
