# ARGUS Live Detection Platform

Upload an image or video; two RT-DETR models (vehicle detection + rider safety/helmet
detection) run simultaneously on it and the tracked, classified results are shown live
in the browser as boxes drawn over the feed.

## Models

Weights are read from the repo's `models/` directory (see `backend/config.py`):

- `models/model2.pt` — vehicles (14 classes): truck, cyclist, bike, tempo, car, jeep, toto,
  e-rickshaw, auto-rickshaw, bus, van, cycle-rickshaw, person, taxi
- `models/model1.pt` — rider safety (3 classes): With Helmet, Without Helmet, licence

Both are RT-DETR-l checkpoints, loaded once at startup and run concurrently per frame with
ByteTrack (`persist=True`, `stream=True`) for stable track IDs, mirroring `dual_model_video.py`.

## Run

From the repo root:

```
uv sync
uv run uvicorn backend.main:app --app-dir webapp --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 — upload a file, the live feed and per-class counts update as
frames are processed. Finished videos get a download link to the annotated `.mp4`.

## Notes

- `pyproject.toml` installs the CUDA 12.6 PyTorch wheels. On a machine without an NVIDIA GPU,
  swap the `[[tool.uv.index]]` url back to `https://download.pytorch.org/whl/cpu`.
- `backend/config.py` knobs: `IMG_SIZE`, `CONFIDENCE`, `STREAM_FPS`, and `FRAME_SKIP`
  (process every Nth frame — skipped frames are written to the output video unannotated).
- One job runs at a time — a new upload replaces whatever is currently processing.
