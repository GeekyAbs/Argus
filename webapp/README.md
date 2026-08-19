# ARGUS Live Detection Platform

Upload an image or video; two RT-DETR models (vehicle detection + rider safety/helmet
detection) run simultaneously on it and the tracked, classified results are shown live
in the browser as boxes drawn over the feed.

## Models

- `argus-other-full-v1/best.pt` — vehicles: truck, cyclist, bike, tempo, car, jeep, toto,
  e-rickshaw, auto-rickshaw, bus, van, cycle-rickshaw, person, taxi
- `argus-i-pytorch-default-v1/best.pt` — rider safety: With Helmet, Without Helmet, licence

Both are loaded once at startup and run concurrently per frame with ByteTrack
(`persist=True`, `stream=True`) for stable track IDs, mirroring `Argus/dual_model_video.py`.

## Run

```
pip install -r webapp/requirements.txt
uvicorn backend.main:app --reload --app-dir webapp
```

Open http://localhost:8000 — upload a file, the live feed and per-class counts update as
frames are processed. Finished videos get a download link to the annotated `.mp4`.

## Notes

- No CUDA detected in this environment; inference runs on CPU, so live video processing
  will be slower than real-time. Tune `IMG_SIZE` / `FRAME_SKIP` in `backend/config.py` if
  needed for your hardware.
- One job runs at a time — a new upload replaces whatever is currently processing.
