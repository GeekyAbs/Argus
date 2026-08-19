from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .inference import load_models
from .processor import IDLE_FRAME, JobProcessor

STATE: dict = {"models": None, "job": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["models"] = load_models()
    yield


app = FastAPI(title="ARGUS Live Detection", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")


def _media_type_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in config.IMAGE_EXTENSIONS:
        return "image"
    if ext in config.VIDEO_EXTENSIONS:
        return "video"
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(config.FRONTEND_DIR / "index.html"))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    media_type = _media_type_for(file.filename)

    previous_job: JobProcessor | None = STATE["job"]
    if previous_job is not None and previous_job.state == "processing":
        previous_job.stop()

    ext = Path(file.filename).suffix.lower()
    dest = config.UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as out:
        out.write(await file.read())

    job = JobProcessor(models=STATE["models"], file_path=dest, media_type=media_type)
    STATE["job"] = job
    job.start()

    return JSONResponse({"job_id": job.id, "media_type": media_type})


@app.get("/api/status")
async def status() -> JSONResponse:
    job: JobProcessor | None = STATE["job"]
    if job is None:
        return JSONResponse({"state": "idle"})
    return JSONResponse(job.status())


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        job: JobProcessor | None = STATE["job"]
        jpeg = job.latest_jpeg() if job is not None else IDLE_FRAME
        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        )
        time.sleep(1 / config.STREAM_FPS)


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
