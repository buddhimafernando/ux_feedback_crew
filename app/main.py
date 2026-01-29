from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from crew_pipeline import run_crew_pipeline

app = FastAPI()

# allow Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}  # in-memory job store

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload")
async def upload_screenshot(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    jobs[job_id] = {
        "status": "uploaded",
        "progress": 0.0,
        "step": None,
        "result": None,
    }

    return {"job_id": job_id}
