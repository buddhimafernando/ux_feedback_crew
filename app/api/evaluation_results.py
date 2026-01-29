from fastapi import Path


@app.post("/evaluate-ui/")
async def evaluate_ui(file: UploadFile = File(...)):
    # 1. Save upload
    filename = f"{uuid.uuid4()}.png"
    upload_path = Path("uploads") / filename

    with open(upload_path, "wb") as f:
        f.write(await file.read())
    # Run Phase 1 Crew
    crew_instance = UxFeedbackCrew()
    result = crew_instance.evaluation_crew().kickoff(inputs={'image_path': str(upload_path)})

    # Save evaluation result (result.raw contains the feedback report)
    output_path = Path("outputs") / f"{upload_path.stem}_evaluation.json"
    with open(output_path, "w") as f:
        json.dump({"report": result.raw}, f)

    return {"evaluation_id": upload_path.stem, "evaluation": result.raw}

