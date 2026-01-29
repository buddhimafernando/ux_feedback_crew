@app.post("/generate-wireframe/")
async def generate_wireframe(evaluation_id: str):
    evaluation_path = Path("outputs") / f"{evaluation_id}_evaluation.json"
    
    with open(evaluation_path) as f:
        evaluation_data = json.load(f)

    # Run Phase 2 Crew
    crew_instance = UxFeedbackCrew()
    # Pass the previous feedback report into the wireframe designer
    result = crew_instance.wireframe_crew().kickoff(inputs={'feedback': evaluation_data['report']})

    return {"wireframe_output": result.raw}
