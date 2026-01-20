# UX Feedback Crew 

Multi-agent AI system for automated UX evaluation and feedback generation.

## Features

- 📸 Vision Agent: Analyzes UI screenshots
- 🔍 Heuristic Agent: Evaluates against Nielsen's heuristics
- 💡 Feedback Agent: Generates actionable developer feedback
- 🎨 Wireframe Agent: Creates improved UI mockups

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment:
```bash
cp .env.example .env
# Add your GEMINI_API_KEY
```

3. Run the crew:
```bash
python -m src.ux_feedback_crew.main
```

## Project Structure

- `src/ux_feedback_crew/` - Main package
  - `tools/` - Agent tools (vision, heuristic, feedback, wireframe)
  - `config/` - Agent and task configurations
  - `crew.py` - Crew orchestration
  - `main.py` - Entry point

## Usage

Place a mobile UI screenshot in `data/screenshots/` and run:
```bash
python -m src.ux_feedback_crew.main --screenshot data/screenshots/your_image.png
```