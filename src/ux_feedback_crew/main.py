#!/usr/bin/env python
import sys
from pathlib import Path
from dotenv import load_dotenv

from ux_feedback_crew.crew import UxFeedbackCrew

# Load environment variables
load_dotenv()

# Add src/ to sys.path dynamically
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run():
    """
    Run the UX Feedback Crew
    """
    
    # Get screenshot path from command line or use default
    if len(sys.argv) > 1:
        screenshot_path = sys.argv[1]
    else:
        screenshot_path = "data/screenshots/test_image.png"
    
    # Verify screenshot exists
    if not Path(screenshot_path).exists():
        print(f"❌ Error: Screenshot not found at {screenshot_path}")
        print("\nUsage: python -m src.ux_feedback_crew.main <path_to_screenshot>")
        return
    
    print("=" * 70)
    print("🚀 UX FEEDBACK CREW - MULTI-AGENT SYSTEM")
    print("=" * 70)
    print(f"\n📸 Analyzing screenshot: {screenshot_path}\n")
    
    # Initialize and run crew
    inputs = {
        'screenshot_path': screenshot_path
    }
    
    crew = UxFeedbackCrew().crew()
    result = crew.kickoff(inputs=inputs)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n{result}\n")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'screenshot_path': 'data/screenshots/test_image.png'
    }
    try:
        UxFeedbackCrew().crew().train(
            n_iterations=int(sys.argv[1]),
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


if __name__ == "__main__":
    run()