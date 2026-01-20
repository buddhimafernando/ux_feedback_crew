from crewai.tools import tool
from google import genai
import os
from dotenv import load_dotenv


@tool("generate_feedback")
def generate_feedback(vision_analysis: str, heuristic_evaluation: str) -> str:
    """
    Converts technical heuristic violations into developer-friendly,
    actionable feedback with priorities and improvement suggestions.
    
    Args:
        vision_analysis: JSON string of vision analysis
        heuristic_evaluation: JSON string of heuristic evaluation
        
    Returns:
        JSON string with actionable feedback items and priorities
    """

    load_dotenv()
    
    # Configure Gemini client
    api_key = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
Transform these UX violations into actionable developer feedback.

## VISION ANALYSIS:
{vision_analysis}

## VIOLATIONS:
{heuristic_evaluation}

## OUTPUT FORMAT:

Return ONLY JSON:

{{
  "feedback_items": [
    {{
      "id": 1,
      "title": "Action-oriented title",
      "priority": "high/medium/low",
      "why_it_matters": "User impact explanation",
      "what_to_do": ["step 1", "step 2"],
      "wireframe_changes": "Visual changes needed"
    }}
  ],
  "quick_wins": [
    {{
      "change": "Easy fix description",
      "impact": "Impact description",
      "effort": "5 minutes"
    }}
  ],
  "summary": {{
    "total_issues": 5,
    "high": 2,
    "medium": 2,
    "low": 1
  }}
}}

Return ONLY the JSON.
"""
    
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt
    )
    
    # Clean response
    result_text = response.text.strip()
    if result_text.startswith("```json"):
        result_text = result_text[7:]
    elif result_text.startswith("```"):
        result_text = result_text[3:]
    if result_text.endswith("```"):
        result_text = result_text[:-3]
    
    return result_text.strip()