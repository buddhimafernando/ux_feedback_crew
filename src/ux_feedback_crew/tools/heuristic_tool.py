from crewai.tools import tool
from google import genai
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


@tool("evaluate_heuristics")
def evaluate_heuristics(vision_analysis: str) -> str:
    """
    Evaluates UI designs against Nielsen's 10 Usability Heuristics and
    mobile UX best practices.
    
    Args:
        vision_analysis: JSON string of vision analysis results
        
    Returns:
        JSON string with violations, severity scores, and overall UX score
    """

    load_dotenv()
    
    # Configure Gemini client
    api_key = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key)
    
    # Load heuristics knowledge base
    heuristics_path = Path(__file__).parent.parent / "config" / "nielsen_heuristics.json"
    
    if heuristics_path.exists():
        with open(heuristics_path, 'r') as f:
            heuristics_data = json.load(f)
        heuristics_info = json.dumps(heuristics_data['heuristics'][:5], indent=2)
    else:
        heuristics_info = "Nielsen's 10 Usability Heuristics"
    
    # Evaluation prompt
    prompt = f"""
You are a UX evaluation expert. Evaluate this mobile UI against Nielsen's 10 Usability Heuristics.

## UI ANALYSIS:
{vision_analysis}

## HEURISTICS TO EVALUATE:
{heuristics_info}

## OUTPUT FORMAT:

Return ONLY valid JSON:

{{
  "violations": [
    {{
      "heuristic_id": 1,
      "heuristic_name": "Visibility of system status",
      "severity": "high/medium/low",
      "issue": "Description of problem",
      "affected_components": ["list"],
      "improvement_suggestion": "How to fix"
    }}
  ],
  "strengths": [
    {{
      "heuristic_name": "Name",
      "observation": "What works well"
    }}
  ],
  "overall_score": 7.5
}}

Return ONLY the JSON.
"""
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
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
    
    result_text = result_text.strip()
    
    # Save to JSON file
    output_dir = Path("data/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"heuristic_evaluation_{timestamp}.json"
    
    try:
        # Parse to validate JSON and pretty print
        json_data = json.loads(result_text)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Heuristic evaluation saved to: {output_path}")
    except json.JSONDecodeError as e:
        print(f"⚠ JSON validation error: {e}")
        # Save raw text anyway
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
    
    return result_text