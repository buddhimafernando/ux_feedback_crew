from crewai.tools import tool
from google import genai
import os
import json
from pathlib import Path
from datetime import datetime
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
    
    # Save to both JSON and Markdown
    output_dir = Path("data/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON version
    json_path = output_dir / f"feedback_{timestamp}.json"
    try:
        json_data = json.loads(result_text)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Feedback JSON saved to: {json_path}")
        
        # Convert to Markdown
        md_path = output_dir / f"feedback_{timestamp}.md"
        markdown_content = convert_feedback_to_markdown(json_data)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"✓ Feedback Markdown saved to: {md_path}")
        
    except json.JSONDecodeError as e:
        print(f"⚠ JSON validation error: {e}")
        # Save raw text
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
    
    return result_text


def convert_feedback_to_markdown(feedback_data: dict) -> str:
    """Convert feedback JSON to formatted Markdown"""
    
    md = f"# UX Feedback Report\n\n"
    md += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    # Summary
    if 'summary' in feedback_data:
        summary = feedback_data['summary']
        md += f"## Summary\n\n"
        md += f"- **Total Issues:** {summary.get('total_issues', 0)}\n"
        md += f"- **High Priority:** {summary.get('high', 0)}\n"
        md += f"- **Medium Priority:** {summary.get('medium', 0)}\n"
        md += f"- **Low Priority:** {summary.get('low', 0)}\n\n"
    
    # Feedback Items
    if 'feedback_items' in feedback_data:
        md += f"## Feedback Items\n\n"
        for item in feedback_data['feedback_items']:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.get('priority', 'low'), '⚪')
            md += f"### {priority_emoji} {item.get('title', 'Untitled')}\n\n"
            md += f"**Priority:** {item.get('priority', 'N/A').upper()}\n\n"
            md += f"**Why it matters:** {item.get('why_it_matters', 'N/A')}\n\n"
            
            if 'what_to_do' in item and item['what_to_do']:
                md += f"**What to do:**\n"
                for step in item['what_to_do']:
                    md += f"- {step}\n"
                md += "\n"
            
            if 'wireframe_changes' in item:
                md += f"**Wireframe changes:** {item['wireframe_changes']}\n\n"
            
            md += "---\n\n"
    
    # Quick Wins
    if 'quick_wins' in feedback_data:
        md += f"## ⚡ Quick Wins\n\n"
        for win in feedback_data['quick_wins']:
            md += f"### {win.get('change', 'N/A')}\n\n"
            md += f"- **Impact:** {win.get('impact', 'N/A')}\n"
            md += f"- **Effort:** {win.get('effort', 'N/A')}\n\n"
    
    return md