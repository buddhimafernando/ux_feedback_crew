from crewai.tools import tool
from google import genai
import os
import webbrowser
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


@tool("create_wireframe")
def create_wireframe(vision_analysis: str, feedback_result: str) -> str:
    """
    Generates improved UI wireframes in HTML/CSS based on feedback.
    Creates interactive, exportable designs showing improvements.
    
    Args:
        vision_analysis: JSON string of vision analysis
        feedback_result: JSON string of feedback
        
    Returns:
        String with path to generated wireframe file
    """
    
    load_dotenv()

    # Configure Gemini client
    api_key = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
Create an improved mobile UI wireframe in HTML/CSS.

## ORIGINAL DESIGN:
{vision_analysis}

## IMPROVEMENTS TO IMPLEMENT:
{feedback_result}

## REQUIREMENTS:

Create a COMPLETE HTML file with:
1. Mobile-first design (375px width)
2. All feedback improvements implemented
3. Clean, modern styling
4. Proper spacing and typography

Return ONLY the complete HTML code between ```html and ```.
Make it look professional and implement all suggested improvements.
"""
    
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt
    )
    
    # Extract HTML
    html_code = response.text.strip()
    if "```html" in html_code:
        start = html_code.find("```html") + 7
        end = html_code.find("```", start)
        html_code = html_code[start:end].strip()
    
    # Save HTML
    output_dir = Path("data/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"wireframe_{timestamp}.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_code)
    
    # Automatically open the HTML file in the default browser
    try:
        webbrowser.open(f'file://{output_path.absolute()}')
        print(f"✓ Wireframe opened in browser: {output_path}")
    except Exception as e:
        print(f"⚠ Could not auto-open browser: {e}")
    
    return f"Wireframe generated and saved to: {output_path}"