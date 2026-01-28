from crewai.tools import tool
from google import genai
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
from datetime import datetime
import os
import io
import json

# Load environment variables at the top
load_dotenv()


@tool("analyze_ui_screenshot")
def analyze_ui_screenshot(image_path: str) -> str:
    """
    Analyzes a mobile UI screenshot and extracts detailed information about
    components, layout, colors, typography, and accessibility.

    Args:
        image_path: Path to the mobile UI screenshot to analyze

    Returns:
        JSON string with comprehensive UI analysis
    """

    # Configure Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")
    client = genai.Client(api_key=api_key)

    # Load image
    img = Image.open(image_path)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=img.format or "PNG")
    img_bytes = img_byte_arr.getvalue()

    # Prompt for Gemini
    prompt = """
Analyze this mobile UI screenshot and extract detailed information.

Return ONLY valid JSON with this structure:

{
  "screen_type": "login/home/profile/list/etc",
  "components": [
    {
      "type": "button/text_input/image/label/icon/etc",
      "text": "visible text if any",
      "position": "top/middle/bottom/etc",
      "color": "describe color",
      "size": "small/medium/large"
    }
  ],
  "layout_structure": "describe overall layout",
  "color_scheme": {
    "primary_colors": ["list of main colors"],
    "background": "background color",
    "text_colors": ["list of text colors"]
  },
  "typography": {
    "heading_sizes": "describe sizes",
    "body_text_size": "describe size"
  },
  "spacing_and_density": {
    "overall_density": "tight/comfortable/spacious",
    "element_spacing": "describe spacing"
  },
  "accessibility_observations": ["list issues"],
  "notable_patterns": ["list UI patterns"]
}
"""

    # Gemini model call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            Image.open(image_path)
        ]
    )

    # Extract text output
    result_text = response.text.strip()

    # Remove markdown code blocks if present
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
    output_path = output_dir / f"vision_analysis_{timestamp}.json"
    
    try:
        # Parse to validate JSON and pretty print
        json_data = json.loads(result_text)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Vision analysis saved to: {output_path}")
    except json.JSONDecodeError as e:
        print(f"⚠ JSON validation error: {e}")
        # Save raw text anyway
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
    
    return result_text