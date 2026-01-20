from crewai.tools import tool
from google import genai
from dotenv import load_dotenv
from PIL import Image
import os
import io

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
        model="gemini-3-flash-preview",
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

    return result_text.strip()
