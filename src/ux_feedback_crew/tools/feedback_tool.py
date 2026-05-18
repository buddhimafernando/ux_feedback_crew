import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from crewai.tools import tool
from src.utils.context_guard import truncate_text

load_dotenv()

OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_name = os.getenv("FINETUNED_FEEDBACK_MODEL") or os.getenv("GENERIC_FEEDBACK_MODEL") or "gemini-2.5-flash"

import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="heuruxagent", location="us-central1")


# Helpers
def _extract_json(text: str) -> dict:
    text = text.strip()

    # remove code fences anywhere
    text = re.sub(r"```json\s*|```", "", text, flags=re.IGNORECASE).strip()

    # try full text first
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("No valid JSON found in model output")


def _normalize_feedback(data: dict) -> dict:
    """
    Normalize inconsistent keys that the model sometimes returns
    instead of the schema we defined in the prompt.

    Handles:
      - actionable_steps / how_to_fix / action_steps / technical_steps -> what_to_do
      - overall_ux_score (flat int) -> ux_score { score, grade, severity, reasoning }
      - summary as a plain string -> leave as-is (frontend handles both)
      - missing priority -> default to "low"
      - missing effort_estimate -> default to "N/A"
    """
    # Normalize feedback_items 
    for item in data.get("feedback_items", []):

        # Steps key normalization
        STEP_KEY_ALIASES = (
            "actionable_steps",
            "how_to_fix",
            "action_steps",
            "technical_steps",
            "steps",
            "action_items",
            "recommendation",
        )
        if "what_to_do" not in item:
            for alias in STEP_KEY_ALIASES:
                if alias in item:
                    item["what_to_do"] = item.pop(alias)
                    break

        # Ensuring what_to_do is always a list
        if "what_to_do" in item and isinstance(item["what_to_do"], str):
            item["what_to_do"] = [item["what_to_do"]]

        if "priority" not in item or not item["priority"]:
            item["priority"] = "low"
        else:
            item["priority"] = item["priority"].lower().strip()

        if "effort_estimate" not in item or not item["effort_estimate"]:
            item["effort_estimate"] = "N/A"

        if "why_it_matters" not in item:
            item["why_it_matters"] = item.pop("why", "") or ""

        if "wireframe_changes" not in item:
            item["wireframe_changes"] = None

    if "ux_score" not in data and "overall_ux_score" in data:
        raw_score = data.pop("overall_ux_score")
        score_int = int(raw_score) if isinstance(raw_score, (int, float)) else 0
        # Normalize 0-10 scale to 0-100
        if score_int <= 10:
            score_int = score_int * 10
        data["ux_score"] = {
            "score": score_int,
            "grade": _score_to_grade(score_int),
            "severity": _score_to_severity(score_int),
            "reasoning": data.get("summary", "") if isinstance(data.get("summary"), str) else "",
        }
    elif "ux_score" in data and isinstance(data["ux_score"], (int, float)):
        score_int = int(data["ux_score"])
        if score_int <= 10:
            score_int = score_int * 10
        data["ux_score"] = {
            "score": score_int,
            "grade": _score_to_grade(score_int),
            "severity": _score_to_severity(score_int),
            "reasoning": "",
        }
    elif "ux_score" in data and isinstance(data["ux_score"], dict):
        s = data["ux_score"]
        if isinstance(s.get("score"), (int, float)) and s["score"] <= 10:
            s["score"] = int(s["score"]) * 10

    #  Ensure summary counts match actual items 
    items = data.get("feedback_items", [])
    high   = sum(1 for i in items if i.get("priority") == "high")
    medium = sum(1 for i in items if i.get("priority") == "medium")
    low    = sum(1 for i in items if i.get("priority") == "low")

    if isinstance(data.get("summary"), dict):
        data["summary"]["total_issues"] = len(items)
        data["summary"]["high"]         = high
        data["summary"]["medium"]       = medium
        data["summary"]["low"]          = low
    elif isinstance(data.get("summary"), str):
        pass
    else:
        # If no summary at all synthesize a minimal one
        data["summary"] = {
            "total_issues": len(items),
            "high": high,
            "medium": medium,
            "low": low,
        }

    return data


def _score_to_grade(score: int) -> str:
    if score >= 85: return "excellent"
    if score >= 70: return "good"
    if score >= 50: return "average"
    return "poor"


def _score_to_severity(score: int) -> str:
    if score >= 75: return "low"
    if score >= 50: return "moderate"
    return "high"


def convert_feedback_to_markdown(feedback_data: dict) -> str:
    md = "# 📋 UX Feedback Report\n\n---\n\n"

    # Summary
    if "summary" in feedback_data:
        s = feedback_data["summary"]
        if isinstance(s, dict):
            total  = s.get('total_issues', 0)
            high   = s.get('high', 0)
            med    = s.get('medium', 0)
            low    = s.get('low', 0)
            effort = s.get('estimated_total_effort', 'N/A')
            md += "## 📊 Summary\n\n"
            md += f"> 🔢 **{total} issues found** — "
            md += f"🔴 {high} High · 🟡 {med} Medium · 🟢 {low} Low\n"
            md += f"> ⏱ Estimated Effort: **{effort}**\n\n---\n\n"
        elif isinstance(s, str) and s.strip():
            md += "## 📊 Summary\n\n"
            md += f"{s}\n\n---\n\n"

    # UX Score
    if "ux_score" in feedback_data:
        score_data = feedback_data["ux_score"]
        score     = score_data.get('score', 0)
        grade     = str(score_data.get('grade', 'N/A')).upper()
        severity  = score_data.get('severity', 'N/A')
        reasoning = score_data.get('reasoning', 'N/A')
        if isinstance(score, (int, float)) and score <= 10:
            score = int(score * 10)
        md += "## 🎯 Overall UX Score\n\n"
        md += f"> ### {score} / 100 — {grade}\n"
        md += f"> **Severity:** {severity}\n\n"
        md += f"{reasoning}\n\n---\n\n"

    # Quick Wins
    if feedback_data.get("quick_wins"):
        md += "## ⚡ Quick Wins\n\n"
        for w in feedback_data["quick_wins"]:
            md += f"- ✅ **{w.get('change', 'N/A')}**\n"
            md += f"  - 💡 {w.get('impact', 'N/A')}\n"
            md += f"  - ⏱ Effort: `{w.get('effort', 'N/A')}`\n"
        md += "\n---\n\n"

    # Recommendations grouped by priority
    items = feedback_data.get("feedback_items", [])
    grouped = {"high": [], "medium": [], "low": []}
    for item in items:
        p = (item.get("priority") or "low").lower()
        grouped.setdefault(p, []).append(item)

    priority_config = [
        ("high",   "🔴 High Priority",   "HIGH"),
        ("medium", "🟡 Medium Priority", "MEDIUM"),
        ("low",    "🟢 Low Priority",    "LOW"),
    ]

    md += "## 🔧 Detailed Recommendations\n\n"
    for key, section_title, tag in priority_config:
        bucket = grouped.get(key, [])
        if not bucket:
            continue
        md += f"### {section_title}\n\n"
        for item in bucket:
            title  = item.get('title', 'Recommendation')
            effort = item.get('effort_estimate', 'N/A')
            why    = item.get('why_it_matters', 'N/A')
            steps  = item.get('what_to_do', [])   
            wf     = item.get('wireframe_changes', 'N/A')

            md += f"#### `[{tag}]` {title}\n\n"
            md += f"> ⏱ Effort: **{effort}**\n\n"
            md += f"**💬 Why it matters**\n\n{why}\n\n"
            md += "**🛠 Implementation Steps**\n\n"
            if isinstance(steps, list):
                for i, step in enumerate(steps, 1):
                    md += f"{i}. {step}\n"
            else:
                md += f"1. {steps}\n"
            md += f"\n**✏️ Wireframe Changes**\n\n{wf}\n\n---\n\n"

    return md


# Tool
@tool("generate_feedback")
def generate_feedback(vision_analysis: str, heuristic_evaluation: str, evaluation_id: str = "") -> str:
    """
    Convert UX violations into developer-friendly feedback JSON and save report.
    Also estimate an overall UX score from 0-100 based on the severity and number of usability issues.

    Args:
        vision_analysis: JSON string from vision tool.
        heuristic_evaluation: JSON string from heuristic tool.
        evaluation_id: Optional ID used to name the saved files.

    Returns:
        Markdown string of the feedback report.
    """
    vision_analysis      = truncate_text(vision_analysis, 4000)
    heuristic_evaluation = truncate_text(heuristic_evaluation, 4000)

    prompt = f"""
TASK: Convert UX violations into structured UX feedback.

You are a UX expert. Based on the given inputs, generate a structured JSON report.

VISION ANALYSIS:
{vision_analysis}

HEURISTIC EVALUATION:
{heuristic_evaluation}

Return ONLY valid JSON. No explanations. No markdown. No extra text.

Use EXACTLY this structure:

{{
  "feedback_items": [
    {{
      "title": "...",
      "priority": "high|medium|low",
      "effort_estimate": "low|medium|high",
      "why_it_matters": "...",
      "what_to_do": ["step 1", "step 2"],
      "wireframe_changes": "..."
    }}
  ],
  "ux_score": {{
    "score": 7.5,
    "grade": "A|B|C|D|F"
  }},
  "summary": {{
    "total_issues": 10,
    "high": 2,
    "medium": 5,
    "low": 3
  }}
}}

STRICT RULES:

1. Do NOT include any fields other than:
   - feedback_items
   - ux_score
   - summary

2. Each feedback item MUST include:
   - title
   - priority
   - effort_estimate
   - why_it_matters
   - what_to_do (array of strings)
   - wireframe_changes

3. "what_to_do" MUST always be a list of strings.

4. Keep recommendations practical and UI-specific.

5. Avoid generic phrases like "apply best practices".

6. UX score MUST be between 0–10 (can include decimals).

7. Grade mapping:
   - 8.5–10 → A
   - 7–8.4 → B
   - 5–6.9 → C
   - 3–4.9 → D
   - <3 → F

8. Summary counts MUST match feedback_items exactly.

9. DO NOT wrap JSON in ``` or add any explanation.

RETURN ONLY JSON.
"""

    model = GenerativeModel(model_name)
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1
            }
        )
    except Exception as e:
        return f"Error calling model: {e}"

    raw_text = (response.text or "").strip()

    try:
        parsed_data = _extract_json(raw_text)
    except Exception as e:
        print(f"JSON parse error: {e}")

        file_id = evaluation_id if evaluation_id else "latest"
        raw_path = OUTPUT_DIR / f"feedback_raw_{file_id}.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        return raw_text

    parsed_data = _normalize_feedback(parsed_data)

    file_id   = evaluation_id if evaluation_id else "latest"
    json_path = OUTPUT_DIR / f"feedback_{file_id}.json"
    md_path   = OUTPUT_DIR / f"feedback_{file_id}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)

    md_content = convert_feedback_to_markdown(parsed_data)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✓ Saved → {json_path} | {md_path}")

    return json.dumps(parsed_data, ensure_ascii=False)