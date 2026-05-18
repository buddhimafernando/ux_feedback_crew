import os
import json
import re
from dotenv import load_dotenv
import litellm

load_dotenv()

JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini/gemini-2.5-pro")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_judge(prompt):
    response = litellm.completion(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```json\s*|^```\s*|```$", "", content, flags=re.MULTILINE).strip()
    return json.loads(content)


def main():
    dataset = read_json("dataset.json")
    vision = read_json("vision_output.json")
    heuristic = read_json("heuristic_output.json")
    feedback = read_json("feedback_output.json")
    wireframe = read_text("wireframe_output.html")
    gt = dataset["ground_truth"]

    prompts = {
        "vision_analyst": f"""
You are a UX expert judge.

Evaluate the Vision Agent output against the ground truth.

Ground truth expected UI elements:
{json.dumps(gt["expected_ui_elements"], indent=2)}

Vision output:
{json.dumps(vision, indent=2)}

Score from 0 to 10:
1. component_accuracy
2. layout_understanding
3. completeness

Return only JSON:
{{
  "component_accuracy": number,
  "layout_understanding": number,
  "completeness": number,
  "overall": number
}}
""",
        "heuristic_evaluator": f"""
You are a UX expert judge.

Evaluate the Heuristic Agent output against the ground truth.

Ground truth expected issues:
{json.dumps(gt["expected_issues"], indent=2)}

Ground truth expected heuristics:
{json.dumps(gt["expected_heuristics"], indent=2)}

Heuristic output:
{json.dumps(heuristic, indent=2)}

Score from 0 to 10:
1. issue_accuracy
2. heuristic_mapping
3. severity_relevance

Return only JSON:
{{
  "issue_accuracy": number,
  "heuristic_mapping": number,
  "severity_relevance": number,
  "overall": number
}}
""",
        "feedback_specialist": f"""
You are a UX and developer-experience judge.

Evaluate the Feedback Agent output against the ground truth.

Ground truth expected issues:
{json.dumps(gt["expected_issues"], indent=2)}

Ground truth expected improvements:
{json.dumps(gt["expected_improvements"], indent=2)}

Feedback output:
{json.dumps(feedback, indent=2)}

Score from 0 to 10:
1. accuracy
2. actionability
3. completeness

Return only JSON:
{{
  "accuracy": number,
  "actionability": number,
  "completeness": number,
  "overall": number
}}
""",
        "wireframe_designer": f"""
You are a senior UI/UX judge.

Evaluate the Wireframe Agent output against the expected improvements.

Ground truth expected improvements:
{json.dumps(gt["expected_improvements"], indent=2)}

Wireframe HTML:
{wireframe[:12000]}

Score from 0 to 10:
1. improvement_implementation
2. usability_enhancement
3. completeness

Return only JSON:
{{
  "improvement_implementation": number,
  "usability_enhancement": number,
  "completeness": number,
  "overall": number
}}
"""
    }

    results = {}

    for agent_name, prompt in prompts.items():
        print(f"Judging {agent_name}...")
        results[agent_name] = call_judge(prompt)

    with open("judge_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved judge_results.json\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()