import pandas as pd
import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

def map_severity(sev):
    """Maps 0-4 scale to Agent priority labels"""
    if sev <= 1: return "low"
    if sev == 2: return "medium"
    return "high"

def calculate_ux_score(group):
    """Calculates a score based on expert severity"""
    # Logic: Start at 10, subtract points based on severity
    avg_sev = group['severity'].mean()
    score = max(0, 10 - (avg_sev * 2))
    return round(score, 2)

def migrate_dataset():
    csv_path = Path("data/ux_expert_evaluation_dataset_v2.csv")
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    # Load Data
    print(f" Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Connect to MongoDB
    uri = os.getenv("MONGO_URI") 
    if not uri:
        print("Error: MONGO_URI not found in .env file")
        return

    client = MongoClient(uri)
    db = client['heuruxagent_db']
    collection = db['expert_validation_dataset_v2']

    documents = []

    # Group by ui_id and perform Mapping
    print("Mapping expert evaluations to Agent JSON format...")
    for ui_id, group in df.groupby('ui_id'):
        first_row = group.iloc[0]
        
        feedback_items = []
        for _, row in group.iterrows():
            feedback_items.append({
                "title": row['heuristic'], 
                "priority": map_severity(row['severity']),
                "effort_estimate": "Medium", 
                "why_it_matters": row['issue'],
                "what_to_do": [row['suggestion']], 
                "wireframe_changes": f"Apply UI fix according to {row['heuristic']} standards."
            })
        
        # Build the structured document
        doc = {
            "ui_id": ui_id,
            "image_url": first_row['image_path'],
            "screen_type": first_row['screen_type'],
            "expert_report": {
                "feedback_items": feedback_items,
                "ux_score": {
                    "score": calculate_ux_score(group),
                    "grade": "Good" if calculate_ux_score(group) >= 7.0 else "Needs Improvement"
                },
                "summary": {
                    "total_issues": len(group),
                    "high": int(len(group[group['severity'] >= 3])),
                    "medium": int(len(group[group['severity'] == 2])),
                    "low": int(len(group[group['severity'] <= 1]))
                }
            },
            "metadata": {
                "source": first_row.get('source', 'internal_dataset'),
                "expert_label": "ui ux expert"
            }
        }
        documents.append(doc)

    # Insert into MongoDB
    if documents:
        print(f"Uploading {len(documents)} documents to MongoDB...")
        collection.insert_many(documents)
        print("Migration Successful!")
    else:
        print("No data found to migrate.")

if __name__ == "__main__":
    migrate_dataset()