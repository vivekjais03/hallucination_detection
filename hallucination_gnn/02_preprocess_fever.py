import pandas as pd
import json
import os
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}

def preprocess(input_csv, output_jsonl, max_evidence=5):
    df = pd.read_csv(input_csv)

    cleaned = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing {input_csv}"):
        claim = str(row.get("claim", "")).strip()
        label_text = str(row.get("label", "")).strip()

        if claim == "" or label_text not in LABEL_MAP:
            continue

        # In this dataset, evidence is already in a column called "evidence"
        evidence_col = row.get("evidence", "")

        evidence_list = []

        # evidence is stored as string -> safely convert to list
        try:
            import ast
            evidence_list = ast.literal_eval(evidence_col) if isinstance(evidence_col, str) else []
        except:
            evidence_list = []

        # keep only valid evidence sentences
        evidence_list = [str(x).strip() for x in evidence_list if str(x).strip() != ""]
        evidence_list = evidence_list[:max_evidence]

        cleaned.append({
            "claim": claim,
            "evidence": evidence_list,
            "label_text": label_text,
            "label": LABEL_MAP[label_text]
        })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item) + "\n")

    print(f"Saved preprocessed file: {output_jsonl}")
    print(f"Total samples: {len(cleaned)}")

if __name__ == "__main__":
    preprocess("data/train_raw.csv", "data/train_clean.jsonl")
    preprocess("data/val_raw.csv", "data/val_clean.jsonl")
