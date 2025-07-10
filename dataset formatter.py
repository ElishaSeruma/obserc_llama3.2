import pandas as pd
from datasets import Dataset
import json

# Load the MentalManip dataset
df = pd.read_csv("mentalmanip_detailed.csv")

# Function to aggregate annotator labels
def aggregate_annotations(row):
    # Binary manipulation label (majority vote)
    manipulative_votes = [row[f"manipulative_{i}"] for i in range(1, 4)]
    manipulative = 1 if sum(manipulative_votes) >= 2 else 0
    
    # Aggregate techniques and vulnerabilities
    techniques = []
    vulnerabilities = []
    marks = []
    confidence = []
    
    for i in range(1, 4):
        if row[f"technique_{i}"] and row[f"technique_{i}"] != "cannot decide":
            techniques.extend(row[f"technique_{i}"].split(","))
        if row[f"vulnerability_{i}"]:
            vulnerabilities.extend(row[f"vulnerability_{i}"].split(","))
        if row[f"marks_{i}"]:
            marks.append(row[f"marks_{i}"])
        confidence.append(row[f"confidence_{i}"])
    
    # Remove duplicates and empty values
    techniques = ", ".join(sorted(set([t for t in techniques if t]))) if techniques else "None"
    vulnerabilities = ", ".join(sorted(set([v for v in vulnerabilities if v]))) if vulnerabilities else "None"
    marks = " | ".join(marks) if marks else "None"
    avg_confidence = sum(confidence) / len(confidence) if confidence else 0
    
    return {
        "manipulative": "Yes" if manipulative else "No",
        "techniques": techniques,
        "vulnerabilities": vulnerabilities,
        "marks": marks,
        "confidence": round(avg_confidence, 2),
        "agreement": row["agreement"]
    }

# Function to create question-answer pairs
def create_qa_pair(row):
    question = (
        "[INST] Analyze the following dialogue for mental manipulation. "
        "Is manipulation present? If yes, identify the technique(s) and vulnerability(ies) targeted. "
        f"Dialogue:\n{row['dialogue']} [/INST]"
    )
    annotations = aggregate_annotations(row)
    answer = json.dumps({
        "manipulative": annotations["manipulative"],
        "techniques": annotations["techniques"],
        "vulnerabilities": annotations["vulnerabilities"],
        "confidence": annotations["confidence"],
        "marks": annotations["marks"]
    }, indent=2)
    return pd.Series([question, answer])

# Apply transformation
qa_df = df.apply(create_qa_pair, axis=1)
qa_df.columns = ["question", "answer"]

# Filter high-agreement samples (optional, e.g., agreement >= 66.67%)
qa_df["agreement"] = df["agreement"]
qa_df = qa_df[qa_df["agreement"] >= 66.67].drop(columns=["agreement"])

# Save to CSV
qa_df.to_csv("mentalmanip_qa.csv", index=False)

# Convert to Hugging Face Dataset for fine-tuning
dataset = Dataset.from_pandas(qa_df)
dataset.save_to_disk("mentalmanip_qa_dataset")
