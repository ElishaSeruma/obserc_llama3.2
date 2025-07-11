import pandas as pd
from datasets import Dataset
import json
import logging
import os
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename=r"C:\Users\elish\Desktop\OBSERC TEST\Dataformatting\outputs\format.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define file paths
INPUT_CSV = r"C:\Users\elish\Desktop\OBSERC TEST\Dataformatting\data\mentalmanip_detailed.csv"
OUTPUT_CSV = r"C:\Users\elish\Desktop\OBSERC TEST\Dataformatting\outputs\mentalmanip_qa.csv"
OUTPUT_DATASET = r"C:\Users\elish\Desktop\OBSERC TEST\Dataformatting\outputs\mentalmanip_qa_dataset"

# Function to aggregate annotator labels
def aggregate_annotations(row):
    try:
        # Binary manipulation label (majority vote)
        manipulative_votes = [row[f"manipulative_{i}"] for i in range(1, 4)]
        manipulative = 1 if sum(manipulative_votes) >= 2 else 0
        
        # Aggregate techniques, vulnerabilities, and marks
        techniques = []
        vulnerabilities = []
        marks = []
        confidence = []
        
        for i in range(1, 4):
            # Handle techniques
            technique_val = row[f"technique_{i}"]
            if isinstance(technique_val, str) and technique_val != "cannot decide":
                techniques.extend(technique_val.split(","))
            
            # Handle vulnerabilities
            vulnerability_val = row[f"vulnerability_{i}"]
            if isinstance(vulnerability_val, str):
                vulnerabilities.extend(vulnerability_val.split(","))
            
            # Handle marks
            mark_val = row[f"marks_{i}"]
            if isinstance(mark_val, str):
                marks.append(mark_val)
            
            # Handle confidence
            confidence_val = row[f"confidence_{i}"]
            if not np.isnan(confidence_val):
                confidence.append(confidence_val)
        
        # Remove duplicates and empty values
        techniques = ", ".join(sorted(set([t.strip() for t in techniques if t.strip()]))) if techniques else "None"
        vulnerabilities = ", ".join(sorted(set([v.strip() for v in vulnerabilities if v.strip()]))) if vulnerabilities else "None"
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
    except Exception as e:
        logging.error(f"Error processing row {row.get('inner_id', 'unknown')}: {str(e)}")
        raise

# Function to create question-answer pairs
def create_qa_pair(row):
    try:
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
    except Exception as e:
        logging.error(f"Error creating QA pair for row {row.get('inner_id', 'unknown')}: {str(e)}")
        raise

# Main execution
def main():
    logging.info("Starting dataset processing...")
    
    # Verify input file exists
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}")
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Load dataset
    logging.info(f"Loading dataset from {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        logging.warning("UTF-8 encoding failed, trying latin1")
        df = pd.read_csv(INPUT_CSV, encoding="latin1")
    
    # Verify expected columns
    expected_columns = [
        "inner_id", "id", "dialogue", "original movie dialogue", "movie name", "agreement",
        "annotator_1", "manipulative_1", "technique_1", "victim_1", "vulnerability_1", "marks_1", "confidence_1",
        "annotator_2", "manipulative_2", "technique_2", "victim_2", "vulnerability_2", "marks_2", "confidence_2",
        "annotator_3", "manipulative_3", "technique_3", "victim_3", "vulnerability_3", "marks_3", "confidence_3"
    ]
    if not all(col in df.columns for col in expected_columns):
        missing = [col for col in expected_columns if col not in df.columns]
        logging.error(f"Missing columns in dataset: {missing}")
        raise ValueError(f"Missing columns in dataset: {missing}")
    
    logging.info(f"Loaded dataset with {len(df)} rows")
    
    # Apply transformation
    qa_df = df.apply(create_qa_pair, axis=1)
    qa_df.columns = ["question", "answer"]
    
    # Filter high-agreement samples (agreement >= 66.67%)
    qa_df["agreement"] = df["agreement"]
    qa_df = qa_df[qa_df["agreement"] >= 66.67].drop(columns=["agreement"])
    
    logging.info(f"Processed {len(qa_df)} samples after filtering")
    
    # Save to CSV
    qa_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logging.info(f"Saved QA pairs to {OUTPUT_CSV}")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(qa_df)
    dataset.save_to_disk(OUTPUT_DATASET)
    logging.info(f"Saved Hugging Face dataset to {OUTPUT_DATASET}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        raise
