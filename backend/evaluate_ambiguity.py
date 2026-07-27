import pandas as pd
import requests
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Load your labeled CSV dataset
CSV_PATH = "backend/data/requirements_dataset.csv"  # Update path if needed
df = pd.read_csv(CSV_PATH)

# Assume your CSV has 'Context' (requirement text) and 'DetectedAs' (label)
y_true = []
y_pred_llm = []
y_pred_ml = []

print("Running sentence-level ambiguity evaluation...")

for idx, row in df.iterrows():
    req_text = str(row["Context"]).strip()
    raw_label = str(row["DetectedAs"]).lower()
    
    # Normalize ground truth: 'ambiguous' vs 'clear'
    ground_truth = "ambiguous" if "ambig" in raw_label else "clear"
    y_true.append(ground_truth)

    # 1. Test LLM Endpoint
    try:
        res_llm = requests.post("http://localhost:3000/api/check-llm", json={"requirement": req_text}).json()
        y_pred_llm.append(res_llm.get("verdict", "clear"))
    except Exception as e:
        y_pred_llm.append("clear")

    # 2. Test ML Endpoint
    try:
        res_ml = requests.post("http://localhost:3000/api/check-ml", json={"requirement": req_text}).json()
        y_pred_ml.append(res_ml.get("verdict", "clear"))
    except Exception as e:
        y_pred_ml.append("clear")

print("\n" + "="*50)
print("             LLM BENCHMARK RESULTS")
print("="*50)
print(f"Accuracy: {accuracy_score(y_true, y_pred_llm) * 100:.2f}%")
print(classification_report(y_true, y_pred_llm))

print("="*50)
print("             ML BENCHMARK RESULTS")
print("="*50)
print(f"Accuracy: {accuracy_score(y_true, y_pred_ml) * 100:.2f}%")
print(classification_report(y_true, y_pred_ml))