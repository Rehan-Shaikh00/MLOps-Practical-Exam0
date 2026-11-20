import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
def run_preprocessing():
 os.makedirs("data", exist_ok=True)
 cancer = load_breast_cancer(as_frame=True)
 df = cancer.frame.copy()
 df.columns = [c.replace(" ", "_") for c in df.columns]
 df = df.drop_duplicates().dropna().reset_index(drop=True)
 df.insert(0, "patient_id", range(1001, 1001 + len(df)))
 df["event_timestamp"] = pd.Timestamp.utcnow()
 df.to_csv("data/breast_cancer_cleaned.csv", index=False)
 print(f"[Preprocessing] Saved data/breast_cancer_cleaned.csv | Shape: {df.shape}")
if __name__ == "__main__":
 run_preprocessing()