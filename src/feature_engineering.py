import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler
def run_feature_engineering():
 os.makedirs("feature_store", exist_ok=True)
 df = pd.read_csv("data/breast_cancer_cleaned.csv")
 df["area_to_perimeter_ratio"] = df["mean_area"] / (df["mean_perimeter"] + 1e-5)
 df["worst_to_mean_radius_ratio"] = df["worst_radius"] / (df["mean_radius"] + 1e-5)
 df["concavity_compactness_product"] = df["mean_concavity"] * df["mean_compactness"]
 meta_cols = ["patient_id", "event_timestamp", "target"]
 feature_cols = [c for c in df.columns if c not in meta_cols]
 scaler = RobustScaler()
 df[feature_cols] = scaler.fit_transform(df[feature_cols])
 joblib.dump(scaler, "feature_store/cancer_scaler.pkl")
 store_path = "feature_store/cancer_features.parquet"
 df.to_parquet(store_path, index=False)
 registry = {
 "feature_group": "breast_cancer_features",
 "version": "v1.0",
 "entity_key": "patient_id",
 "timestamp_key": "event_timestamp",
 "target": "target",
 "features": feature_cols,
 "storage_path": store_path
 }
 with open("feature_store/registry.json", "w") as f:
  json.dump(registry, f, indent=4)
 print(f"[Feature Store] Materialized {len(feature_cols)} features to {store_path}")
if __name__ == "__main__":
 run_feature_engineering()