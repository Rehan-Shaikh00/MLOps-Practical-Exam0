import json
import pandas as pd
def run_validation():
 with open("feature_store/registry.json", "r") as f:
  registry = json.load(f)
 df = pd.read_parquet(registry["storage_path"])
 assert df.isnull().sum().sum() == 0, "Null values detected!"
 expected = set(registry["features"] + [registry["entity_key"], registry["target"]])
 assert expected.issubset(set(df.columns)), "Feature Store schema mismatch!"
 assert set(df["target"].unique()) == {0, 1}, "Binary target must be {0, 1}!"
 print("[Validation] All Clinical schema and quality checks PASSED.")
if __name__ == "__main__":
 run_validation()
