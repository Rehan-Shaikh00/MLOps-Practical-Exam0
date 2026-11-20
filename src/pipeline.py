import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from preprocessing import run_preprocessing
from feature_engineering import run_feature_engineering
from validation import run_validation

def run_training_and_selection():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Breast_Cancer_Classification_Team_G2")

    with open("feature_store/registry.json", "r") as f:
        registry = json.load(f)

    df = pd.read_parquet(registry["storage_path"])
    X = df[registry["features"]]
    y = df[registry["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "SVC_RBF": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=150, random_state=42)
    }

    best_f1, best_model, best_name = -1.0, None, ""
    for name, model in models.items():
        with mlflow.start_run(run_name=f"Cancer_{name}"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_param("feature_group", registry["feature_group"])
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("recall", recall_score(y_test, preds))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))
            mlflow.log_artifact("feature_store/registry.json")
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"[MLflow SQLite] {name:16s} | Acc: {acc:.4f} | F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1, best_model, best_name = f1, model, name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_cancer_model.pkl")
    print(f"[Model Selection] Winner: {best_name} (F1: {best_f1:.4f})")

if __name__ == "__main__":
    run_preprocessing()
    run_feature_engineering()
    run_validation()
    run_training_and_selection()