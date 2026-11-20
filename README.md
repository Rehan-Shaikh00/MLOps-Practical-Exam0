# -MLOps-Practical-Exam0
# MLOps Practical Exam: Wine Classification (TY DS A Group G2)
End-to-end collaborative MLOps pipeline built by 6 team members using Git branches, DVC, Parquet Feature
Store, and MLflow SQLite tracking.
## Team Contributions
- **Member 1 (ML Pipeline):** `src/pipeline.py` (MLflow SQLite tracking & model selection)
- **Member 2 (DVC):** `dvc.yaml` & `dvc.lock` (Reproducible 3-stage pipeline)
- **Member 3 (Preprocessing):** `src/preprocessing.py`
- **Member 4 (Feature Store):** `src/feature_engineering.py` (Parquet Feature Store)
- **Member 5 (Validation):** `src/validation.py`
- **Member 6 (Docs & QA):** `README.md` & `tests/test_integration.py`
## Quickstart
```bash
pip install -r requirements.txt
dvc repro
pytest -v
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

