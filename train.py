import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, classification_report

mlflow.set_tracking_uri("mlflow_runs")
mlflow.set_experiment("fraud_monitoring")

df = pd.read_csv("data/creditcard.csv")
df["Amount_log"] = np.log1p(df["Amount"])
df["Hour"] = (df["Time"] % 86400) / 3600
feature_cols = [c for c in df.columns if c not in ["Class", "Time", "Amount"]]

X = df[feature_cols]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="baseline_training"):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, average="macro")
    pr_auc = average_precision_score(y_test, y_proba)

    mlflow.log_params({"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05})
    mlflow.log_metrics({"baseline_f1": f1, "baseline_pr_auc": pr_auc})
    mlflow.xgboost.log_model(model, "model")

    print(f"Baseline — F1: {f1:.4f} | PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# Save reference dataset (training distribution) for Evidently comparison
X_train_ref = X_train.copy()
X_train_ref["target"] = y_train.values
X_train_ref.to_csv("data/reference_data.csv", index=False)
print(f"Reference dataset saved: {len(X_train_ref):,} rows")

os.makedirs("data", exist_ok=True)
with open("data/baseline_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Done. Run:  mlflow ui --backend-store-uri mlflow_runs  to view at localhost:5000")
