import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Point Airflow at your project root so imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["your-email@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_model_monitoring",
    default_args=default_args,
    description="Weekly fraud model drift detection and performance monitoring",
    schedule_interval="0 9 * * 1",  # Every Monday at 9am
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "monitoring", "fraud"],
) as dag:

    def ingest_batch(**context):
        """Pull latest production batch — in production: query data warehouse."""
        execution_date = context["execution_date"]
        week_num = (execution_date - datetime(2024, 1, 1)).days // 7 + 1
        print(f"Ingesting production batch for week {week_num}")
        context["ti"].xcom_push(key="week_num", value=week_num)
        return week_num

    def run_evidently(**context):
        """Run Evidently drift report against reference dataset."""
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        week_num = context["ti"].xcom_pull(task_ids="ingest_batch", key="week_num")
        batch_path = os.path.join(PROJECT_ROOT, f"data/production_batches/week_{week_num:02d}.csv")
        ref_path = os.path.join(PROJECT_ROOT, "data/reference_data.csv")

        reference = pd.read_csv(ref_path)
        current = pd.read_csv(batch_path)
        feature_cols = [c for c in reference.columns if c not in ["target", "week"]]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference[feature_cols], current_data=current[feature_cols])

        report_path = os.path.join(PROJECT_ROOT, f"reports/evidently/week_{week_num:02d}_drift.html")
        report.save_html(report_path)
        print(f"Evidently report saved: {report_path}")

    def check_and_alert(**context):
        """Check PSI thresholds and fire alerts if needed."""
        import pandas as pd
        import numpy as np
        import pickle

        week_num = context["ti"].xcom_pull(task_ids="ingest_batch", key="week_num")
        batch_path = os.path.join(PROJECT_ROOT, f"data/production_batches/week_{week_num:02d}.csv")
        ref_path = os.path.join(PROJECT_ROOT, "data/reference_data.csv")
        model_path = os.path.join(PROJECT_ROOT, "data/baseline_model.pkl")

        reference = pd.read_csv(ref_path)
        current = pd.read_csv(batch_path)
        feature_cols = [c for c in reference.columns if c not in ["target", "week"]]

        with open(model_path, "rb") as fh:
            model = pickle.load(fh)

        # PSI
        PSI_WARNING = 0.10
        PSI_CRITICAL = 0.20
        eps = 1e-4

        psi_values = {}
        for feat in feature_cols:
            ref_vals = reference[feat].dropna().values
            cur_vals = current[feat].dropna().values
            breakpoints = np.unique(np.nanpercentile(ref_vals, np.linspace(0, 100, 11)))
            if len(breakpoints) < 2:
                continue
            ref_c, _ = np.histogram(ref_vals, bins=breakpoints)
            cur_c, _ = np.histogram(cur_vals, bins=breakpoints)
            ref_pct = ref_c / len(ref_vals) + eps
            cur_pct = cur_c / len(cur_vals) + eps
            psi_values[feat] = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        max_psi = max(psi_values.values()) if psi_values else 0.0
        n_drifted = sum(1 for p in psi_values.values() if p > PSI_WARNING)

        from sklearn.metrics import f1_score
        y_pred = model.predict(current[feature_cols].values)
        f1 = f1_score(current["target"].values, y_pred, average="macro")

        from alert_manager import evaluate_alert
        evaluate_alert(week_num, max_psi, f1, n_drifted)

    def log_to_mlflow(**context):
        """Log monitoring results to MLflow."""
        import mlflow
        import pandas as pd
        import numpy as np
        import pickle

        week_num = context["ti"].xcom_pull(task_ids="ingest_batch", key="week_num")
        batch_path = os.path.join(PROJECT_ROOT, f"data/production_batches/week_{week_num:02d}.csv")
        ref_path = os.path.join(PROJECT_ROOT, "data/reference_data.csv")
        model_path = os.path.join(PROJECT_ROOT, "data/baseline_model.pkl")

        reference = pd.read_csv(ref_path)
        current = pd.read_csv(batch_path)
        feature_cols = [c for c in reference.columns if c not in ["target", "week"]]

        with open(model_path, "rb") as fh:
            model = pickle.load(fh)

        from sklearn.metrics import f1_score, average_precision_score
        y_pred = model.predict(current[feature_cols].values)
        y_proba = model.predict_proba(current[feature_cols].values)[:, 1]
        f1 = f1_score(current["target"].values, y_pred, average="macro")
        pr_auc = average_precision_score(current["target"].values, y_proba)

        mlflow.set_tracking_uri(os.path.join(PROJECT_ROOT, "mlflow_runs"))
        mlflow.set_experiment("fraud_monitoring")
        with mlflow.start_run(run_name=f"monitoring_week_{week_num:02d}"):
            mlflow.log_metric("f1_score", f1, step=week_num)
            mlflow.log_metric("pr_auc", pr_auc, step=week_num)
        print(f"MLflow logging complete for week {week_num}")

    t1 = PythonOperator(task_id="ingest_batch", python_callable=ingest_batch)
    t2 = PythonOperator(task_id="run_evidently", python_callable=run_evidently)
    t3 = PythonOperator(task_id="check_and_alert", python_callable=check_and_alert)
    t4 = PythonOperator(task_id="log_to_mlflow", python_callable=log_to_mlflow)

    t1 >> t2 >> t3 >> t4
