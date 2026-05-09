import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import pickle
import os
from sklearn.metrics import f1_score, average_precision_score
from alert_manager import evaluate_alert, PSI_WARNING, PSI_CRITICAL

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except Exception:
    EVIDENTLY_AVAILABLE = False

os.makedirs("reports/evidently", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

mlflow.set_tracking_uri("mlflow_runs")
mlflow.set_experiment("fraud_monitoring")

reference = pd.read_csv("data/reference_data.csv")
with open("data/baseline_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_cols = [c for c in reference.columns if c not in ["target", "week"]]


def compute_psi(ref_vals, cur_vals, n_bins=10, eps=1e-4):
    """Population Stability Index computed over quantile-based buckets."""
    breakpoints = np.nanpercentile(ref_vals, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_vals, bins=breakpoints)
    cur_counts, _ = np.histogram(cur_vals, bins=breakpoints)

    ref_pct = ref_counts / len(ref_vals) + eps
    cur_pct = cur_counts / len(cur_vals) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def _write_html_report(path, week, psi_values, max_psi):
    """Generate a lightweight HTML drift report when Evidently is unavailable."""
    rows = ""
    for feat, psi in sorted(psi_values.items(), key=lambda x: x[1], reverse=True):
        if psi >= PSI_CRITICAL:
            colour = "#f8d7da"
            badge = "CRITICAL"
        elif psi >= PSI_WARNING:
            colour = "#fff3cd"
            badge = "WARNING"
        else:
            colour = "#d1e7dd"
            badge = "OK"
        rows += (
            f"<tr style='background:{colour}'>"
            f"<td>{feat}</td><td>{psi:.4f}</td><td>{badge}</td></tr>\n"
        )
    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Drift Report — Week {week}</title>
<style>
  body{{font-family:sans-serif;padding:2rem;max-width:900px;margin:auto}}
  h1{{color:#1A3A6B}} table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #ccc;padding:.5rem .75rem;text-align:left}}
  th{{background:#1A3A6B;color:white}}
</style></head><body>
<h1>Drift Report — Week {week}</h1>
<p><strong>Max PSI:</strong> {max_psi:.4f} &nbsp;|&nbsp;
   <strong>Features drifted:</strong> {sum(1 for p in psi_values.values() if p > PSI_WARNING)}</p>
<h2>PSI per Feature</h2>
<table><tr><th>Feature</th><th>PSI</th><th>Status</th></tr>
{rows}</table>
<p style='margin-top:2rem;color:#666;font-size:.85rem'>
PSI thresholds: &lt;0.10 stable · 0.10–0.20 warning · &gt;0.20 critical</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


results_rows = []

print(f"{'Week':>4} | {'F1':>7} | {'PR-AUC':>7} | {'MaxPSI':>7} | {'Drifted':>7} | Status")
print("-" * 60)

for week in range(1, 13):
    batch_path = f"data/production_batches/week_{week:02d}.csv"
    if not os.path.exists(batch_path):
        continue

    current = pd.read_csv(batch_path)

    # ── PSI per feature (manual calculation — reliable across Evidently versions) ──
    psi_values = {}
    for feat in feature_cols:
        if feat in current.columns:
            psi_values[feat] = compute_psi(
                reference[feat].dropna().values,
                current[feat].dropna().values,
            )

    max_psi = max(psi_values.values()) if psi_values else 0.0
    n_drifted = sum(1 for p in psi_values.values() if p > PSI_WARNING)

    # ── HTML drift report ─────────────────────────────────────────────────────
    report_path = f"reports/evidently/week_{week:02d}_drift.html"
    if EVIDENTLY_AVAILABLE:
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=reference[feature_cols],
                current_data=current[feature_cols],
            )
            report.save_html(report_path)
        except Exception:
            report_path = _write_html_report(report_path, week, psi_values, max_psi)
    else:
        report_path = _write_html_report(report_path, week, psi_values, max_psi)

    # ── Model performance metrics ─────────────────────────────────────────────
    X_current = current[feature_cols].values
    y_current = current["target"].values
    y_pred = model.predict(X_current)
    y_proba = model.predict_proba(X_current)[:, 1]
    f1 = f1_score(y_current, y_pred, average="macro")
    pr_auc = average_precision_score(y_current, y_proba)

    # ── Alerting ──────────────────────────────────────────────────────────────
    alerts = evaluate_alert(week, max_psi, f1, n_drifted)
    alert_status = (
        "CRITICAL" if any(a["severity"] == "CRITICAL" for a in alerts)
        else "WARNING" if alerts
        else "OK"
    )

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"monitoring_week_{week:02d}"):
        mlflow.log_metric("f1_score", f1, step=week)
        mlflow.log_metric("pr_auc", pr_auc, step=week)
        mlflow.log_metric("max_psi", max_psi, step=week)
        mlflow.log_metric("n_drifted_features", n_drifted, step=week)
        mlflow.log_metric("alert_fired", 0 if alert_status == "OK" else 1, step=week)
        mlflow.log_param("week", week)
        mlflow.log_param("alert_status", alert_status)
        # Log top drifted features as params for easy filtering in MLflow UI
        top_drifted = sorted(psi_values.items(), key=lambda x: x[1], reverse=True)[:3]
        for feat, psi in top_drifted:
            mlflow.log_metric(f"psi_{feat}", psi, step=week)
        if report_path and os.path.exists(report_path):
            mlflow.log_artifact(report_path)

    results_rows.append({
        "week": week,
        "f1": f1,
        "pr_auc": pr_auc,
        "max_psi": max_psi,
        "n_drifted": n_drifted,
        "alert_status": alert_status,
    })

    print(
        f"{week:>4} | {f1:>7.4f} | {pr_auc:>7.4f} | {max_psi:>7.4f} | "
        f"{n_drifted:>7} | {alert_status}"
    )

# ── Save results CSV for the Streamlit dashboard ──────────────────────────────
results_df = pd.DataFrame(results_rows)
results_df.to_csv("outputs/monitoring_results.csv", index=False)
print("\nResults saved to outputs/monitoring_results.csv")
print("Run: streamlit run app/dashboard.py  to launch the dashboard")
