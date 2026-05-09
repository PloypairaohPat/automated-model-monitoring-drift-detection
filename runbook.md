# Fraud Model Monitoring — Operational Runbook

## Overview

This runbook describes how to respond to monitoring alerts from the fraud detection model
monitoring pipeline. Written for on-call ML engineers and fraud operations team members.

The pipeline runs weekly via Airflow and generates:
- An Evidently HTML drift report per batch
- PSI values and F1/PR-AUC metrics logged to MLflow
- A JSONL alert log at `outputs/alert_log.jsonl`
- A monitoring results CSV at `outputs/monitoring_results.csv`

---

## Alert: WARNING — PSI between 0.10 and 0.20

**What it means:** One or more input features have shifted slightly from the training
distribution. May be temporary fluctuation or early signal of genuine distribution shift.

**Immediate steps:**
1. Open the Evidently report for the current week (`reports/evidently/week_XX_drift.html`)
2. Identify which features triggered the warning
3. Ask data engineering: any recent upstream pipeline changes?
4. Compare the current week's distributions to the previous 3 weeks
5. Consistent shift across weeks → likely genuine drift
6. Single-week spike → likely a data pipeline issue

**Do NOT retrain yet.** Monitor for 2 more weeks.

---

## Alert: CRITICAL — PSI >= 0.20

**What it means:** Significant drift confirmed. Model performance is likely degrading or
will degrade within 1–2 weeks.

**Immediate steps:**
1. Confirm alert is not a data pipeline outage (check row counts in the batch file)
2. Identify all drifted features and their PSI values from the MLflow run
3. Investigate whether drift is seasonal (expected) or anomalous
4. If F1 has also dropped below 0.80: begin retraining immediately
5. If performance still above threshold: schedule retraining for next sprint

**Retraining process:**
- Retrain on rolling 90-day window of production data
- Evaluate new model on a held-out recent test set
- Compare new model vs current model on the drifted features
- Deploy only if performance improves on the holdout set

---

## Alert: CRITICAL — F1 below 0.80

**What it means:** Model is no longer meeting the performance SLA.

**Immediate steps:**
1. Check if data drift was the root cause (review PSI logs in MLflow)
2. If drift preceded the F1 drop → retraining is the fix
3. If no drift detected → investigate label quality or data pipeline issues
4. Consider fallback to a rule-based system while retraining

---

## Updating the Reference Dataset

After a confirmed distribution shift and successful retraining:
1. Update `data/reference_data.csv` with the most recent 60 days of production data
2. Re-run all historical batches against the new reference to recalibrate PSI baselines
3. Document the reference update date in the model changelog

---

## When NOT to Retrain

- Single-week PSI spike that returns to normal → likely noise, not drift
- F1 drop during a known seasonal period (e.g. December fraud spikes) → expected, not failure
- PSI warning on one feature only → monitor, do not retrain yet

---

## Running the Pipeline Manually

```bash
# Step 1: Train baseline model (once)
python train.py

# Step 2: Generate 12 weekly batches
python simulate_production.py

# Step 3: Run monitoring across all batches
python monitor.py

# Step 4: Generate drift timeline chart
python visualize.py

# Step 5: Launch Streamlit dashboard
streamlit run app/dashboard.py

# Step 6: View MLflow experiment tracking
mlflow ui --backend-store-uri mlflow_runs
```

---

## PSI Threshold Reference

| PSI Range        | Severity    | Action                                        |
|-----------------|-------------|-----------------------------------------------|
| PSI < 0.10       | Stable      | No action required                            |
| 0.10 – 0.20      | Warning     | Investigate; monitor more frequently          |
| 0.20 – 0.25      | Significant | Escalate; begin root cause investigation      |
| PSI >= 0.25      | Critical    | Initiate retraining; notify stakeholders      |
