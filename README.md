# Automated Model Monitoring & Drift Detection

**MLOps capstone project** — A complete production monitoring system for a deployed ML model.
Detects data drift using PSI + KS test, tracks model performance over 12 simulated weeks,
and fires configurable alerts via an Airflow-orchestrated pipeline.

**Stack:** Python · Evidently AI · MLflow · Apache Airflow · Streamlit · XGBoost

---

## What This Project Demonstrates

- Data drift detection with **Population Stability Index (PSI)** and KS test via Evidently AI
- **PSI as a leading indicator on imbalanced classification** — drift detected at Week 5 while
  macro-F1 stays silent across all 12 weeks (the central finding of this project)
- Full MLflow experiment tracking across training and 12 weekly monitoring runs
- Airflow DAG operationalising monitoring into a scheduled, observable pipeline
- Live Streamlit dashboard with F1 trend, drift heatmap, and embedded Evidently reports
- Operational runbook translating ML alerts into actionable investigation procedures

---

## Project Structure

```
model_monitoring/
├── train.py                    # Train XGBoost baseline, log to MLflow, save reference data
├── simulate_production.py      # Generate 12 weekly batches with controlled drift injection
├── monitor.py                  # Evidently drift reports, PSI alerting, MLflow logging
├── alert_manager.py            # Threshold logic and alert dispatch
├── visualize.py                # F1 vs PSI drift timeline chart
├── dags/
│   └── monitoring_dag.py       # Airflow DAG: ingest → Evidently → alert → log
├── app/
│   └── dashboard.py            # Streamlit monitoring dashboard
├── data/
│   ├── creditcard.csv          # Source data (Kaggle Credit Card Fraud)
│   ├── reference_data.csv      # Training distribution saved by train.py
│   └── production_batches/     # 12 weekly batch files
├── reports/
│   └── evidently/              # One HTML report per batch
├── outputs/
│   ├── monitoring_results.csv  # Aggregated weekly metrics for the dashboard
│   ├── drift_timeline.png      # Key portfolio chart
│   └── alert_log.jsonl         # Machine-readable alert history
├── mlflow_runs/                # MLflow tracking store (auto-created)
├── runbook.md                  # Operational runbook
└── requirements.txt
```

---

## Drift Simulation Design

| Weeks | Drift Magnitude | Expected Behaviour |
|-------|----------------|--------------------|
| 1–4   | 0.0 std devs   | Stable — no drift, monitoring silent |
| 5–8   | 0.15–0.60 std devs | Gradual drift — PSI **WARNING** at Week 5, **CRITICAL** at Week 6 |
| 9–12  | 0.85–1.60 std devs | Severe drift — Max PSI reaches 2.96 by Week 12 |

Features drifted: `V14`, `V10`, `V4`, `Amount_log` — known high-importance fraud signals.

---

## Key Result

Across 12 simulated production weeks with controlled drift on four features, the monitoring
pipeline **detected the shift at Week 5 — before any meaningful change appeared in model
performance metrics** — and stayed silent across the first four stable weeks.

| Week | F1 | PR-AUC | Max PSI | Status |
|------|------|--------|---------|--------|
| 1 | 1.000 | 1.000 | 0.004 | OK |
| 2 | 1.000 | 1.000 | 0.003 | OK |
| 3 | 0.967 | 1.000 | 0.004 | OK |
| 4 | 0.981 | 0.995 | 0.005 | OK |
| **5** | 0.937 | 1.000 | **0.106** | **WARNING** |
| **6** | 0.921 | 0.882 | **0.201** | **CRITICAL** |
| 7 | 0.955 | 0.917 | 0.410 | CRITICAL |
| 8 | 1.000 | 1.000 | 0.651 | CRITICAL |
| 9 | 0.912 | 0.918 | 1.154 | CRITICAL |
| 10 | 0.935 | 0.918 | 1.687 | CRITICAL |
| 11 | 0.961 | 0.936 | 2.218 | CRITICAL |
| 12 | 0.929 | 0.938 | **2.958** | CRITICAL |

- **Max PSI rose from 0.004 to 2.96 — an 800× increase across the run**
- WARNING fired at Week 5 (PSI = 0.106, threshold 0.10)
- CRITICAL fired at Week 6 (PSI = 0.201, threshold 0.20) — one week after WARNING
- Macro-F1 stayed in the **0.91–1.00 band the entire run**, never crossing the F1 alert threshold of 0.80

![Drift Timeline](outputs/drift_timeline.png)

### Why F1 didn't move — and why that's the interesting finding

The Credit Card Fraud dataset is heavily imbalanced (fraud ≈ 0.17% of rows). On imbalanced
classification, macro-F1 is dominated by the majority class staying classifiable, so
input-feature drift on this dataset does **not** translate cleanly into a degraded F1 score
even when PSI is unambiguously screaming.

This is precisely the failure mode that input-distribution monitoring exists to catch. If
this model were monitored on F1 alone, the team would have **zero actionable signal across
12 weeks** despite four features drifting by more than 1.5 standard deviations. PSI provides
that signal at Week 5 — long before F1, PR-AUC, or any downstream business KPI would reveal
the same shift.

**Takeaway for production ML on imbalanced classification:** performance metrics are lagging,
low-resolution, and often silent. Input-distribution monitoring is not a nice-to-have — on
rare-event problems it is the only reliable early-warning signal.

---

## Quick Start

```bash
pip install -r requirements.txt

# Place creditcard.csv in data/ (download from Kaggle)
python train.py
python simulate_production.py
python monitor.py
python visualize.py

# Dashboard
streamlit run app/dashboard.py

# MLflow UI
mlflow ui --backend-store-uri mlflow_runs

# Airflow (Linux / WSL / Docker only — see note below)
airflow db init
cp dags/monitoring_dag.py ~/airflow/dags/
airflow standalone
```

---

## A note on Airflow

`dags/monitoring_dag.py` is authored against Airflow 2.x and is the orchestration layer for
the weekly monitoring cycle. It runs on Linux, in WSL, or in Docker.

**It does not run on native Windows.** Airflow depends on Unix-only modules (`pwd`, `grp`,
`fcntl`), and the Airflow maintainers do not support a Windows port — this is documented
upstream.

For local development on Windows, this project runs without Airflow: `python monitor.py`
produces the same outputs that the DAG would produce on a schedule. The DAG file is included
as a deliverable showing how the monitoring loop is operationalised; running it end-to-end
requires a Linux environment.

---

## PSI Thresholds

| PSI       | Severity | Action |
|-----------|----------|--------|
| < 0.10    | Stable   | No action — continue monitoring on schedule |
| 0.10–0.20 | Warning  | Monitor closely, investigate, no retraining yet |
| ≥ 0.20    | Critical | Escalate, investigate root cause, prepare retraining |

Thresholds follow standard financial risk modelling conventions where PSI < 0.10 is considered
stable and 0.20+ is where empirical studies show model performance consistently degrades.

---

## Production Gaps

This project demonstrates the monitoring loop end-to-end. To run in production, the following
would need to be added or replaced:

- **Real data ingestion** — `monitor.py` currently reads CSVs from `data/production_batches/`.
  Production would pull the last 7 days of transactions from a warehouse (Snowflake/BigQuery)
  or a streaming sink (Kafka/Kinesis).
- **Alert delivery** — alerts log to `outputs/alert_log.jsonl` and stdout. Production needs
  a Slack webhook, PagerDuty integration, or email transport.
- **Dashboard authentication** — Streamlit is open. Real deployment needs SSO or at minimum
  a shared-password layer.
- **Multi-model support** — the pipeline is hardcoded to one model and one reference dataset.
  A model registry would let the same DAG monitor N models with per-model thresholds.
- **Reference dataset refresh** — the runbook documents when to refresh the reference, but
  the pipeline doesn't automate it. Production would version references in S3/GCS and promote
  new ones through a PR.
- **Tests and CI** — no unit tests on `compute_psi`, threshold logic, or the alert manager.
  Production would have pytest coverage and a CI gate that parses `dags/monitoring_dag.py`.
- **Backfill** — the DAG runs forward only. Recovering from an outage would need a backfill
  command for missed weeks.

These are deliberately out of scope for a portfolio project — the goal is to demonstrate
the monitoring loop, not ship a managed service.

---

## Resume Bullets

- Built an automated ML monitoring system using Evidently AI to detect data drift (KS test +
  PSI) across 12 simulated weekly production batches on the Credit Card Fraud dataset (0.17%
  positive class), with configurable alerting at PSI > 0.10 (warning) and PSI > 0.20 (critical),
  scheduled via an Apache Airflow DAG.

- Demonstrated that input-distribution monitoring is the only reliable early-warning signal
  on heavily imbalanced classification: PSI detected drift at Week 5 and rose 800× across
  the run (0.004 → 2.96), while macro-F1 stayed in the 0.91–1.00 band the entire 12 weeks
  and never crossed performance alert thresholds — quantifying why production ML on
  rare-event problems cannot rely on performance metrics alone.

- Delivered a Streamlit monitoring dashboard with F1 trend, feature drift heatmap, and
  embedded Evidently HTML reports, alongside an operational runbook translating MLOps alerts
  into actionable investigation and retraining criteria for non-ML stakeholders.