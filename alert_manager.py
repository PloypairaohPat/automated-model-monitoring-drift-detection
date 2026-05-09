import json
import os
from datetime import datetime

PSI_WARNING = 0.10
PSI_CRITICAL = 0.20
F1_THRESHOLD = 0.80


def evaluate_alert(week, max_psi, f1_score, n_drifted_features):
    alerts = []
    os.makedirs("outputs", exist_ok=True)

    if max_psi >= PSI_CRITICAL:
        alerts.append({
            "severity": "CRITICAL",
            "type": "data_drift",
            "message": (
                f"Week {week}: Max PSI={max_psi:.3f} — severe drift detected in "
                f"{n_drifted_features} features. Immediate investigation required."
            ),
            "action": "Initiate retraining. Check data pipeline for upstream issues.",
            "timestamp": datetime.now().isoformat(),
        })
    elif max_psi >= PSI_WARNING:
        alerts.append({
            "severity": "WARNING",
            "type": "data_drift",
            "message": (
                f"Week {week}: Max PSI={max_psi:.3f} — minor drift in "
                f"{n_drifted_features} features."
            ),
            "action": "Monitor closely. Investigate root cause. No retraining yet.",
            "timestamp": datetime.now().isoformat(),
        })

    if f1_score < F1_THRESHOLD:
        alerts.append({
            "severity": "CRITICAL",
            "type": "performance_degradation",
            "message": f"Week {week}: F1={f1_score:.4f} — below threshold of {F1_THRESHOLD}.",
            "action": "Retraining required. Check if drift was the root cause.",
            "timestamp": datetime.now().isoformat(),
        })

    for alert in alerts:
        print(f"  [{alert['severity']}] {alert['message']}")
        print(f"    Action: {alert['action']}")
        with open("outputs/alert_log.jsonl", "a") as f:
            f.write(json.dumps(alert) + "\n")

    return alerts
