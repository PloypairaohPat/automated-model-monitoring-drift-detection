import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data/production_batches", exist_ok=True)

df = pd.read_csv("data/creditcard.csv")
df["Amount_log"] = np.log1p(df["Amount"])
df["Hour"] = (df["Time"] % 86400) / 3600
feature_cols = [c for c in df.columns if c not in ["Class", "Time", "Amount"]]

X = df[feature_cols].values
y = df["Class"].values
feature_names = feature_cols

BATCH_SIZE = 5000
N_WEEKS = 12

# Features chosen because they are important fraud signals
DRIFT_FEATURES = ["V14", "V10", "V4", "Amount_log"]
drift_feature_indices = [feature_names.index(f) for f in DRIFT_FEATURES]

print("Generating 12 weekly production batches...\n")

for week in range(1, N_WEEKS + 1):
    idx = np.random.choice(len(X), BATCH_SIZE, replace=False)
    X_batch = X[idx].copy().astype(float)
    y_batch = y[idx].copy()

    # Drift schedule:
    # Weeks 1-4:  no drift
    # Weeks 5-8:  gradual shift — mean shifts by 0.15 * (week - 4) std devs
    # Weeks 9-12: large shift — mean shifts by 0.60 + 0.25 * (week - 8) std devs
    if week <= 4:
        drift_magnitude = 0.0
    elif week <= 8:
        drift_magnitude = 0.15 * (week - 4)  # 0.15, 0.30, 0.45, 0.60
    else:
        drift_magnitude = 0.60 + 0.25 * (week - 8)  # 0.85, 1.10, 1.35, 1.60

    if drift_magnitude > 0:
        for feat_idx in drift_feature_indices:
            feat_std = X[:, feat_idx].std()
            X_batch[:, feat_idx] += drift_magnitude * feat_std
            noise = np.random.normal(0, drift_magnitude * feat_std * 0.3, BATCH_SIZE)
            X_batch[:, feat_idx] += noise

    batch_df = pd.DataFrame(X_batch, columns=feature_names)
    batch_df["target"] = y_batch
    batch_df["week"] = week
    batch_df.to_csv(f"data/production_batches/week_{week:02d}.csv", index=False)

    print(
        f"Week {week:2d}: drift_magnitude={drift_magnitude:.2f} | "
        f"fraud_rate={y_batch.mean():.4f} | "
        f"status={'NO DRIFT' if drift_magnitude == 0 else 'GRADUAL DRIFT' if drift_magnitude <= 0.60 else 'SEVERE DRIFT'}"
    )

print("\nDone. 12 batch files saved to data/production_batches/")
