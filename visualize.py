import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("outputs", exist_ok=True)

results = pd.read_csv("outputs/monitoring_results.csv")
weeks = results["week"].tolist()
f1_scores = results["f1"].tolist()
max_psi = results["max_psi"].tolist()

fig, ax1 = plt.subplots(figsize=(13, 6))

colour_f1 = "#1A3A6B"
ax1.plot(weeks, f1_scores, color=colour_f1, lw=2.5, marker="o", markersize=6, label="F1 Score")
ax1.set_xlabel("Production Week", fontsize=13)
ax1.set_ylabel("F1 Score", color=colour_f1, fontsize=13)
ax1.tick_params(axis="y", labelcolor=colour_f1)
ax1.set_ylim(0.40, 1.00)
ax1.axhline(0.80, color=colour_f1, lw=1, ls="--", alpha=0.5, label="F1 Alert Threshold (0.80)")

ax2 = ax1.twinx()
colour_psi = "#D4621A"
ax2.plot(
    weeks, max_psi, color=colour_psi, lw=2.5, marker="s",
    markersize=6, ls="--", label="Max PSI"
)
ax2.set_ylabel("Max PSI (across features)", color=colour_psi, fontsize=13)
ax2.tick_params(axis="y", labelcolor=colour_psi)
ax2.axhline(0.10, color="gold", lw=1, ls=":", alpha=0.8, label="PSI Warning (0.10)")
ax2.axhline(0.20, color=colour_psi, lw=1, ls=":", alpha=0.8, label="PSI Critical (0.20)")

ax1.axvspan(4.5, 8.5, alpha=0.08, color="orange", label="Gradual drift zone")
ax1.axvspan(8.5, 12.5, alpha=0.12, color="red", label="Severe drift zone")

# Find the first warning and critical weeks from actual data
warn_weeks = results[results["alert_status"] == "WARNING"]["week"].tolist()
crit_weeks = results[results["alert_status"] == "CRITICAL"]["week"].tolist()

if warn_weeks:
    w = warn_weeks[0]
    f1_at_w = results.loc[results["week"] == w, "f1"].values[0]
    ax1.annotate(
        f"PSI warning\nfires (Week {w})",
        xy=(w, f1_at_w),
        fontsize=9,
        color="darkorange",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="darkorange"),
        xytext=(w, f1_at_w - 0.12),
    )

if crit_weeks:
    c = crit_weeks[0]
    f1_at_c = results.loc[results["week"] == c, "f1"].values[0]
    ax1.annotate(
        f"PSI critical\nfires (Week {c})",
        xy=(c, f1_at_c),
        fontsize=9,
        color="red",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="red"),
        xytext=(c, f1_at_c - 0.14),
    )

ax1.set_title(
    "Model Monitoring: F1 Score Degradation vs Feature Drift (12-Week Simulation)",
    fontsize=13,
    fontweight="bold",
)
ax1.set_xticks(weeks)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/drift_timeline.png", dpi=150, bbox_inches="tight")
print("Chart saved to outputs/drift_timeline.png")
plt.show()
