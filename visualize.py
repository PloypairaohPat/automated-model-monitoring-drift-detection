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

# ─── F1 line (left axis) ───────────────────────────────────────────────
colour_f1 = "#1A3A6B"
ax1.plot(weeks, f1_scores, color=colour_f1, lw=2.5, marker="o", markersize=6, label="F1 Score")
ax1.set_xlabel("Production Week", fontsize=13)
ax1.set_ylabel("F1 Score", color=colour_f1, fontsize=13)
ax1.tick_params(axis="y", labelcolor=colour_f1)
ax1.set_ylim(0.40, 1.00)
ax1.axhline(0.80, color=colour_f1, lw=1, ls="--", alpha=0.5, label="F1 Alert Threshold (0.80)")

# ─── PSI line (right axis) ─────────────────────────────────────────────
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

# ─── Drift zone shading ────────────────────────────────────────────────
ax1.axvspan(4.5, 8.5, alpha=0.08, color="orange", label="Gradual drift zone")
ax1.axvspan(8.5, 12.5, alpha=0.12, color="red", label="Severe drift zone")

# ─── Annotations: point at the PSI line (right axis), not the F1 line ──
# The story is "PSI detects, F1 doesn't react" — arrows must land on the
# orange line at the week the threshold is crossed, with the actual PSI
# value shown so the chart is self-documenting.
warn_weeks = results[results["alert_status"] == "WARNING"]["week"].tolist()
crit_weeks = results[results["alert_status"] == "CRITICAL"]["week"].tolist()

if warn_weeks:
    w = warn_weeks[0]
    psi_at_w = results.loc[results["week"] == w, "max_psi"].values[0]
    ax2.annotate(
        f"WARNING fires (Week {w})\nMax PSI = {psi_at_w:.3f}",
        xy=(w, psi_at_w),
        xytext=(2.5, 1.15),
        fontsize=9,
        color="darkorange",
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.3),
    )

if crit_weeks:
    c = crit_weeks[0]
    psi_at_c = results.loc[results["week"] == c, "max_psi"].values[0]
    ax2.annotate(
        f"CRITICAL fires (Week {c})\nMax PSI = {psi_at_c:.3f}",
        xy=(c, psi_at_c),
        xytext=(8, 1.65),
        fontsize=9,
        color="red",
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.3),
    )

# ─── Title: reframed to match the real result ──────────────────────────
ax1.set_title(
    "PSI Detects Drift While F1 Stays Silent — 12-Week Imbalanced Simulation",
    fontsize=13,
    fontweight="bold",
)
ax1.set_xticks(weeks)

# ─── Combined legend ───────────────────────────────────────────────────
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/drift_timeline.png", dpi=150, bbox_inches="tight")
print("Chart saved to outputs/drift_timeline.png")
plt.show()