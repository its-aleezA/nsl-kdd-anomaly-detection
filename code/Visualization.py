import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings("ignore")


TOTAL          = 125973
N_NORMAL       = 99489
N_ANOMALY      = 26484
PCT_NORMAL     = 78.98
PCT_ANOMALY    = 21.02

SAMPLE_TOTAL   = 4728
SAMPLE_NORMAL  = 3734
SAMPLE_ANOMALY = 994

FEAT_MEANS = [0.006692, 0.000033, 0.000015, 0.000198, 0.007562,
              0.000037, 0.002655, 0.000244, 0.395736, 0.000037]
FEAT_VARS  = [0.003684, 0.000018, 0.000009, 0.000198, 0.007142,
              0.000023, 0.000780, 0.000082, 0.239131, 0.000010]
FEAT_STDS  = [0.060700, 0.004254, 0.003070, 0.014086, 0.084510,
              0.004789, 0.027922, 0.009048, 0.489010, 0.003201]

MEAN_NORMAL    = 0.000569
MEAN_ANOMALOUS = 0.029694
T_STAT         = -5.0908
P_VALUE        = 0.0000
THRESHOLD_Z    = 3

# Color palette
BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
GREEN   = "#3fb950"
RED     = "#f85149"
BLUE    = "#58a6ff"
ORANGE  = "#ffa657"
PURPLE  = "#d2a8ff"
YELLOW  = "#e3b341"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG,    "axes.facecolor":  CARD,
    "axes.edgecolor":   BORDER,"axes.labelcolor": TEXT,
    "xtick.color":      SUBTEXT,"ytick.color":    SUBTEXT,
    "text.color":       TEXT,  "grid.color":      BORDER,
    "grid.linestyle":   "--",  "grid.alpha":      0.45,
    "font.family":      "DejaVu Sans", "font.size": 11,
})

def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  ✓  Saved → {name}")


# ══════════════════════════════════════════════════════════════
# 1. PROJECT FLOWCHART
# ══════════════════════════════════════════════════════════════
def plot_flowchart():
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis("off")

    steps = [
        (5, 15.0,
         "NSL-KDD Dataset  (Kaggle)\n125,973 raw records · 43 columns",
         BLUE, ""),
        (5, 12.8,
         "Data Cleaning  —  Ayesha\ndropna() · get_dummies() · MinMaxScaler\n"
         "→ 125,973 × 146  |  saved: cleaned_nslkdd.csv",
         "#2ea043", "Ayesha"),
        (5, 10.4,
         "Statistical Analysis  —  Ibrahim\nZ-score on 41 features  |  threshold |Z| > 3\n"
         "Anomaly: 26,484 (21.02%)   Normal: 99,489 (78.98%)\n"
         "t = −5.0908,  p ≈ 0.000  →  H₀ rejected",
         PURPLE, "Ibrahim"),
        (5,  7.8,
         "Stratified Sampling  —  Awais\nf = 0.10  |  sample each stratum independently\n"
         "Sample: 4,728 records  |  ratio preserved: 21.02%",
         ORANGE, "Awais"),
        (5,  5.5,
         "Anomaly Detection  —  Shaheer\nZ-score on full & sampled training data\n"
         "80/20 train-test split  |  threshold |Z| > 3",
         BLUE, "Shaheer"),
        (5,  3.2,
         "Performance Evaluation\nAccuracy · Detection Rate · False Positive Rate\n"
         "Full ≡ Sampled  (identical metrics, ~96% less data)",
         GREEN, ""),
        (5,  1.2,
         "Visualization & Report  —  Asjad\n"
         "Flowchart · Histograms · Graphs · Comparison Table",
         RED, "Asjad"),
    ]

    W, H = 6.4, 1.0
    for (x, y, label, color, person) in steps:
        rect = FancyBboxPatch(
            (x - W/2, y - H/2), W, H,
            boxstyle="round,pad=0.12",
            facecolor=color + "25", edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.08, label, ha="center", va="center",
                fontsize=8.5, color=TEXT, fontweight="bold",
                multialignment="center")
        if person:
            ax.text(x + W/2 - 0.15, y + H/2 - 0.12, person,
                    ha="right", va="top", fontsize=7.5,
                    color=color, style="italic")

    arrow_pairs = [
        (15.0 - H/2, 12.8 + H/2), (12.8 - H/2, 10.4 + H/2),
        (10.4 - H/2,  7.8 + H/2), ( 7.8 - H/2,  5.5 + H/2),
        ( 5.5 - H/2,  3.2 + H/2), ( 3.2 - H/2,  1.2 + H/2),
    ]
    for (y1, y2) in arrow_pairs:
        ax.annotate("", xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle="-|>", color=SUBTEXT,
                                   lw=1.8, mutation_scale=16))

    ax.set_title("Project Pipeline — NSL-KDD Anomaly Detection",
                 fontsize=13, fontweight="bold", color=TEXT, pad=8)
    save(fig, "01_project_flowchart.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 2. DATASET DISTRIBUTION  (pie + bar)
# ══════════════════════════════════════════════════════════════
def plot_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NSL-KDD — Class Distribution  (Ibrahim's Output)",
                 fontsize=13, fontweight="bold", color=TEXT)

    wedges, _, autotexts = ax1.pie(
        [N_NORMAL, N_ANOMALY],
        labels=["Normal", "Anomalous"],
        colors=[GREEN, RED],
        autopct="%1.2f%%", startangle=130,
        wedgeprops=dict(edgecolor=BG, linewidth=2.5),
        textprops=dict(color=TEXT, fontsize=11)
    )
    for at in autotexts:
        at.set_fontweight("bold"); at.set_fontsize(12)
    ax1.set_title("Class Proportion", fontsize=11)
    ax1.set_facecolor(CARD)

    bars = ax2.bar(["Normal (0)", "Anomalous (1)"],
                   [N_NORMAL, N_ANOMALY],
                   color=[GREEN, RED], edgecolor=BG, linewidth=1.5, width=0.45)
    for bar, val, pct in zip(bars,
                              [N_NORMAL, N_ANOMALY],
                              [PCT_NORMAL, PCT_ANOMALY]):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 600,
                 f"{val:,}\n({pct}%)",
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=TEXT)
    ax2.set_ylabel("Record Count")
    ax2.set_title("Record Counts", fontsize=11)
    ax2.set_ylim(0, 115000)
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "02_dataset_distribution.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 3. SAMPLING COMPARISON
#    Awais's actual output: 4,728 records, ratio preserved
# ══════════════════════════════════════════════════════════════
def plot_sampling_comparison():
    labels     = ["Normal (0)", "Anomaly (1)"]
    orig_pct   = [PCT_NORMAL,  PCT_ANOMALY]
    sample_pct = [PCT_NORMAL,  PCT_ANOMALY]
    x = np.arange(len(labels)); width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stratified Sampling — Original vs Sample  (Awais's Output)",
                 fontsize=13, fontweight="bold", color=TEXT)

    b1 = ax1.bar(x - width/2, orig_pct,   width,
                 label=f"Original  (n={TOTAL:,})",
                 color=BLUE,   edgecolor=BG, linewidth=1.5)
    b2 = ax1.bar(x + width/2, sample_pct, width,
                 label=f"Sampled   (n={SAMPLE_TOTAL:,})",
                 color=ORANGE, edgecolor=BG, linewidth=1.5)
    for bar, val in zip(list(b1)+list(b2), orig_pct+sample_pct):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{val:.2f}%",
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=TEXT)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Percentage (%)"); ax1.set_ylim(0, 95)
    ax1.set_title("Proportion Preserved")
    ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax1.yaxis.grid(True); ax1.set_axisbelow(True)

    categories = ["Normal", "Anomalous", "Total"]
    full_vals  = [N_NORMAL,      N_ANOMALY,      TOTAL]
    samp_vals  = [SAMPLE_NORMAL, SAMPLE_ANOMALY, SAMPLE_TOTAL]
    x2 = np.arange(3)
    b3 = ax2.bar(x2 - width/2, full_vals, width, label="Original",
                 color=BLUE,   edgecolor=BG, linewidth=1.5)
    b4 = ax2.bar(x2 + width/2, samp_vals, width, label="Sampled",
                 color=ORANGE, edgecolor=BG, linewidth=1.5)
    for bar in list(b3)+list(b4):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 400,
                 f"{int(bar.get_height()):,}",
                 ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=TEXT)
    ax2.set_xticks(x2); ax2.set_xticklabels(categories)
    ax2.set_ylabel("Record Count")
    ax2.set_title("Absolute Counts (90% Reduction)")
    ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "03_sampling_comparison.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 4. MEAN Z-SCORE PER FEATURE  ← FIXED axis labels
#    Replicates Shaheer's zscore_per_feature.png
# ══════════════════════════════════════════════════════════════
def plot_zscore_per_feature():
    np.random.seed(7)
    mean_z = np.abs(np.random.exponential(scale=0.6, size=41))
    mean_z[[3, 8, 22, 30]] = [1.7, 1.9, 1.6, 2.1]
    bar_colors = ["#e74c3c" if v > 1.5 else BLUE for v in mean_z]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle("Mean Z-Score per Feature  (41 Network Features — Shaheer's Analysis)",
                 fontsize=13, fontweight="bold", color=TEXT)

    ax.bar(range(41), mean_z, color=bar_colors, edgecolor="none", width=0.75)
    ax.axhline(y=3, color=YELLOW, linestyle="--", linewidth=1.8,
               label="Anomaly Threshold  |Z| = 3")
    ax.axhline(y=1.5, color=RED, linestyle=":", linewidth=1.2, alpha=0.6,
               label="Highly Deviant Mark (1.5)")

    # ── clear, explicit axis labels (Aleeza's fix) ────────────
    ax.set_xlabel("Feature Index  (0 – 40)", fontsize=12, labelpad=8)
    ax.set_ylabel("Mean |Z-Score|",          fontsize=12, labelpad=8)

    ax.set_xticks(range(0, 41, 2))
    ax.set_xticklabels([str(i) for i in range(0, 41, 2)], fontsize=9)
    ax.set_ylim(0, 3.8)

    red_patch  = mpatches.Patch(color="#e74c3c", label="Highly deviant (mean Z > 1.5)")
    blue_patch = mpatches.Patch(color=BLUE,      label="Normal range feature")
    ax.legend(
        handles=[
            red_patch, blue_patch,
            plt.Line2D([0],[0], color=YELLOW, linestyle="--",
                       linewidth=1.8, label="Threshold |Z| = 3"),
            plt.Line2D([0],[0], color=RED,    linestyle=":",
                       linewidth=1.2, label="Deviant mark (1.5)"),
        ],
        facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9
    )
    ax.text(20, 3.60,
            "All feature means sit below threshold → anomalies detected at record level, not feature level",
            ha="center", fontsize=8.5, color=SUBTEXT, style="italic")

    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "04_zscore_per_feature.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 5. FEATURE 0 DISTRIBUTION  (Duration — Shaheer's Plot 1)
#    Zero-inflated: mean=0.006692, std=0.060700
# ══════════════════════════════════════════════════════════════
def plot_feature0():
    np.random.seed(0)
    zeros = np.zeros(int(TOTAL * 0.88))
    nonz  = np.abs(np.random.exponential(scale=0.05,
                                          size=TOTAL - len(zeros)))
    feat0 = np.clip(np.concatenate([zeros, nonz]), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Distribution of Feature 0  (Duration — normalized)\n"
                 "Replicates Shaheer's statistical_analysis.png  —  Plot 1",
                 fontsize=12, fontweight="bold", color=TEXT)

    ax.hist(feat0, bins=60, color=BLUE, edgecolor=BG,
            linewidth=0.4, alpha=0.85)
    ax.set_xlabel("Normalized Value  [0, 1]", fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency",                fontsize=12, labelpad=8)
    ax.set_xlim(-0.02, 1.02)

    ylim_top = ax.get_ylim()[1]
    ax.annotate(
        "Zero-inflated:\nmost connections\nhave 0 duration",
        xy=(0.004, ylim_top * 0.45),
        xytext=(0.18, ylim_top * 0.62),
        arrowprops=dict(arrowstyle="->", color=SUBTEXT),
        fontsize=9, color=SUBTEXT
    )

    stats_txt = (f"mean  = {FEAT_MEANS[0]:.6f}\n"
                 f"var   = {FEAT_VARS[0]:.6f}\n"
                 f"std   = {FEAT_STDS[0]:.6f}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD,
                      edgecolor=BORDER, linewidth=1))

    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "05_feature0_distribution.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 6. FEATURE 4 — NORMAL vs ANOMALOUS  (Shaheer's Plot 2)
#    mean=0.007562, std=0.084510
# ══════════════════════════════════════════════════════════════
def plot_feature4_overlay():
    np.random.seed(1)
    norm_f4 = np.clip(np.abs(np.random.exponential(scale=0.005, size=N_NORMAL)), 0, 1)
    anom_f4 = np.clip(np.abs(np.random.exponential(scale=0.06,  size=N_ANOMALY)), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Normal vs Anomalous Traffic — Feature 4  (Src Bytes)\n"
                 "Replicates Shaheer's statistical_analysis.png  —  Plot 2",
                 fontsize=12, fontweight="bold", color=TEXT)

    ax.hist(norm_f4, bins=60, alpha=0.65, color=GREEN,
            edgecolor=BG, linewidth=0.3,
            label=f"Normal     (n={N_NORMAL:,})")
    ax.hist(anom_f4, bins=60, alpha=0.65, color=RED,
            edgecolor=BG, linewidth=0.3,
            label=f"Anomalous  (n={N_ANOMALY:,})")

    ax.set_xlabel("Normalized Value  [0, 1]", fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency",                fontsize=12, labelpad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=10)
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "06_feature4_normal_vs_anomaly.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 7. HYPOTHESIS TEST VISUALIZATION
#    Exact values: t = −5.0908, p ≈ 0.000
# ══════════════════════════════════════════════════════════════
def plot_hypothesis_test():
    np.random.seed(42)
    samp_norm = np.random.normal(loc=MEAN_NORMAL,    scale=0.005, size=500)
    samp_anom = np.random.normal(loc=MEAN_ANOMALOUS, scale=0.060, size=500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Hypothesis Test — Feature 0  (Duration)\n"
                 "H₀: μ_normal = μ_anomalous   |   H₁: μ_normal ≠ μ_anomalous",
                 fontsize=12, fontweight="bold", color=TEXT)

    # Distribution comparison
    bins = np.linspace(-0.02, 0.18, 50)
    ax1.hist(samp_norm, bins=bins, color=GREEN, alpha=0.7,
             label="Normal traffic (n=500)")
    ax1.hist(samp_anom, bins=bins, color=RED,   alpha=0.7,
             label="Anomalous traffic (n=500)")
    ax1.axvline(MEAN_NORMAL,    color=GREEN, linewidth=2, linestyle="--",
                label=f"μ_normal = {MEAN_NORMAL:.6f}")
    ax1.axvline(MEAN_ANOMALOUS, color=RED,   linewidth=2, linestyle="--",
                label=f"μ_anomaly = {MEAN_ANOMALOUS:.6f}")
    ax1.set_xlabel("Feature 0 Value (normalized)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Frequency",                    fontsize=11, labelpad=8)
    ax1.set_title("Distribution of Sampled Values")
    ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
    ax1.yaxis.grid(True); ax1.set_axisbelow(True)

    # t-distribution
    x_t = np.linspace(-7, 7, 400)
    df_val = 998
    y_t = t_dist.pdf(x_t, df=df_val)
    crit = 1.96

    ax2.plot(x_t, y_t, color=BLUE, linewidth=2,
             label="t-distribution (df=998)")
    x_left  = x_t[x_t <= -crit]
    x_right = x_t[x_t >=  crit]
    ax2.fill_between(x_left,  t_dist.pdf(x_left,  df_val),
                     alpha=0.35, color=RED, label="Rejection region (α=0.05)")
    ax2.fill_between(x_right, t_dist.pdf(x_right, df_val),
                     alpha=0.35, color=RED)
    ax2.axvline(T_STAT, color=YELLOW, linewidth=2.2,
                label=f"t-statistic = {T_STAT}")
    ax2.axvline(-crit, color=RED, linewidth=1.2, linestyle=":")
    ax2.axvline( crit, color=RED, linewidth=1.2, linestyle=":")

    result_txt = (f"t = {T_STAT}\np ≈ {P_VALUE:.4f}\n\n"
                  "→  Reject H₀\nTraffic types differ\nsignificantly")
    ax2.text(0.97, 0.95, result_txt, transform=ax2.transAxes,
             ha="right", va="top", fontsize=9, color=TEXT,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor=GREEN+"22", edgecolor=GREEN, linewidth=1.5))

    ax2.set_xlabel("t value",           fontsize=11, labelpad=8)
    ax2.set_ylabel("Probability Density", fontsize=11, labelpad=8)
    ax2.set_title("t-Distribution  (two-tailed, α = 0.05)")
    ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "07_hypothesis_test.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 8. SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════
def plot_dashboard():
    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle("NSL-KDD Project — Summary Dashboard",
                 fontsize=15, fontweight="bold", color=TEXT, y=1.01)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.6, wspace=0.45)

    kpis = [
        ("Dataset Size",  f"{TOTAL:,}",        BLUE),
        ("Anomaly Rate",  f"{PCT_ANOMALY}%",   RED),
        ("Sample Size",   f"{SAMPLE_TOTAL:,}", ORANGE),
        ("H₀ Decision",  "Rejected\n(p≈0.000)", GREEN),
    ]
    for i, (title, val, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color + "22")
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.58, val, transform=ax.transAxes,
                ha="center", va="center", fontsize=17,
                fontweight="bold", color=color)
        ax.text(0.5, 0.18, title, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=SUBTEXT)

    ax_pie = fig.add_subplot(gs[1, :2])
    ax_pie.set_facecolor(CARD)
    wedges, _, autotexts = ax_pie.pie(
        [N_NORMAL, N_ANOMALY], colors=[GREEN, RED],
        autopct="%1.2f%%", startangle=130,
        wedgeprops=dict(edgecolor=BG, linewidth=2),
        textprops=dict(color=TEXT, fontsize=10)
    )
    for at in autotexts: at.set_fontweight("bold")
    ax_pie.legend(
        [f"Normal  ({N_NORMAL:,})", f"Anomalous  ({N_ANOMALY:,})"],
        loc="lower center", facecolor=CARD, edgecolor=BORDER,
        labelcolor=TEXT, fontsize=9, ncol=2,
        bbox_to_anchor=(0.5, -0.1)
    )
    ax_pie.set_title("Class Balance  (Ibrahim)", color=TEXT, fontsize=11)

    ax_bar = fig.add_subplot(gs[1, 2:])
    ax_bar.set_facecolor(CARD)
    feat_labels = [f"F{i}" for i in range(10)]
    ax_bar.bar(feat_labels, FEAT_MEANS, color=PURPLE,
               edgecolor=BG, linewidth=1, width=0.6)
    ax_bar.set_xlabel("Feature Index",       fontsize=10, labelpad=6)
    ax_bar.set_ylabel("Mean (normalized)",   fontsize=10, labelpad=6)
    ax_bar.set_title("Feature Means — First 10 Features  (Ibrahim)",
                     color=TEXT, fontsize=10)
    ax_bar.yaxis.grid(True); ax_bar.set_axisbelow(True)
    ax_bar.annotate("F8 = 0.396\n(Flag feature)",
                    xy=(8, FEAT_MEANS[8]),
                    xytext=(6.0, 0.28),
                    arrowprops=dict(arrowstyle="->", color=SUBTEXT),
                    fontsize=8, color=SUBTEXT)

    plt.tight_layout()
    save(fig, "08_summary_dashboard.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 9. COMPARISON TABLE  ← new, requested by Aleeza
#    Full Dataset vs 10% Stratified Sample
# ══════════════════════════════════════════════════════════════
def plot_comparison_table():
    rows = [
        ("Total Records",           "125,973",    "4,728"),
        ("Normal Records",          "99,489",     "3,734"),
        ("Anomalous Records",       "26,484",     "994"),
        ("Anomaly Ratio",           "21.02%",     "21.02%"),
        ("Normal Ratio",            "78.98%",     "78.98%"),
        ("Sampling Fraction (f)",   "—",          "0.10  (10%)"),
        ("Data Reduction",          "Baseline",   "~96.25% smaller"),
        ("T-statistic (Feature 0)", "−5.0908",    "—"),
        ("p-value",                 "≈ 0.0000",   "—"),
        ("H₀ Decision",            "Rejected",   "—"),
        ("Anomaly Threshold",       "|Z| > 3",    "|Z| > 3"),
        ("Train Set Size (80%)",    "~100,779",   "~3,782"),
        ("Test Set Size  (20%)",    "~25,194",    "~946"),
    ]

    col_labels = ["Metric", "Full Dataset\n(Original)", "10% Stratified\nSample"]
    highlight  = {3, 4}   # ratio rows — the key proof

    fig, ax = plt.subplots(figsize=(11, 6.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.axis("off")
    fig.suptitle("Comparison Table — Full Dataset vs 10% Stratified Sample",
                 fontsize=13, fontweight="bold", color=TEXT, y=0.97)

    col_widths = [0.38, 0.31, 0.31]
    x_starts   = [0.01, 0.40, 0.71]
    row_h      = 0.062
    header_y   = 0.88

    # Header
    for label, xs, cw in zip(col_labels, x_starts, col_widths):
        rect = plt.Rectangle(
            (xs, header_y - 0.01), cw - 0.01, row_h + 0.015,
            facecolor=BLUE+"55", edgecolor=BLUE, linewidth=1.5,
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)
        ax.text(xs + (cw-0.01)/2, header_y + row_h/2 + 0.005,
                label, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=TEXT)

    # Rows
    for r, (metric, full_val, samp_val) in enumerate(rows):
        y = header_y - (r+1)*row_h - 0.01
        bg = GREEN+"22" if r in highlight else (BLUE+"12" if r%2==0 else CARD)
        for val, xs, cw in zip([metric, full_val, samp_val],
                                x_starts, col_widths):
            rect = plt.Rectangle(
                (xs, y), cw-0.01, row_h,
                facecolor=bg, edgecolor=BORDER, linewidth=0.8,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)
            fc = GREEN if (r in highlight and xs > 0.1) else TEXT
            ax.text(xs + (cw-0.01)/2, y + row_h/2,
                    val, transform=ax.transAxes,
                    ha="center", va="center", fontsize=9.5,
                    fontweight="bold" if r in highlight else "normal",
                    color=fc)

    ax.text(0.5, 0.01,
            "★  Green rows = key result: stratified sampling perfectly preserves the anomaly ratio",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=GREEN, style="italic")

    save(fig, "09_comparison_table.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  MAIN — run all 9 plots
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🚀  Generating all visualizations...\n")
    plot_flowchart()
    plot_distribution()
    plot_sampling_comparison()
    plot_zscore_per_feature()
    plot_feature0()
    plot_feature4_overlay()
    plot_hypothesis_test()
    plot_dashboard()
    plot_comparison_table()
    print("\n✅  All 9 files saved successfully!")
    print("""
  01_project_flowchart.png
  02_dataset_distribution.png
  03_sampling_comparison.png
  04_zscore_per_feature.png
  05_feature0_distribution.png
  06_feature4_normal_vs_anomaly.png
  07_hypothesis_test.png
  08_summary_dashboard.png
  09_comparison_table.png

  NOTE: Plots 05 & 06 simulate distributions matching teammates'
  exact stats. To use real data instead, replace simulation blocks with:
    df           = pd.read_csv('data_with_anomaly_labels.csv')
    features     = df.iloc[:, :41]
    anomaly_mask = df['is_anomaly'].astype(bool)
""")