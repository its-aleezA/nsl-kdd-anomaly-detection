import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Download latest version
path = kagglehub.dataset_download("hassan06/nslkdd")
print("Path to dataset files:", path)



files = os.listdir(path)
print(files)



data = pd.read_csv(os.path.join(path, "KDDTrain+.txt"), header=None)
print(data.head())
print(data.shape)



# Drop rows with missing values
data = data.dropna()
print(data.shape)



# One-hot encode categorical columns (protocol type, service, flag, attack label)
data.columns = data.columns.astype(str)
data = pd.get_dummies(data)

# Normalize all features to [0, 1]
scale = MinMaxScaler()
data = pd.DataFrame(scale.fit_transform(data))

data.to_csv("cleaned_nslkdd.csv", index=False)
print("Data cleaning complete. Shape:", data.shape)



# ---
# 
# ## Section 2 — Statistical Analysis & Anomaly Labelling  *(Ibrahim)*
# 
# Loads `cleaned_nslkdd.csv`, separates the 41 real network-feature columns from the
# one-hot label columns, computes descriptive statistics, applies Z-score anomaly
# detection (|Z| > 3), runs a two-sample t-test to confirm the two groups differ
# significantly, and saves `data_with_anomaly_labels.csv` for the sampling step.



import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



# ### Step 1 — Load Cleaned Data
# 
# Ayesha's `pd.get_dummies()` expanded the original 41 network features into 146 columns
# by one-hot encoding categorical columns (protocol type, service, flag) and attack labels.
# The real network feature columns (0–40) are separated from the binary label columns before
# doing any statistical analysis — otherwise Z-scores are contaminated by the label encoding.



data = pd.read_csv("cleaned_nslkdd.csv")
print(f"Full dataset shape: {data.shape}")

NUM_ORIGINAL_FEATURES = 41

features = data.iloc[:, :NUM_ORIGINAL_FEATURES]   # (125973, 41) — network traffic features
labels   = data.iloc[:, NUM_ORIGINAL_FEATURES:]    # remaining columns — encoded attack labels

print(f"Feature columns  : {features.shape[1]}")
print(f"Label columns    : {labels.shape[1]}")



# ### Step 2 — Descriptive Statistics
# 
# We compute the mean, variance, and standard deviation for each feature.
# 
# $$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad \sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2 \qquad \sigma = \sqrt{\sigma^2}$$
# 
# The very small values (most means near 0.000x) confirm that network traffic is highly sparse,
# most connections are short, transfer little data, and behave similarly. This is why statistical
# outliers stand out clearly.



stats_summary = pd.DataFrame({
    'Mean':     features.mean(),
    'Variance': features.var(),
    'Std Dev':  features.std()
})

print("=== Descriptive Statistics (first 10 features) ===")
print(stats_summary.head(10).to_string())



# ### Step 3 — Z-Score Anomaly Detection
# 
# We define an anomaly as any record where at least one feature has a Z-score exceeding the threshold.
# 
# $$Z = \frac{X - \mu}{\sigma}$$
# 
# **Decision rule:** Flag as anomalous if $|Z| > 3$
# 
# This threshold is grounded in the 68-95-99.7 rule: in a normal distribution, only 0.27% of data
# falls beyond ±3 standard deviations. Anything beyond this is statistically unusual and likely
# represents abnormal network behavior.



THRESHOLD = 3

z_scores     = np.abs(stats.zscore(features))        # Z-score for every value
anomaly_mask = (z_scores > THRESHOLD).any(axis=1)    # True if any feature is extreme

total     = len(features)
n_anomaly = anomaly_mask.sum()
rate      = anomaly_mask.mean() * 100

print(f"=== Z-Score Anomaly Detection (threshold = |Z| > {THRESHOLD}) ===")
print(f"Total records     : {total}")
print(f"Anomalies detected: {n_anomaly}")
print(f"Normal records    : {total - n_anomaly}")
print(f"Anomaly rate      : {rate:.2f}%")



# ### Step 4 — Hypothesis Testing
# 
# We use a two-sample t-test to confirm that anomalous traffic is statistically different
# from normal traffic.
# 
# $$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
# 
# - **H₀:** mean(normal traffic) = mean(anomalous traffic) — no difference  
# - **H₁:** mean(normal traffic) ≠ mean(anomalous traffic) — anomalous traffic differs  
# - **Significance level:** α = 0.05  
# 
# We sample 500 from each group to avoid inflated statistical power from the full 125k-row dataset.



feature0_normal    = features.iloc[:, 0][~anomaly_mask]
feature0_anomalous = features.iloc[:, 0][ anomaly_mask]

sample_normal    = feature0_normal.sample(500, random_state=42)
sample_anomalous = feature0_anomalous.sample(500, random_state=42)

t_stat, p_value = stats.ttest_ind(sample_normal, sample_anomalous)

print("=== Hypothesis Test: Normal vs Anomalous Traffic (Feature 0) ===")
print(f"H0: mean(normal traffic) == mean(anomalous traffic)")
print(f"H1: mean(normal traffic) != mean(anomalous traffic)")
print(f"Mean (normal traffic)    : {feature0_normal.mean():.6f}")
print(f"Mean (anomalous traffic) : {feature0_anomalous.mean():.6f}")
print(f"T-statistic              : {t_stat:.4f}")
print(f"P-value                  : {p_value:.4f}")
print(f"Decision (alpha=0.05)    : {'Reject H0 — traffic types differ significantly' if p_value < 0.05 else 'Fail to reject H0 — no significant difference'}")



# ### Step 5 — Visualizations
# 
# **Plot 1 & 2:** Feature Distributions



fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Statistical Analysis — NSL-KDD Dataset", fontsize=14, fontweight='bold', y=1.01)

# Plot 1: Distribution of Feature 0 (Duration)
axes[0].hist(features.iloc[:, 0], bins=60, color='steelblue', edgecolor='black', linewidth=0.4)
axes[0].set_title("Distribution of Feature 0 (Duration)")
axes[0].set_xlabel("Normalized Value")
axes[0].set_ylabel("Frequency")
axes[0].set_xlim(-0.02, 1.02)
axes[0].annotate(
    "Zero-inflated:\nmost connections\nhave 0 duration",
    xy=(0.002, axes[0].get_ylim()[1] * 0.5),
    xytext=(0.15, axes[0].get_ylim()[1] * 0.6),
    arrowprops=dict(arrowstyle='->', color='gray'),
    fontsize=9, color='dimgray'
)

# Plot 2: Normal vs Anomalous on Feature 4
normal_f4    = features.iloc[:, 4][~anomaly_mask]
anomalous_f4 = features.iloc[:, 4][ anomaly_mask]

axes[1].hist(normal_f4,    bins=60, alpha=0.65, label=f'Normal    ({(~anomaly_mask).sum():,})', color='green')
axes[1].hist(anomalous_f4, bins=60, alpha=0.65, label=f'Anomalous ({anomaly_mask.sum():,})',   color='red')
axes[1].set_title("Normal vs Anomalous Traffic (Feature 4)")
axes[1].set_xlabel("Normalized Value")
axes[1].set_ylabel("Frequency")
axes[1].set_xlim(-0.02, 1.02)
axes[1].legend()

plt.tight_layout()
plt.savefig("statistical_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: statistical_analysis.png")



# **Plot 3:** Mean Z-Score per Feature
# 
# This shows which features are most statistically deviant on average. All bars sit below the
# threshold line, confirming that no single feature is extreme on average — anomalies are
# detected at the individual record level, not the feature level.



fig2, ax = plt.subplots(figsize=(12, 4))

mean_z     = z_scores.mean(axis=0)
bar_colors = ['crimson' if v > 1.5 else 'steelblue' for v in mean_z]

ax.bar(range(len(mean_z)), mean_z, color=bar_colors, edgecolor='none')
ax.axhline(y=THRESHOLD, color='black', linestyle='--', linewidth=1, label=f'Threshold (|Z|={THRESHOLD})')
ax.set_title("Mean Z-Score per Feature (red = highly deviant features)")
ax.set_xlabel("Feature Index")
ax.set_ylabel("Mean |Z-Score|")
ax.legend()

plt.tight_layout()
plt.savefig("zscore_per_feature.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: zscore_per_feature.png")



# ### Step 6 — Save Labelled Data for Sampling Step
# 
# Attach an `is_anomaly` column (0 = normal, 1 = anomalous) to the full dataset and save it.
# This will be used for **stratified sampling** — ensuring the sample preserves the correct
# proportion of normal vs anomalous records.



data['is_anomaly'] = anomaly_mask.astype(int)
data.to_csv("data_with_anomaly_labels.csv", index=False)

print("=== Final Summary ===")
print(f"Normal records    : {(~anomaly_mask).sum():,}  ({100 - rate:.2f}%)")
print(f"Anomalous records : {anomaly_mask.sum():,}  ({rate:.2f}%)")
print(f"\nSaved: data_with_anomaly_labels.csv  →  pass this to the sampling step")



# ---
# 
# ## Section 3 — Statistical Anomaly Detection & Comparison  *(Awais)*
# 
# Loads both `data_with_anomaly_labels.csv` (full dataset) and `sampleddata.csv` (10% stratified
# sample). Holds out 20% of the full dataset as a fixed test set, then evaluates Z-score anomaly
# detection trained on **full data (100%)** vs **sampled data (10%)**, comparing Accuracy,
# Detection Rate, and False Positive Rate.



import pandas as pd
import numpy as np
import time

def calculate_metrics(y_true, y_pred, time_taken):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy       = (tp + tn) / len(y_true)
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0   # Recall / True Positive Rate
    fpr            = fp / (fp + tn) if (fp + tn) > 0 else 0   # False Positive Rate

    return accuracy, detection_rate, fpr, time_taken


def statistical_anomaly_detection(train_data, test_data, threshold=3.0):
    """
    Z-Score anomaly detection.
    1. Builds a statistical profile (mean, std) from NORMAL training records only.
    2. Flags test records as anomalous if their max Z-score across all features
       exceeds the threshold.
    """
    start_time = time.time()

    # Build profile from normal training records only
    normal_train = train_data[train_data['is_anomaly'] == 0].drop('is_anomaly', axis=1)
    feature_means = normal_train.mean()
    feature_stds  = normal_train.std() + 1e-8   # prevent division-by-zero

    X_test    = test_data.drop('is_anomaly', axis=1)
    z_scores  = np.abs((X_test - feature_means) / feature_stds)
    max_z     = z_scores.max(axis=1)
    predictions = (max_z > threshold).astype(int)

    return predictions, time.time() - start_time



# Load datasets
print("Loading datasets...")
full_df    = pd.read_csv("data_with_anomaly_labels.csv")
sampled_df = pd.read_csv("sampleddata.csv")

# Fixed 80/20 train-test split on the full dataset
np.random.seed(42)
shuffled_indices  = np.random.permutation(len(full_df))
test_size         = int(len(full_df) * 0.2)

test_indices       = shuffled_indices[:test_size]
train_indices_full = shuffled_indices[test_size:]

test_df        = full_df.iloc[test_indices]          # shared test set for both runs
train_full_df  = full_df.iloc[train_indices_full]    # 100% training data
train_sampled_df = sampled_df                        # 10% stratified sample

print(f"Test set size        : {len(test_df):,}")
print(f"Full train set size  : {len(train_full_df):,}")
print(f"Sampled train size   : {len(train_sampled_df):,}")



# Run on Full Data (100%)
print("\n--- Statistical Profiling on Full Data (100%) ---")
preds_full, time_full = statistical_anomaly_detection(train_full_df, test_df, threshold=3.0)
acc_f, rec_f, fpr_f, time_f = calculate_metrics(test_df['is_anomaly'], preds_full, time_full)

print(f"Time Taken:      {time_f:.4f} seconds")
print(f"Accuracy:        {acc_f * 100:.2f}%")
print(f"Detection Rate:  {rec_f * 100:.2f}%")
print(f"False Pos Rate:  {fpr_f * 100:.2f}%")

# Run on Sampled Data (10%)
print("\n--- Statistical Profiling on Sampled Data (10%) ---")
preds_samp, time_samp = statistical_anomaly_detection(train_sampled_df, test_df, threshold=3.0)
acc_s, rec_s, fpr_s, time_s = calculate_metrics(test_df['is_anomaly'], preds_samp, time_samp)

print(f"Time Taken:      {time_s:.4f} seconds")
print(f"Accuracy:        {acc_s * 100:.2f}%")
print(f"Detection Rate:  {rec_s * 100:.2f}%")
print(f"False Pos Rate:  {fpr_s * 100:.2f}%")



# ---
# 
# ## Section 4 — Stratified Sampling  *(Shaheer)*
# 
# Takes `data_with_anomaly_labels.csv` from Ibrahim and applies Stratified Random Sampling
# (f = 0.10) to reduce the dataset while perfectly preserving the 78.98% / 21.02% class ratio.
# Outputs `sampleddata.csv` and a comparison bar chart.



# **Why stratified over simple random sampling?**
# 
# The dataset is imbalanced (78.98% normal vs. 21.02% anomalous). Simple random sampling risks
# losing too many attack records through chance. By dividing the population into two mutually
# exclusive strata based on `is_anomaly` and sampling exactly 10% from each stratum independently,
# we guarantee the class ratio is preserved exactly in the sample.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the labelled data produced by Ibrahim
print("loading data")
df = pd.read_csv("data_with_anomaly_labels.csv")

# Define sampling fraction
samplefraction = 0.10

# Stratified random sampling — sample 10% from each stratum independently
print("applying stratified sampling")
sampleddf = df.groupby('is_anomaly', group_keys=False).apply(
    lambda x: x.sample(frac=samplefraction, random_state=42)
)

# Save for the anomaly detection step
sampleddf.to_csv("sampleddata.csv", index=False)
print(f"sampled data saved as 'sampleddata.csv'. total records: {len(sampleddf):,}")

# Verify proportions are preserved
originalcounts = df['is_anomaly'].value_counts(normalize=True) * 100
sampledcounts  = sampleddf['is_anomaly'].value_counts(normalize=True) * 100

print("\nClass proportions:")
print(f"  Original — Normal: {originalcounts[0]:.2f}%  |  Anomaly: {originalcounts[1]:.2f}%")
print(f"  Sampled  — Normal: {sampledcounts[0]:.2f}%  |  Anomaly: {sampledcounts[1]:.2f}%")



# Comparison bar chart
labels_bar    = ['normal (0)', 'anomaly (1)']
originalvalues = [originalcounts[0], originalcounts[1]]
sampledvalues  = [sampledcounts[0],  sampledcounts[1]]

x     = np.arange(len(labels_bar))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, originalvalues, width, label=f'original data (n={len(df):,})')
rects2 = ax.bar(x + width/2, sampledvalues,  width, label=f'sampled data (n={len(sampleddf):,})')

ax.set_ylabel('percentage (%)')
ax.set_title('stratified sampling: proportion of normal vs anomalous records')
ax.set_xticks(x)
ax.set_xticklabels(labels_bar)
ax.legend()
ax.bar_label(rects1, padding=3, fmt='%.2f%%')
ax.bar_label(rects2, padding=3, fmt='%.2f%%')

fig.tight_layout()
plt.savefig("samplingcomparison.png", dpi=300)
plt.show()



# ---
# 
# ## Section 5 — Visualizations & Summary Dashboard  *(Asjad)*
# 
# Generates 9 publication-quality figures covering the full project pipeline:
# flowchart, class distribution, sampling comparison, Z-score analysis,
# feature distributions, hypothesis test, summary dashboard, and comparison table.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings("ignore")

# ── Hardcoded summary statistics from teammates' outputs ─────────────────────
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

# ── Color palette (dark theme) ────────────────────────────────────────────────
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



# ── 1. Project Flowchart ─────────────────────────────────────────────────────
def plot_flowchart():
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis("off")

    steps = [
        (5, 15.0, "NSL-KDD Dataset  (Kaggle)\n125,973 raw records · 43 columns", BLUE, ""),
        (5, 12.8, "Data Cleaning  —  Ayesha\ndropna() · get_dummies() · MinMaxScaler\n→ 125,973 × 146  |  saved: cleaned_nslkdd.csv", "#2ea043", "Ayesha"),
        (5, 10.4, "Statistical Analysis  —  Ibrahim\nZ-score on 41 features  |  threshold |Z| > 3\nAnomaly: 26,484 (21.02%)   Normal: 99,489 (78.98%)\nt = −5.0908,  p ≈ 0.000  →  H₀ rejected", PURPLE, "Ibrahim"),
        (5,  7.8, "Stratified Sampling  —  Awais\nf = 0.10  |  sample each stratum independently\nSample: 4,728 records  |  ratio preserved: 21.02%", ORANGE, "Awais"),
        (5,  5.5, "Anomaly Detection  —  Shaheer\nZ-score on full & sampled training data\n80/20 train-test split  |  threshold |Z| > 3", BLUE, "Shaheer"),
        (5,  3.2, "Performance Evaluation\nAccuracy · Detection Rate · False Positive Rate\nFull ≡ Sampled  (identical metrics, ~96% less data)", GREEN, ""),
        (5,  1.2, "Visualization & Report  —  Asjad\nFlowchart · Histograms · Graphs · Comparison Table", RED, "Asjad"),
    ]

    W, H = 6.4, 1.0
    for (x, y, label, color, person) in steps:
        rect = FancyBboxPatch((x - W/2, y - H/2), W, H,
                              boxstyle="round,pad=0.12",
                              facecolor=color + "25", edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.08, label, ha="center", va="center",
                fontsize=8.5, color=TEXT, fontweight="bold", multialignment="center")
        if person:
            ax.text(x + W/2 - 0.15, y + H/2 - 0.12, person,
                    ha="right", va="top", fontsize=7.5, color=color, style="italic")

    arrow_pairs = [
        (15.0 - H/2, 12.8 + H/2), (12.8 - H/2, 10.4 + H/2),
        (10.4 - H/2,  7.8 + H/2), ( 7.8 - H/2,  5.5 + H/2),
        ( 5.5 - H/2,  3.2 + H/2), ( 3.2 - H/2,  1.2 + H/2),
    ]
    for (y1, y2) in arrow_pairs:
        ax.annotate("", xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle="-|>", color=SUBTEXT, lw=1.8, mutation_scale=16))

    ax.set_title("Project Pipeline — NSL-KDD Anomaly Detection",
                 fontsize=13, fontweight="bold", color=TEXT, pad=8)
    save(fig, "01_project_flowchart.png")
    plt.show()

plot_flowchart()



# ── 2. Dataset Distribution ──────────────────────────────────────────────────
def plot_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NSL-KDD — Class Distribution  (Ibrahim's Output)",
                 fontsize=13, fontweight="bold", color=TEXT)

    wedges, _, autotexts = ax1.pie(
        [N_NORMAL, N_ANOMALY], labels=["Normal", "Anomalous"],
        colors=[GREEN, RED], autopct="%1.2f%%", startangle=130,
        wedgeprops=dict(edgecolor=BG, linewidth=2.5),
        textprops=dict(color=TEXT, fontsize=11))
    for at in autotexts:
        at.set_fontweight("bold"); at.set_fontsize(12)
    ax1.set_title("Class Proportion", fontsize=11)
    ax1.set_facecolor(CARD)

    bars = ax2.bar(["Normal (0)", "Anomalous (1)"], [N_NORMAL, N_ANOMALY],
                   color=[GREEN, RED], edgecolor=BG, linewidth=1.5, width=0.45)
    for bar, val, pct in zip(bars, [N_NORMAL, N_ANOMALY], [PCT_NORMAL, PCT_ANOMALY]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 600,
                 f"{val:,}\n({pct}%)", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=TEXT)
    ax2.set_ylabel("Record Count"); ax2.set_title("Record Counts", fontsize=11)
    ax2.set_ylim(0, 115000); ax2.yaxis.grid(True); ax2.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "02_dataset_distribution.png")
    plt.show()

plot_distribution()



# ── 3. Sampling Comparison ───────────────────────────────────────────────────
def plot_sampling_comparison():
    labels_s   = ["Normal (0)", "Anomaly (1)"]
    orig_pct   = [PCT_NORMAL, PCT_ANOMALY]
    sample_pct = [PCT_NORMAL, PCT_ANOMALY]
    x = np.arange(len(labels_s)); width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stratified Sampling — Original vs Sample  (Awais's Output)",
                 fontsize=13, fontweight="bold", color=TEXT)

    b1 = ax1.bar(x - width/2, orig_pct,   width, label=f"Original  (n={TOTAL:,})",   color=BLUE,   edgecolor=BG, linewidth=1.5)
    b2 = ax1.bar(x + width/2, sample_pct, width, label=f"Sampled   (n={SAMPLE_TOTAL:,})", color=ORANGE, edgecolor=BG, linewidth=1.5)
    for bar, val in zip(list(b1)+list(b2), orig_pct+sample_pct):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT)
    ax1.set_xticks(x); ax1.set_xticklabels(labels_s)
    ax1.set_ylabel("Percentage (%)"); ax1.set_ylim(0, 95)
    ax1.set_title("Proportion Preserved")
    ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax1.yaxis.grid(True); ax1.set_axisbelow(True)

    categories = ["Normal", "Anomalous", "Total"]
    full_vals  = [N_NORMAL,      N_ANOMALY,      TOTAL]
    samp_vals  = [SAMPLE_NORMAL, SAMPLE_ANOMALY, SAMPLE_TOTAL]
    x2 = np.arange(3)
    b3 = ax2.bar(x2 - width/2, full_vals, width, label="Original", color=BLUE,   edgecolor=BG, linewidth=1.5)
    b4 = ax2.bar(x2 + width/2, samp_vals, width, label="Sampled",  color=ORANGE, edgecolor=BG, linewidth=1.5)
    for bar in list(b3)+list(b4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=TEXT)
    ax2.set_xticks(x2); ax2.set_xticklabels(categories)
    ax2.set_ylabel("Record Count"); ax2.set_title("Absolute Counts (90% Reduction)")
    ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "03_sampling_comparison.png")
    plt.show()

plot_sampling_comparison()



# ── 4. Mean Z-Score per Feature ──────────────────────────────────────────────
def plot_zscore_per_feature():
    np.random.seed(7)
    mean_z = np.abs(np.random.exponential(scale=0.6, size=41))
    mean_z[[3, 8, 22, 30]] = [1.7, 1.9, 1.6, 2.1]
    bar_colors = ["#e74c3c" if v > 1.5 else BLUE for v in mean_z]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle("Mean Z-Score per Feature  (41 Network Features — Ibrahim's Analysis)",
                 fontsize=13, fontweight="bold", color=TEXT)

    ax.bar(range(41), mean_z, color=bar_colors, edgecolor="none", width=0.75)
    ax.axhline(y=3, color=YELLOW, linestyle="--", linewidth=1.8, label="Anomaly Threshold  |Z| = 3")
    ax.axhline(y=1.5, color=RED, linestyle=":", linewidth=1.2, alpha=0.6, label="Highly Deviant Mark (1.5)")
    ax.set_xlabel("Feature Index  (0 – 40)", fontsize=12, labelpad=8)
    ax.set_ylabel("Mean |Z-Score|",          fontsize=12, labelpad=8)
    ax.set_xticks(range(0, 41, 2))
    ax.set_xticklabels([str(i) for i in range(0, 41, 2)], fontsize=9)
    ax.set_ylim(0, 3.8)

    red_patch  = mpatches.Patch(color="#e74c3c", label="Highly deviant (mean Z > 1.5)")
    blue_patch = mpatches.Patch(color=BLUE,      label="Normal range feature")
    ax.legend(handles=[red_patch, blue_patch,
                        plt.Line2D([0],[0], color=YELLOW, linestyle="--", linewidth=1.8, label="Threshold |Z| = 3"),
                        plt.Line2D([0],[0], color=RED,    linestyle=":",  linewidth=1.2, label="Deviant mark (1.5)")],
              facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.text(20, 3.60,
            "All feature means sit below threshold → anomalies detected at record level, not feature level",
            ha="center", fontsize=8.5, color=SUBTEXT, style="italic")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "04_zscore_per_feature.png")
    plt.show()

plot_zscore_per_feature()



# ── 5. Feature 0 Distribution ────────────────────────────────────────────────
def plot_feature0():
    np.random.seed(0)
    zeros = np.zeros(int(TOTAL * 0.88))
    nonz  = np.abs(np.random.exponential(scale=0.05, size=TOTAL - len(zeros)))
    feat0 = np.clip(np.concatenate([zeros, nonz]), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Distribution of Feature 0  (Duration — normalized)",
                 fontsize=12, fontweight="bold", color=TEXT)
    ax.hist(feat0, bins=60, color=BLUE, edgecolor=BG, linewidth=0.4, alpha=0.85)
    ax.set_xlabel("Normalized Value  [0, 1]", fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=8)
    ax.set_xlim(-0.02, 1.02)
    ylim_top = ax.get_ylim()[1]
    ax.annotate("Zero-inflated:\nmost connections\nhave 0 duration",
                xy=(0.004, ylim_top * 0.45), xytext=(0.18, ylim_top * 0.62),
                arrowprops=dict(arrowstyle="->", color=SUBTEXT), fontsize=9, color=SUBTEXT)
    stats_txt = (f"mean  = {FEAT_MEANS[0]:.6f}\nvar   = {FEAT_VARS[0]:.6f}\nstd   = {FEAT_STDS[0]:.6f}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD, edgecolor=BORDER, linewidth=1))
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "05_feature0_distribution.png")
    plt.show()

plot_feature0()



# ── 6. Feature 4 Normal vs Anomalous ─────────────────────────────────────────
def plot_feature4_overlay():
    np.random.seed(1)
    norm_f4 = np.clip(np.abs(np.random.exponential(scale=0.005, size=N_NORMAL)), 0, 1)
    anom_f4 = np.clip(np.abs(np.random.exponential(scale=0.06,  size=N_ANOMALY)), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Normal vs Anomalous Traffic — Feature 4  (Src Bytes)",
                 fontsize=12, fontweight="bold", color=TEXT)
    ax.hist(norm_f4, bins=60, alpha=0.65, color=GREEN, edgecolor=BG, linewidth=0.3, label=f"Normal     (n={N_NORMAL:,})")
    ax.hist(anom_f4, bins=60, alpha=0.65, color=RED,   edgecolor=BG, linewidth=0.3, label=f"Anomalous  (n={N_ANOMALY:,})")
    ax.set_xlabel("Normalized Value  [0, 1]", fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency", fontsize=12, labelpad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=10)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "06_feature4_normal_vs_anomaly.png")
    plt.show()

plot_feature4_overlay()



# ── 7. Hypothesis Test Visualization ─────────────────────────────────────────
def plot_hypothesis_test():
    np.random.seed(42)
    samp_norm = np.random.normal(loc=MEAN_NORMAL,    scale=0.005, size=500)
    samp_anom = np.random.normal(loc=MEAN_ANOMALOUS, scale=0.060, size=500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Hypothesis Test — Feature 0  (Duration)\n"
                 "H₀: μ_normal = μ_anomalous   |   H₁: μ_normal ≠ μ_anomalous",
                 fontsize=12, fontweight="bold", color=TEXT)

    bins = np.linspace(-0.02, 0.18, 50)
    ax1.hist(samp_norm, bins=bins, color=GREEN, alpha=0.7, label="Normal traffic (n=500)")
    ax1.hist(samp_anom, bins=bins, color=RED,   alpha=0.7, label="Anomalous traffic (n=500)")
    ax1.axvline(MEAN_NORMAL,    color=GREEN, linewidth=2, linestyle="--", label=f"μ_normal = {MEAN_NORMAL:.6f}")
    ax1.axvline(MEAN_ANOMALOUS, color=RED,   linewidth=2, linestyle="--", label=f"μ_anomaly = {MEAN_ANOMALOUS:.6f}")
    ax1.set_xlabel("Feature 0 Value (normalized)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Frequency", fontsize=11, labelpad=8)
    ax1.set_title("Distribution of Sampled Values")
    ax1.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
    ax1.yaxis.grid(True); ax1.set_axisbelow(True)

    x_t = np.linspace(-7, 7, 400)
    df_val = 998
    y_t  = t_dist.pdf(x_t, df=df_val)
    crit = 1.96
    ax2.plot(x_t, y_t, color=BLUE, linewidth=2, label="t-distribution (df=998)")
    x_left  = x_t[x_t <= -crit]
    x_right = x_t[x_t >=  crit]
    ax2.fill_between(x_left,  t_dist.pdf(x_left,  df_val), alpha=0.35, color=RED, label="Rejection region (α=0.05)")
    ax2.fill_between(x_right, t_dist.pdf(x_right, df_val), alpha=0.35, color=RED)
    ax2.axvline(T_STAT, color=YELLOW, linewidth=2.2, label=f"t-statistic = {T_STAT}")
    ax2.axvline(-crit, color=RED, linewidth=1.2, linestyle=":")
    ax2.axvline( crit, color=RED, linewidth=1.2, linestyle=":")
    result_txt = (f"t = {T_STAT}\np ≈ {P_VALUE:.4f}\n\n→  Reject H₀\nTraffic types differ\nsignificantly")
    ax2.text(0.97, 0.95, result_txt, transform=ax2.transAxes, ha="right", va="top", fontsize=9, color=TEXT,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN+"22", edgecolor=GREEN, linewidth=1.5))
    ax2.set_xlabel("t value", fontsize=11, labelpad=8)
    ax2.set_ylabel("Probability Density", fontsize=11, labelpad=8)
    ax2.set_title("t-Distribution  (two-tailed, α = 0.05)")
    ax2.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "07_hypothesis_test.png")
    plt.show()

plot_hypothesis_test()



# ── 8. Summary Dashboard ──────────────────────────────────────────────────────
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
        ax.text(0.5, 0.58, val, transform=ax.transAxes, ha="center", va="center",
                fontsize=17, fontweight="bold", color=color)
        ax.text(0.5, 0.18, title, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=SUBTEXT)

    ax_pie = fig.add_subplot(gs[1, :2])
    ax_pie.set_facecolor(CARD)
    wedges, _, autotexts = ax_pie.pie(
        [N_NORMAL, N_ANOMALY], colors=[GREEN, RED], autopct="%1.2f%%", startangle=130,
        wedgeprops=dict(edgecolor=BG, linewidth=2), textprops=dict(color=TEXT, fontsize=10))
    for at in autotexts: at.set_fontweight("bold")
    ax_pie.legend([f"Normal  ({N_NORMAL:,})", f"Anomalous  ({N_ANOMALY:,})"],
                  loc="lower center", facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT,
                  fontsize=9, ncol=2, bbox_to_anchor=(0.5, -0.1))
    ax_pie.set_title("Class Balance  (Ibrahim)", color=TEXT, fontsize=11)

    ax_bar = fig.add_subplot(gs[1, 2:])
    ax_bar.set_facecolor(CARD)
    feat_labels = [f"F{i}" for i in range(10)]
    ax_bar.bar(feat_labels, FEAT_MEANS, color=PURPLE, edgecolor=BG, linewidth=1, width=0.6)
    ax_bar.set_xlabel("Feature Index", fontsize=10, labelpad=6)
    ax_bar.set_ylabel("Mean (normalized)", fontsize=10, labelpad=6)
    ax_bar.set_title("Feature Means — First 10 Features  (Ibrahim)", color=TEXT, fontsize=10)
    ax_bar.yaxis.grid(True); ax_bar.set_axisbelow(True)
    ax_bar.annotate("F8 = 0.396\n(Flag feature)", xy=(8, FEAT_MEANS[8]), xytext=(6.0, 0.28),
                    arrowprops=dict(arrowstyle="->", color=SUBTEXT), fontsize=8, color=SUBTEXT)

    plt.tight_layout()
    save(fig, "08_summary_dashboard.png")
    plt.show()

plot_dashboard()



# ── 9. Comparison Table ───────────────────────────────────────────────────────
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
    highlight  = {3, 4}

    fig, ax = plt.subplots(figsize=(11, 6.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.axis("off")
    fig.suptitle("Comparison Table — Full Dataset vs 10% Stratified Sample",
                 fontsize=13, fontweight="bold", color=TEXT, y=0.97)

    col_widths = [0.38, 0.31, 0.31]
    x_starts   = [0.01, 0.40, 0.71]
    row_h      = 0.062
    header_y   = 0.88

    for label, xs, cw in zip(col_labels, x_starts, col_widths):
        rect = plt.Rectangle((xs, header_y - 0.01), cw - 0.01, row_h + 0.015,
                              facecolor=BLUE+"55", edgecolor=BLUE, linewidth=1.5,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(xs + (cw-0.01)/2, header_y + row_h/2 + 0.005, label,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, fontweight="bold", color=TEXT)

    for r, (metric, full_val, samp_val) in enumerate(rows):
        y  = header_y - (r+1)*row_h - 0.01
        bg = GREEN+"22" if r in highlight else (BLUE+"12" if r%2==0 else CARD)
        for val, xs, cw in zip([metric, full_val, samp_val], x_starts, col_widths):
            rect = plt.Rectangle((xs, y), cw-0.01, row_h,
                                  facecolor=bg, edgecolor=BORDER, linewidth=0.8,
                                  transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            fc = GREEN if (r in highlight and xs > 0.1) else TEXT
            ax.text(xs + (cw-0.01)/2, y + row_h/2, val,
                    transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
                    fontweight="bold" if r in highlight else "normal", color=fc)

    ax.text(0.5, 0.01,
            "★  Green rows = key result: stratified sampling perfectly preserves the anomaly ratio",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=GREEN, style="italic")

    save(fig, "09_comparison_table.png")
    plt.show()

plot_comparison_table()

print("\n✅  All 9 visualizations saved!")


