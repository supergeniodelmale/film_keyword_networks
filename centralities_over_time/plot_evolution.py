import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# PARAMETERS
TYPE = "revenue"
CSV_FILE = f"{TYPE}_slope_sorted.csv"
N = 10  # top/worst keywords

# ------------------------------------------------------------------
# LOAD AND SORT
df = pd.read_csv(CSV_FILE)
folds = [c for c in df.columns if c not in ["keyword", "mean_rank_slope"]]

df_sorted = df.sort_values("mean_rank_slope", ascending=False)
top_keywords = df_sorted.head(N)
worst_keywords = df_sorted.tail(N)

# ------------------------------------------------------------------
# PLOT
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# rising
for _, row in top_keywords.iterrows():
    axes[0].plot(folds, row[folds], marker="o", label=row["keyword"])

axes[0].set_title(f"Rising ({TYPE})")
axes[0].set_xlabel("Time Fold")
axes[0].set_ylabel("Rank (lower = better)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# declining
for _, row in worst_keywords.iterrows():
    axes[1].plot(folds, row[folds], marker="o", label=row["keyword"])

axes[1].set_title(f"Declining ({TYPE})")
axes[1].set_xlabel("Time Fold")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

axes[0].invert_yaxis()

plt.tight_layout()
plt.show()
