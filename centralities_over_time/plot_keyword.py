import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# PARAMETERS
CSV_FILE = "revenue_slope_sorted.csv"
KEYWORDS = [
    "adventure",
    "mystery",
    "divorced couple",
    "supervillain",
    "superhero team",
]

# ------------------------------------------------------------------
# LOAD
df = pd.read_csv(CSV_FILE)
folds = [c for c in df.columns if c not in ["keyword", "mean_rank_slope"]]
df_sel = df[df["keyword"].isin(KEYWORDS)]

# ------------------------------------------------------------------
# PLOT
fig, ax = plt.subplots(figsize=(7, 5))

for _, row in df_sel.iterrows():
    ax.plot(
        folds,
        row[folds],
        marker="o",
        label=row["keyword"]
    )

ax.set_title("revenue")
ax.set_xlabel("Time Fold")
ax.set_ylabel("Rank (lower = better)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

ax.invert_yaxis()

plt.tight_layout()
plt.show()
