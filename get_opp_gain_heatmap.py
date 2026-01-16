import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


df = pd.read_csv("closeness_all_metrics/exact_closeness_all_metrics_movies_2000_2025.csv")

metrics = ["closeness_co", "closeness_vote_count", "closeness_vote_average", "closeness_revenue"]

for m in metrics:
    df[f"rank_{m}"] = df[m].rank(ascending=False, method="min")

co_rank_col = "rank_closeness_co"

diff_cols = []
for m in metrics:
    rank_col = f"rank_{m}"
    if rank_col != co_rank_col:
        diff_col = f"{co_rank_col}_minus_{rank_col}"
        df[diff_col] = df[co_rank_col] - df[rank_col]
        diff_cols.append(diff_col)

top_keywords = df.nsmallest(50, co_rank_col)['keyword']

heatmap_df = df.set_index('keyword').loc[top_keywords, diff_cols]
heatmap_df.columns = ["co - vote count", "co - vote average", "co - revenue"]

heatmap_values = heatmap_df.values.astype(float)
scaled_values = np.zeros_like(heatmap_values)

neg_mask = heatmap_values < 0
pos_mask = heatmap_values > 0

scaled_values[neg_mask] = -np.log1p(np.abs(heatmap_values[neg_mask]))
scaled_values[pos_mask] = np.log1p(heatmap_values[pos_mask])

divnorm = Normalize(vmin=scaled_values.min(), vmax=scaled_values.max())

plt.figure(figsize=(10, 14))
ax = sns.heatmap(
    scaled_values,
    cmap='RdYlBu_r',  
    annot=heatmap_df.round(2),
    fmt='',
    linewidths=0.5,
    yticklabels=heatmap_df.index,
    xticklabels=heatmap_df.columns,
    center=0,
    cbar_kws={
        'label': 'Rank Difference (original values)',
        'ticks': [scaled_values.min(), 0, scaled_values.max()]
    }
)

cbar = ax.collections[0].colorbar
tick_locs = [scaled_values.min(), 0, scaled_values.max()]
tick_vals = np.sign(tick_locs) * (np.expm1(np.abs(tick_locs)))  # invert log scaling
cbar.set_ticks(tick_locs)
cbar.set_ticklabels([f"{v:.2f}" for v in tick_vals])

plt.title("Top 50 Opportunity Gain Keywords")
plt.ylabel("Keyword")
plt.xlabel("Rank Difference")
plt.tight_layout()
plt.savefig("opp_gain_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()  