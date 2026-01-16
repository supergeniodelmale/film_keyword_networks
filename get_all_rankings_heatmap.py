import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# LOAD
closeness = pd.read_csv("closeness_all_metrics/exact_closeness_all_metrics_movies_2000_2025.csv")
betweenness = pd.read_csv("betweenness_all_metrics/exact_betweenness_all_metrics_movies_2000_2025.csv")
K = 50


# ------------------------------------------------------------------
# RANK, MERGE AND PLOT
keyword_order = closeness["keyword"].tolist()

betweenness = (
    betweenness
    .set_index("keyword")
    .loc[keyword_order]
    .reset_index()
)

combined = pd.merge(
    closeness,
    betweenness,
    on="keyword",
    suffixes=("_closeness", "_betweenness")
)

combined = combined.set_index("keyword").iloc[:K]


metrics = ["co", "vote_count", "vote_average", "revenue"]

ordered_columns = []
pretty_labels = []

for m in metrics:
    ordered_columns.append(f"closeness_{m}")
    ordered_columns.append(f"betweenness_{m}")
    pretty_labels.append(f"{m}\ncloseness")
    pretty_labels.append(f"{m}\nbetweenness")

combined = combined[ordered_columns]


ranked = combined.rank(
    ascending=False,   # high centrality = rank 1
    method="min"
)

plt.figure(figsize=(14, max(6, K * 0.35)))

ax = sns.heatmap(
    ranked,
    cmap="viridis_r",
    annot=True,
    fmt=".0f",
    linewidths=0.5,
    cbar_kws={"label": "Rank (1 = best)"}
)

ax.set_xticklabels(pretty_labels, rotation=0)
cbar = ax.collections[0].colorbar
cbar.set_ticks([ranked.min().min(), ranked.max().max()])
cbar.set_ticklabels([int(ranked.min().min()), int(ranked.max().max())])
cbar.ax.invert_yaxis()   

plt.title(f"Top {K} Keywords — Closeness vs Betweenness")
#plt.ylabel("Keyword (closeness order)")
#plt.xlabel("Centrality + Metric")

plt.tight_layout()
plt.savefig(
    f"keyword_centrality_heatmap_top_{K}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
