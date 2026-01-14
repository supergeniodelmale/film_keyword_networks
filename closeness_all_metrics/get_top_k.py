import pandas as pd


CSV_FILE = "exact_closeness_ranks_movies_2000_2025.csv"  
METRIC = "closeness_revenue"  
TOP_N = 10

df = pd.read_csv(CSV_FILE)
rank_col = f"rank_{METRIC}"

if rank_col in df.columns:
    top_df = df.nsmallest(TOP_N, rank_col)
else:
    top_df = df.nlargest(TOP_N, METRIC)
top_df = top_df[["keyword", METRIC, rank_col] if rank_col in df.columns else ["keyword", METRIC]]
print(top_df)
# top_df.to_csv(f"top_{TOP_N}_{METRIC}.csv", index=False)
