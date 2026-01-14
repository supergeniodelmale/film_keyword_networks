import pandas as pd

csv_file = "exact_closeness_all_metrics_movies_2000_2025.csv"
df = pd.read_csv(csv_file)

metrics = ["closeness_co", "closeness_vote_count", "closeness_vote_average", "closeness_revenue"]

for m in metrics:
    df[f"rank_{m}"] = df[m].rank(ascending=False, method="min") 

output_file = "exact_closeness_ranks_movies_2000_2025.csv"
df.to_csv(output_file, index=False)

print(f"Ranks saved to {output_file}")
