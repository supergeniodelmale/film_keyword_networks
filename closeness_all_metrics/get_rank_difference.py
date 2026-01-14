import pandas as pd

input_csv = 'exact_closeness_ranks_movies_2000_2025.csv' 
output_csv = 'output_rank_diff.csv'  
co_rank_col = 'rank_closeness_co' 


df = pd.read_csv(input_csv)
rank_cols = [col for col in df.columns if col.startswith('rank_') and col != co_rank_col]

for col in rank_cols:
    diff_col_name = f'{co_rank_col}_minus_{col}'
    df[diff_col_name] = df[co_rank_col] - df[col]

df_sorted = df.sort_values(by=co_rank_col, ascending=True)
output_columns = ['keyword', co_rank_col]
metric_cols = [col for col in df.columns if not col.startswith('rank_') and col != 'keyword']
output_columns.extend(metric_cols)
rank_diff_cols = [col for col in df.columns if col.endswith(tuple(rank_cols))]
output_columns.extend(rank_diff_cols)

df_sorted[output_columns].to_csv(output_csv, index=False)
print(f"Output saved to {output_csv}")