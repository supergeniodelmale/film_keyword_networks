# Closeness

- `convert_to_ranking.py` generates `exact_closeness_ranks_movies_2000_2025.csv` from  `exact_closeness_all_metrics_movies_2000_2025.csv` with the rankings for each metric.
- `get_rank_difference.py` generates `output_rank_diff.csv` from `exact_closeness_all_metrics_movies_2000_2025.csv` with all the rank differences (i.e. the opportunity gains).
- `get_top_k.py` returns the top-K keywords from `exact_closeness_all_metrics_movies_2000_2025.csv` accordingly to a selected metric.
