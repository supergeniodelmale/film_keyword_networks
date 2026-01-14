# Analyzing Film Markets with Keyword Networks

## 0) Requirements
<pre>
igraph
seaborn
tqdm
</pre>

A requirements.txt file is available:
```
pip install -r requirements.txt
```

## 1) Data & Results
Complete data is in:
<pre>
film_keyword_networks/
├── centralities_over_time/
│   └── folds/
│       ├── movies_2000_2004.csv
│       ├── movies_2005_2009.csv
│       ├── movies_2010_2014.csv
│       ├── movies_2015_2019.csv
│       └── movies_2020_2024.csv
├── closeness_all_metrics/
│   ├── exact_closeness_all_metrics_movies_2000_2025.csv
│   ├── exact_closeness_ranks_movies_2000_2025.csv
│   └── output_rank_diff.csv
├── betweenness_all_metrics/
│   └── exact_betweenness_all_metrics_movies_2000_2025.csv
├── significance_testing/
│   ├── co_significance.csv
│   ├── vote_count_significance.csv
│   ├── vote_average_significance.csv
│   └── revenue_significance.csv
└── movies_2000_2025.csv
</pre>


## 2) Usage
- `exact_closeness.py` builds multigraph and computes exact closeness centralities on `movies_2000_2025.csv` and outputs `closeness_all_metrics/exact_closeness_all_metrics_movies_2000_2025.csv`.
- `exact_betweenness.py` builds multigraph and computes exact betweenness centralities on `movies_2000_2025.csv` and outputs `betweenness_all_metrics/exact_betweenness_all_metrics_movies_2000_2025.csv`.
- `plot_p_value.py` plots the p-value distribution of `significance_testing/*_significance.py`.
