import pandas as pd
import itertools
import numpy as np
import igraph as ig
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import time

# ------------------------------------------------------------------
# PARAMETERS
INPUT_FILE = "movies_2000_2025.csv"
OUTPUT_FILE = "co_significance.csv"
REFERENCE_FILE = "closeness_all_metrics/exact_closeness_all_metrics_movies_2000_2025.csv"

METRIC = "co"  # metric to test: co, vote_count, vote_average, revenue
IGNORE_KEYWORDS = {"aftercreditsstinger", "duringcreditsstinger"}
MIN_KEYWORD_FREQ = 2
MIN_VOTE_COUNT = 10
DEBUG = True

N_SHUFFLES = 500
N_PIVOTS = 500
N_THREADS = max(1, os.cpu_count() - 1)
SHUFFLES_PER_TASK = 2

def dprint(msg):
    if DEBUG:
        print(msg, flush=True)

# ------------------------------------------------------------------
# LOAD AND FILTER DATA
def load_and_filter_data(csv_file):
    dprint(f"Loading {csv_file}")
    df = pd.read_csv(csv_file)
    df = df[(df["revenue"] > 0) & (df["vote_count"] >= MIN_VOTE_COUNT)].copy()

    df["keywords"] = (
        df["keywords"]
        .astype(str)
        .str.lower()
        .str.split(",")
        .apply(lambda kws: [k.strip() for k in kws if k.strip() and k.strip() not in IGNORE_KEYWORDS])
    )

    counts = df["keywords"].explode().value_counts()
    valid_keywords = set(counts[counts >= MIN_KEYWORD_FREQ].index)
    df["keywords"] = df["keywords"].apply(lambda kws: [k for k in kws if k in valid_keywords])
    df = df[df["keywords"].map(len) >= 2]

    dprint(f"Filtered dataframe size: {df.shape}")
    return df

# ------------------------------------------------------------------
# BIPARTITE
def build_bipartite(df):
    movie_keywords = df["keywords"].tolist()
    all_keywords = sorted(set(itertools.chain.from_iterable(movie_keywords)))
    keyword_to_idx = {k: i for i, k in enumerate(all_keywords)}
    return movie_keywords, keyword_to_idx

def rewire_bipartite(movie_keywords, n_swaps=10_000):
    movies = [set(kws) for kws in movie_keywords]
    n_movies = len(movies)
    for _ in range(n_swaps):
        m1, m2 = np.random.choice(n_movies, 2, replace=False)
        if not movies[m1] or not movies[m2]:
            continue
        k1 = np.random.choice(list(movies[m1]))
        k2 = np.random.choice(list(movies[m2]))
        if k1 == k2 or k2 in movies[m1] or k1 in movies[m2]:
            continue
        movies[m1].remove(k1)
        movies[m2].remove(k2)
        movies[m1].add(k2)
        movies[m2].add(k1)
    return [list(m) for m in movies]

def project_bipartite(movie_keywords, df, keyword_to_idx):
    edge_data = {}
    for kws, (_, r) in zip(movie_keywords, df.iterrows()):
        kws = sorted(kws)
        for k1, k2 in itertools.combinations(kws, 2):
            key = (k1, k2)
            if key not in edge_data:
                edge_data[key] = {
                    "co": 0,
                    "vote_count": 0,
                    "vote_average_num": 0,
                    "vote_average_den": 0,
                    "revenue": 0,
                }
            edge_data[key]["co"] += 1
            edge_data[key]["vote_count"] += r["vote_count"]
            edge_data[key]["vote_average_num"] += r["vote_average"] * r["vote_count"]
            edge_data[key]["vote_average_den"] += r["vote_count"]
            edge_data[key]["revenue"] += np.log1p(r["revenue"])

    edges = [(keyword_to_idx[k1], keyword_to_idx[k2]) for k1, k2 in edge_data]
    g = ig.Graph(edges=edges, directed=False)
    g.vs["name"] = list(keyword_to_idx.keys())

    g.es["co"] = [edge_data[e]["co"] for e in edge_data]
    g.es["vote_count"] = [edge_data[e]["vote_count"] for e in edge_data]
    g.es["vote_average"] = [
        edge_data[e]["vote_average_num"] / edge_data[e]["vote_average_den"] if edge_data[e]["vote_average_den"] > 0 else 0
        for e in edge_data
    ]
    g.es["revenue"] = [edge_data[e]["revenue"] for e in edge_data]

    return g

def bipartite_null_graph(df, movie_keywords, keyword_to_idx):
    rewired = rewire_bipartite(movie_keywords)
    return project_bipartite(rewired, df, keyword_to_idx)

# ------------------------------------------------------------------
# APPROX. CLOSENESS + PARALLEL
def approx_closeness_pivots(g, metric, n_pivots=N_PIVOTS):
    n = g.vcount()
    pivots = np.random.choice(n, min(n_pivots, n), replace=False)
    weights = np.array(g.es[metric], dtype=float)
    inv_weights = np.reciprocal(weights)
    closeness = np.zeros(n, dtype=float)
    for pivot in pivots:
        dist = np.array(g.shortest_paths_dijkstra(source=pivot, weights=inv_weights)[0])
        finite = np.where(dist > 0, dist, np.nan)
        closeness += 1 / finite # inverse because if metric is high than keywords are close
    closeness /= len(pivots)
    return closeness

def shuffle_worker_multi(df, movie_keywords, keyword_to_idx, metric, n_shuffles):
    results = []
    for _ in range(n_shuffles):
        g_null = bipartite_null_graph(df, movie_keywords, keyword_to_idx)
        closeness = approx_closeness_pivots(g_null, metric)
        results.append(closeness)
    return results

# ------------------------------------------------------------------
# SIGNIFICANCE TEST (CENTRALITIES)
def run_significance_test_with_external_reference(df, movie_keywords, keyword_to_idx, metric, reference_file):
    # observed centrality
    ref_df = pd.read_csv(reference_file).set_index("keyword")
    keywords = sorted(set(itertools.chain.from_iterable(df["keywords"])))
    ref_closeness = ref_df.loc[keywords, f"closeness_{metric}"].values

    result_df = pd.DataFrame({"keyword": keywords})
    result_df[f"closeness_{metric}"] = ref_closeness

    # generate distribution
    null_closeness_list = []
    tasks = [(df, movie_keywords, keyword_to_idx, metric, SHUFFLES_PER_TASK)
             for _ in range(N_SHUFFLES // SHUFFLES_PER_TASK)]

    with ProcessPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(shuffle_worker_multi, *task) for task in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Shuffling ({metric})"):
            null_closeness_list.extend(f.result())

    null_closeness = np.array(null_closeness_list)

    # mean and std
    null_mean = np.nanmean(null_closeness, axis=0)
    null_std  = np.nanstd(null_closeness, axis=0) + 1e-9

    z_scores = (ref_closeness - null_mean) / null_std

    # p-values (null <= observed)
    p_values = (np.sum(null_closeness <= ref_closeness[None, :], axis=0) + 1) / (null_closeness.shape[0] + 1)

    result_df["z_score"] = z_scores
    result_df["p_value"] = p_values

    # rank by observed
    result_df = result_df.sort_values(by=f"closeness_{metric}", ascending=False).reset_index(drop=True)

    return result_df
    
    
# ------------------------------------------------------------------
# SIGNIFICANCE TEST (RANKS)
def run_significance_test_on_ranks(df, movie_keywords, keyword_to_idx, metric, reference_file):
    # observed centrality
    ref_df = pd.read_csv(reference_file).set_index("keyword")
    keywords = sorted(set(itertools.chain.from_iterable(df["keywords"])))
    ref_values = ref_df.loc[keywords, f"closeness_{metric}"].values

    # compute ranks
    ref_ranks = np.argsort(-ref_values).argsort() + 1  

    result_df = pd.DataFrame({"keyword": keywords})
    result_df[f"rank_{metric}"] = ref_ranks

    # generate distribution
    null_ranks_list = []
    tasks = [(df, movie_keywords, keyword_to_idx, metric, SHUFFLES_PER_TASK)
             for _ in range(N_SHUFFLES // SHUFFLES_PER_TASK)]

    with ProcessPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(shuffle_worker_multi, *task) for task in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Shuffling ({metric})"):
            null_closeness_list = f.result()
            # convert to ranks
            for null_values in null_closeness_list:
                null_ranks = np.argsort(-null_values).argsort() + 1
                null_ranks_list.append(null_ranks)

    null_ranks = np.array(null_ranks_list)

    # mean and std
    null_mean = np.nanmean(null_ranks, axis=0)
    null_std  = np.nanstd(null_ranks, axis=0) + 1e-9 


    z_scores = (ref_ranks - null_mean) / null_std

    # p-values (null <= observed)
    p_values = (np.sum(null_ranks <= ref_ranks[None, :], axis=0) + 1) / (null_ranks.shape[0] + 1)

    result_df["z_score"] = z_scores
    result_df["p_value"] = p_values

    # rank by observed
    result_df = result_df.sort_values(by=f"rank_{metric}", ascending=True).reset_index(drop=True)

    return result_df



# ------------------------------------------------------------------
# MAIN
if __name__ == "__main__":
    dprint("Loading and filtering data...")
    df = load_and_filter_data(INPUT_FILE)
    movie_keywords, keyword_to_idx = build_bipartite(df)

    dprint(f"Running significance test for {METRIC}...")
    results_df = run_significance_test_on_ranks(df, movie_keywords, keyword_to_idx, METRIC, REFERENCE_FILE)

    results_df.to_csv(OUTPUT_FILE, index=False)
    dprint(f"Saved results to {OUTPUT_FILE}")
