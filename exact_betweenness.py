import pandas as pd
import itertools
import numpy as np
import igraph as ig
import multiprocessing as mp
import time
from datetime import datetime
import os

# ------------------------------------------------------------------
# PARAMETERS
METRICS = ["co", "vote_count", "vote_average", "revenue"]
INPUT_FILE = "movies_2000_2025.csv"  
OUTPUT_DIR = "betweenness_all_metrics"
IGNORE_KEYWORDS = {"aftercreditsstinger", "duringcreditsstinger"}
MIN_KEYWORD_FREQ = 2
MIN_VOTE_COUNT = 10
DEBUG = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

def dprint(msg):
    if DEBUG:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ------------------------------------------------------------------
# LOAD AND FILTER DATA
def load_and_filter_data(csv_file):
    dprint(f"Loading {csv_file}")
    t0 = time.time()

    df = pd.read_csv(csv_file)
    dprint(f"Original dataframe size: {df.shape}")

    df = df[
        (df["revenue"] > 0) &
        (df["vote_count"] >= MIN_VOTE_COUNT)
    ].copy()

    df["keywords"] = (
        df["keywords"]
        .astype(str)
        .str.lower()
        .str.split(",")
        .apply(lambda kws: [
            k.strip() for k in kws
            if k.strip() and k.strip() not in IGNORE_KEYWORDS
        ])
    )

    counts = df["keywords"].explode().value_counts()
    valid_keywords = set(counts[counts >= MIN_KEYWORD_FREQ].index)

    df["keywords"] = df["keywords"].apply(
        lambda kws: [k for k in kws if k in valid_keywords]
    )

    df = df[df["keywords"].map(len) >= 2]

    dprint(f"Filtered dataframe size: {df.shape}")
    dprint(f"Loaded in {time.time() - t0:.2f}s")
    return df

# ------------------------------------------------------------------
# BUILD MULTIGRAPH
def build_multigraph(df):
    t0 = time.time()
    edge_data = {}

    for _, r in df.iterrows():
        kws = sorted(r["keywords"])
        for k1, k2 in itertools.combinations(kws, 2):
            if (k1, k2) not in edge_data:
                edge_data[(k1, k2)] = {
                    "co": 0,
                    "vote_count": 0,
                    "vote_average_num": 0,  
                    "vote_average_den": 0,  
                    "revenue": 0,
                }

            # co
            edge_data[(k1, k2)]["co"] += 1
            # vote_count
            edge_data[(k1, k2)]["vote_count"] += r["vote_count"]
            # vote_revenue (weighted sum)
            edge_data[(k1, k2)]["vote_average_num"] += r["vote_average"] * r["vote_count"]
            edge_data[(k1, k2)]["vote_average_den"] += r["vote_count"]
            # revenue (log sum)
            edge_data[(k1, k2)]["revenue"] += np.log1p(r["revenue"])  

    nodes = sorted(set(itertools.chain.from_iterable(edge_data)))
    node_to_idx = {k: i for i, k in enumerate(nodes)}
    edges = [(node_to_idx[k1], node_to_idx[k2]) for k1, k2 in edge_data]

    g = ig.Graph(edges=edges, directed=False)
    g.vs["name"] = nodes

    # assign final edge weights
    g.es["co"] = [edge_data[e]["co"] for e in edge_data]
    g.es["vote_count"] = [edge_data[e]["vote_count"] for e in edge_data]
    # num/den
    g.es["vote_average"] = [
        edge_data[e]["vote_average_num"] / edge_data[e]["vote_average_den"]
        if edge_data[e]["vote_average_den"] > 0 else 0
        for e in edge_data
    ]
    g.es["revenue"] = [edge_data[e]["revenue"] for e in edge_data]  

    dprint(f"Graph built: nodes={g.vcount()}, edges={g.ecount()} ({time.time() - t0:.2f}s)")
    return g

# ------------------------------------------------------------------
# EXACT BETWEENNESS + PARALLEL
def exact_weighted_betweenness_worker(nodes, g, metric):
    weights = np.array(g.es[metric], dtype=float)
    inv_weights = np.reciprocal(weights)  

    betweenness = g.betweenness(vertices=nodes, directed=False, weights=inv_weights)
    return betweenness

def compute_exact_betweenness_parallel(g, metric):
    n_workers = max(1, mp.cpu_count() - 1)
    nodes = list(range(g.vcount()))
    chunks = np.array_split(nodes, n_workers)

    with mp.Pool(n_workers) as pool:
        results = pool.starmap(
            exact_weighted_betweenness_worker,
            [(chunk, g, metric) for chunk in chunks]
        )

    return np.concatenate(results)

# ------------------------------------------------------------------
# MAIN
if __name__ == "__main__":
    dprint(f"Starting exact betweenness computation for {INPUT_FILE}")

    df = load_and_filter_data(INPUT_FILE)
    if df.empty:
        dprint("No data after filtering — exiting")
        exit()

    G = build_multigraph(df)

    result_df = pd.DataFrame({"keyword": G.vs["name"]})

    for metric in METRICS:
        dprint(f"Computing exact betweenness for metric '{metric}'")
        t0 = time.time()
        betweenness = compute_exact_betweenness_parallel(G, metric)
        result_df[f"betweenness_{metric}"] = betweenness
        dprint(f"Done {metric} in {time.time() - t0:.2f}s")

    # sort by co and save
    result_df = result_df.sort_values("betweenness_co", ascending=False)
    base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    output_file = os.path.join(OUTPUT_DIR, f"exact_betweenness_all_metrics_{base_name}.csv")
    result_df.to_csv(output_file, index=False)
    dprint(f"Results saved to {output_file}")
    dprint("Done.")
