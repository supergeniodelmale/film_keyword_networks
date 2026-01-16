import pandas as pd
import numpy as np
import os
import glob
import re

# ------------------------------------------------------------------
# PARAMETERS
INPUT_DIR = "vote_average"   # folder containing exact_closeness_*.csv
FILE_PATTERN = "exact_closeness_*.csv"
KEYWORD_COLUMN = "keyword"
VALUE_COLUMN = "closeness"   # column to rank
OUTPUT_CSV = f"{INPUT_DIR}_slope_sorted.csv"

# ------------------------------------------------------------------
# LOAD, RANK, SAVE
def extract_start_year(filename): 
    m = re.search(r"_(\d{4})_(\d{4})", filename)
    return int(m.group(1)) if m else float("inf")

input_files = glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN))
input_files = sorted(input_files, key=extract_start_year)

if not input_files:
    raise RuntimeError(f"No files found in {INPUT_DIR}")

print("Using folds in order:")
for f in input_files:
    print(" ", os.path.basename(f))


rankings = {}
all_keywords = set()

for file in input_files:
    df = pd.read_csv(file)

    fold_name = os.path.splitext(os.path.basename(file))[0].replace(
        "exact_closeness_", ""
    )


    # ranks from closeness
    # higher closeness > rank 1
    df["rank"] = df[VALUE_COLUMN].rank(method="min", ascending=False).astype(int)

    rank_map = dict(zip(df[KEYWORD_COLUMN], df["rank"]))
    rankings[fold_name] = rank_map
    all_keywords.update(rank_map.keys())

folds = list(rankings.keys())
t = np.arange(len(folds))


rows = []

for kw in sorted(all_keywords):
    row = {"keyword": kw}
    ranks = []

    for fold in folds:
        rank_map = rankings[fold]
        r = rank_map.get(kw, len(rank_map) + 1)  # missing keywords get worst rank
        row[fold] = r
        ranks.append(r)

    # linear regression
    b, a = np.polyfit(t, ranks, 1)
    row["mean_rank_slope"] = -b
    rows.append(row)

result_df = pd.DataFrame(rows)

# sort by best improvement
result_df = result_df.sort_values(
    by="mean_rank_slope",
    ascending=False
)


result_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nMean-rank-slope evolution saved to {OUTPUT_CSV}")
print(f"Keywords tracked: {len(result_df)}")
print(f"Folds used: {folds}")
