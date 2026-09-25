import os
import sqlite3
import pandas as pd
import sqlglot
from sqlglot import exp
import networkx as nx

ROOT = "/Users/baesubin/Schema_LLM_FE"
GOLD_CSV = os.path.join(ROOT, "RO2", "dtat", "financial_hop_count_results.csv")
DB_PATH = os.path.join(ROOT, "RQ1", "RQ1_data", "financial.sqlite")
OUT_DIR = os.path.join(ROOT, "RQ3", "RQ3_data", "new")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "rq3_schema_text.csv")

edges = [
    ("client", "district"),
    ("account", "district"),
    ("disp", "client"),
    ("disp", "account"),
    ("card", "disp"),
    ("loan", "account"),
    ("order", "account"),
    ("trans", "account"),
]
G = nx.Graph()
G.add_edges_from(edges)
all_tables = list(G.nodes())

def get_gold_tables_ordered(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        seen = []
        for t in tree.find_all(exp.Table):
            name = t.name.lower()
            if name not in seen:
                seen.append(name)
        return seen
    except:
        return []

def min_dist(t, gold_list):
    dists = []
    for g in gold_list:
        try:
            dists.append(nx.shortest_path_length(G, t, g))
        except:
            dists.append(99)
    if dists:
        return min(dists)
    return 99

def rank_tables(gold_list):
    gold_set = set(gold_list)
    rest = [t for t in all_tables if t not in gold_set]
    rest = sorted(rest, key=lambda t: (min_dist(t, gold_list), t))
    return gold_list + rest

def tables_to_schema_text(table_list, col_cache):
    txt = ""
    for t in table_list:
        txt += "- {}({})\n".format(t, ", ".join(col_cache[t]))
    return txt

UNIFORM_K = 2
ADAPTIVE_MAP = {0: 1, 1: 2, 2: 4, 3: 5}

if __name__ == "__main__":
    df = pd.read_csv(GOLD_CSV)
    print(len(df))
    conn = sqlite3.connect(DB_PATH)

    col_cache = {}
    for t in all_tables:
        cols = pd.read_sql('PRAGMA table_info("{}");'.format(t), conn)
        col_cache[t] = cols["name"].tolist()

    rows = []
    n_skip = 0
    for _, row in df.iterrows():
        gold_list = get_gold_tables_ordered(row["gold_sql"])
        if not gold_list:
            n_skip += 1
            continue
        ranked = rank_tables(gold_list)
        hop = int(row["hop_diameter"])
        n_ad = ADAPTIVE_MAP.get(hop, 8)
        u_tbl = ranked[:UNIFORM_K]
        a_tbl = ranked[:n_ad]
        rows.append({
            "question_id": row["question_id"],
            "hop_diameter": hop,
            "uniform_tables": ",".join(u_tbl),
            "adaptive_tables": ",".join(a_tbl),
            "uniform_schema_text": tables_to_schema_text(u_tbl, col_cache),
            "adaptive_schema_text": tables_to_schema_text(a_tbl, col_cache),
        })

    print("skip", n_skip)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(len(out_df), OUT_CSV)
    print(out_df.head(2))
