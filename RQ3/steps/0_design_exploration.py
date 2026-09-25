import os
import pandas as pd

ROOT = "/Users/baesubin/Schema_LLM_FE"

if __name__ == "__main__":
    # check data
    GOLD_CSV = os.path.join(ROOT, "RO2", "dtat", "financial_hop_count_results.csv")
    df = pd.read_csv(GOLD_CSV)
    print(df.columns.tolist())
    print(len(df))
    print(df['hop_diameter'].value_counts().sort_index() if 'hop_diameter' in df.columns else df.head(2))


    # check this data monotonicity
    import sqlglot
    from sqlglot import exp
    import networkx as nx

    GOLD_CSV = os.path.join(ROOT, "RO2", "dtat", "financial_hop_count_results.csv")
    df = pd.read_csv(GOLD_CSV)
    print(len(df))
    print(df["hop_diameter"].value_counts().sort_index())

    def get_gold_tables(sql):
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
            return set(t.name.lower() for t in tree.find_all(exp.Table))
        except:
            return set()

    df["n_tables"] = df["gold_sql"].apply(lambda s: len(get_gold_tables(s)))

    agg = df.groupby("hop_diameter")["n_tables"].agg(["count", "min", lambda x: x.mode()[0], "max"])
    agg.columns = ["n", "min", "mode", "max"]
    print(agg)

    maxes = agg["max"].tolist()
    ok = all(maxes[i] <= maxes[i + 1] for i in range(len(maxes) - 1))
    print("mono", ok)
    if not ok:
        print(pd.Series(maxes).cummax().tolist())


    # what to do about uniform candidate group of data

    GOLD_CSV = os.path.join(ROOT, "RO2", "dtat", "financial_hop_count_results.csv")
    df = pd.read_csv(GOLD_CSV)
    print(len(df))
    print(df["hop_diameter"].value_counts().sort_index())

    def get_gold_tables(sql):
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
            return set(t.name.lower() for t in tree.find_all(exp.Table))
        except:
            return set()

    df["gold_tables"] = df["gold_sql"].apply(get_gold_tables)
    df["n_tables"] = df["gold_tables"].apply(len)

    zero_rows = df[df["n_tables"] == 0]
    print("n0", len(zero_rows))
    if len(zero_rows) > 0:
        print(zero_rows[["question_id", "question", "gold_sql"]].to_string())

    for h in sorted(df["hop_diameter"].unique()):
        sub = df[df["hop_diameter"] == h]
        med = sub["n_tables"].median()
        outliers = sub[sub["n_tables"] > med + 2]
        if len(outliers) > 0:
            print("hop", h, "med", med)
            print(outliers[["question_id", "n_tables", "gold_sql"]].to_string())

    def safe_mode(x):
        m = x.mode()
        if len(m) > 0:
            return m.iloc[0]
        return None

    agg = df.groupby("hop_diameter")["n_tables"].agg(["count", "min", safe_mode, "max"])
    agg.columns = ["n", "min", "mode", "max"]
    print(agg)

    maxes = agg["max"].tolist()
    ok = all(maxes[i] <= maxes[i + 1] for i in range(len(maxes) - 1))
    print("mono", ok)
    if not ok:
        print(pd.Series(maxes).cummax().tolist())

    print(df["n_tables"].median(), df["n_tables"].mode().iloc[0])
    hop1 = df[df["hop_diameter"] == 1]["n_tables"]
    print(len(hop1), hop1.median(), hop1.mode().iloc[0])
    print(df["n_tables"].median() == hop1.median() and df["n_tables"].mode().iloc[0] == hop1.mode().iloc[0])


    # verify to check what rule to use at step2

    GOLD_CSV = os.path.join(ROOT, "RO2", "dtat", "financial_hop_count_results.csv")
    df = pd.read_csv(GOLD_CSV)
    print(len(df))

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

    def min_dist_to_gold(t, gold_list):
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
        rest = sorted(rest, key=lambda t: (min_dist_to_gold(t, gold_list), t))
        return gold_list + rest

    adaptive_map = {0: 1, 1: 2, 2: 4, 3: 5}
    UNIFORM_K = 2

    results = []
    n_skip = 0
    weird = []

    for _, row in df.iterrows():
        gold_list = get_gold_tables_ordered(row["gold_sql"])
        if not gold_list:
            n_skip += 1
            continue
        for t in gold_list:
            if t not in G:
                weird.append((row["question_id"], t))

        ranked = rank_tables(gold_list)
        hop = row["hop_diameter"]
        n_ad = adaptive_map.get(int(hop), 8)
        gold_set = set(gold_list)
        u = set(ranked[:UNIFORM_K])
        a = set(ranked[:n_ad])
        results.append({
            "question_id": row["question_id"],
            "hop": hop,
            "n_gold": len(gold_set),
            "uniform_covered": len(gold_set & u),
            "uniform_missing": len(gold_set) - len(gold_set & u),
            "adaptive_covered": len(gold_set & a),
            "adaptive_missing": len(gold_set) - len(gold_set & a),
        })

    res_df = pd.DataFrame(results)
    print("skip", n_skip)
    print("weird", weird)
    print(res_df.groupby("hop")["uniform_missing"].agg(["mean", "max"]))
    bad = res_df[res_df["adaptive_missing"] > 0]
    print(len(bad), len(res_df))
    if len(bad) > 0:
        print(bad)
