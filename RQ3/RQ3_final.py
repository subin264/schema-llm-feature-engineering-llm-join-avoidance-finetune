"""
step5: RQ3 core statistic — baseline vs uniform vs adaptive, Cochran's Q test.
use Cochran's Q because same question(question_id) is repeated measured across
three condition (RQ1's chi-square/trend test was independent comparison between
hop group, but here is repeated measure of same subject across three condition,
so test method different).
N=5 repeat is first aggregated at question level (majority avoidance=1) before
put into test (prevent pseudo-replication, same principle as 5.2).
hop=3(n=14) has small sample so treat only as exploratory observation and mark
separately.
"""
# RQ3 statistic.
# look at same question in baseline / uniform / adaptive three condition -> Cochran's Q
# 5 repeat first collapsed per question by majority(AA 3 or more times) (block pseudo-replication)
# hop=3 has small n so reference only

import os
import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # submission/
RQ1_DIR = os.path.join(_ROOT, "data")  # shared financial_hop_count_results.csv
RQ3_DIR = os.path.join(_HERE, "data")
gold_df = pd.read_csv(os.path.join(RQ1_DIR, "financial_hop_count_results.csv"))

def q_avoid(df):
    # one question = 0/1. 1 if AA is 3 or more out of 5
    return df.groupby("question_id")["final_label"].apply(
        lambda x: int((x == "AA").sum() >= 3)
    )

def add_label(df):
    # step4 csv doesn't have final_label so make it here
    df = df.copy()
    labs = []
    for _, r in df.iterrows():
        if r["step3_avoidance"] == True:
            labs.append("AA")
        elif r.get("step4_label") in ["success", "semantic"]:
            labs.append(r.get("step4_label"))
        else:
            labs.append("FA")
    df["final_label"] = labs
    return df

if __name__ == "__main__":
    for model in ["qwen", "llama"]:
        print(model)
        base = pd.read_csv(os.path.join(RQ3_DIR, model + "_baseline_reclassified.csv"))
        uni = add_label(pd.read_csv(os.path.join(RQ3_DIR, model + "_uniform_step4.csv")))
        adp = add_label(pd.read_csv(os.path.join(RQ3_DIR, model + "_adaptive_step4.csv")))

        b = q_avoid(base)
        u = q_avoid(uni)
        a = q_avoid(adp)
        qids = sorted(set(b.index) & set(u.index) & set(a.index))
        print("n_q", len(qids))

        tab = pd.DataFrame({
            "baseline": b.reindex(qids),
            "uniform": u.reindex(qids),
            "adaptive": a.reindex(qids),
        }).dropna()
        print("n_test", len(tab))
        print("sum", tab.sum().to_dict())

        res = cochrans_q(tab.values)
        print("Q", res.statistic, "p", res.pvalue, "df", res.df)

        hops = gold_df.set_index("question_id")["hop_diameter"]
        tab["hop"] = tab.index.map(hops)
        for h in sorted(tab["hop"].dropna().unique()):
            sub = tab[tab["hop"] == h][["baseline", "uniform", "adaptive"]]
            n = len(sub)
            note = "small" if h == 3 or n < 20 else ""
            print("hop", int(h), "n", n, note)
            print("sum", sub.sum().to_dict())
            if n >= 5 and sub.values.std() > 0:
                try:
                    r = cochrans_q(sub.values)
                    print("Q", round(r.statistic, 3), "p", round(r.pvalue, 3))
                except Exception as e:
                    print("skip", e)
            else:
                print("skip")
        print()