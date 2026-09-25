"""
1. avoidance rate(%) by hop count — whether the pattern "avoidance grow as hop grow" really show
2. chi-square test — whether avoidance rate difference between hop group is statistically significant
3. expected cell < 5 warning — if sample too small cell exist, warn automatic (brought same logic already did this check before)
4. Cochran-Armitage trend test — go beyond just "there is a difference", verify also whether "avoidance grow monotonic(keep one direction) as hop grow"
5. whole label distribution table by hop(crosstab) — see at one glance how avoidance/failed/success split by hop
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm

DATA_DIR = "/Users/baesubin/Schema_LLM_FE/RQ1/RQ1_data"

gold_df = pd.read_csv(os.path.join(DATA_DIR, "financial_hop_count_results.csv"))
qwen_df = pd.read_csv(os.path.join(DATA_DIR, "qwen_classified_v3_n5.csv"))
llama_df = pd.read_csv(os.path.join(DATA_DIR, "llama_classified_v3_n5.csv"))
# if no hop, attach from gold
if "hop_diameter" not in qwen_df.columns:
    qwen_df = qwen_df.merge(gold_df[["question_id", "hop_diameter"]], on="question_id", how="left")
if "hop_diameter" not in llama_df.columns:
    llama_df = llama_df.merge(gold_df[["question_id", "hop_diameter"]], on="question_id", how="left")

def run_tests(df, name):
    hops = sorted(df["hop_diameter"].dropna().unique())
    avoid = []
    n = []
    for h in hops:
        sub = df[df["hop_diameter"] == h]
        n.append(len(sub))
        avoid.append((sub["label"] == "avoidance").sum())
    avoid = np.array(avoid)
    n = np.array(n)
    rest = n - avoid

    table = np.array([avoid, rest])
    chi2, p, dof, expected = chi2_contingency(table)

    print(name)
    print("n", dict(zip(hops, n)))
    print("avoid", dict(zip(hops, avoid)))
    pct = {}
    for h, a, nn in zip(hops, avoid, n):
        pct[h] = round(a / nn * 100, 1)
    print("pct", pct)
    print("chi2", chi2, "p", p, "dof", dof)
    if expected.min() < 5:
        print("expected min", expected.min())

    scores = np.array(hops)
    p_bar = avoid.sum() / n.sum()
    mean_score = np.average(scores, weights=n)
    num = np.sum(n * (scores - mean_score) * (avoid / n - p_bar))
    den = np.sqrt(p_bar * (1 - p_bar) * np.sum(n * (scores - mean_score) ** 2))
    z = num / den
    p_trend = 2 * (1 - norm.cdf(abs(z)))
    print("trend z", z, "p", p_trend)

    return {
        "hops": hops,
        "n": n,
        "avoid": avoid,
        "chi2": chi2,
        "p_chi2": p,
        "z_trend": z,
        "p_trend": p_trend,
    }

qwen_result = run_tests(qwen_df, "qwen")
llama_result = run_tests(llama_df, "llama")

print("crosstab qwen")
print(pd.crosstab(qwen_df["hop_diameter"], qwen_df["label"]))
print("crosstab llama")
print(pd.crosstab(llama_df["hop_diameter"], llama_df["label"]))