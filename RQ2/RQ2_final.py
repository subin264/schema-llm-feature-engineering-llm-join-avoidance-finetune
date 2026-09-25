"""
step5 core = core statistic(verify comparing overt/covert by relationship type
means split failed answer into visible failure / quiet failure two, and make
table of disp question

"""

import os
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data")
OUT = os.path.join(_HERE, "data")
RQ3 = os.path.join(_HERE, "data")
os.makedirs(OUT, exist_ok=True)

gold = pd.read_csv(os.path.join(DATA, "financial_hop_count_results.csv"))

def flag_disp(sql):
    if sql is None or (isinstance(sql, float) and pd.isna(sql)):
        return False
    return "disp" in str(sql).lower()

gold["has_disp"] = gold["gold_sql"].apply(flag_disp)

def pick_hop(df):
    for c in ["hop_diameter", "hop", "h_required"]:
        if c in df.columns:
            return c
    return None

def peek(df, name):
    print("----", name, "----")
    if "label" in df.columns:
        print("label", df["label"].value_counts(dropna=False).to_dict())
    if "error" in df.columns:
        print("error", df["error"].value_counts(dropna=False).head(10).to_dict())
    if "final_label" in df.columns:
        print("final_label", df["final_label"].value_counts(dropna=False).to_dict())
    print()

def err_legacy(row):
    # old 3 cell: avoidance / failed / success
    lab = row.get("label")
    err = row.get("error")
    if lab in ("avoidance", "success"):
        return None
    if lab == "failed":
        if err == "wrong_value":
            return "covert"
        return "overt"
    return None

def err_final(row):
    lab = row.get("final_label")
    if lab == "AA" or lab in ("SA", "success"):
        return None
    if lab == "semantic":
        return "covert"
    if lab == "FA":
        return "overt"
    return None

def run(df, name, fn, outp):
    """
    run()function is the first part, check which column to get from gold data
    """

    cols = ["question_id", "has_disp"]
    if "hop_diameter" in gold.columns:
        cols.append("hop_diameter")
    df = df.merge(gold[cols], on="question_id", how="left", suffixes=("", "_g"))
    hc = pick_hop(df)
    if hc is None:
        print(name, "no hop")
        return
    df = df[df[hc] >= 1]
    df["error_type"] = df.apply(fn, axis=1)
    sub = df[df["error_type"].isin(["overt", "covert"])]
    print(name, "n", len(sub), "nq", sub["question_id"].nunique())
    tab = pd.crosstab(sub["has_disp"], sub["error_type"])
    tab = tab.reindex(index=[False, True], columns=["overt", "covert"], fill_value=0)
    print(tab)
    if tab.values.sum() == 0:
        print("empty table")
    else:
        chi2, p, dof, exp = chi2_contingency(tab)
        print("chi2", round(chi2, 3), "p", p)
        print("expected min", exp.min())
        if exp.min() < 5:
            odds, fp = fisher_exact(tab.values)
            print("fisher p", fp)
        print("covert%", (tab["covert"] / tab.sum(axis=1) * 100).round(1))
    sub.to_csv(outp, index=False)
    print(outp)
    print()

if __name__ == "__main__":
    # old file: label / error
    print("LEGACY")
    lq = pd.read_csv(os.path.join(DATA, "q2_step4_qwen_n5.csv"))
    ll = pd.read_csv(os.path.join(DATA, "q2_step4_llama_n5.csv"))
    peek(lq, "qwen_step4")
    peek(ll, "llama_step4")
    run(lq, "qwen_legacy", err_legacy, os.path.join(OUT, "qwen_rq2_legacy.csv"))
    run(ll, "llama_legacy", err_legacy, os.path.join(OUT, "llama_rq2_legacy.csv"))

    # re-score: final_label
    print("FINAL")
    fq = pd.read_csv(os.path.join(RQ3, "qwen_baseline_reclassified.csv"))
    fl = pd.read_csv(os.path.join(RQ3, "llama_baseline_reclassified.csv"))
    peek(fq, "qwen_final")
    peek(fl, "llama_final")
    run(fq, "qwen_final", err_final, os.path.join(OUT, "qwen_rq2_final.csv"))
    run(fl, "llama_final", err_final, os.path.join(OUT, "llama_rq2_final.csv"))