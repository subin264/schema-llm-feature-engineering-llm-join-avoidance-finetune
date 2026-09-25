import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

files = {
    "qwen": os.path.join(DATA, "qwen_baseline_reclassified.csv"),
    "llama": os.path.join(DATA, "llama_baseline_reclassified.csv"),
}

# column name little different every file
lab_names = ["final_label", "label", "rq_label", "f_label"]
hop_names = ["hop_diameter", "hop", "h_required"]

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    print(df.columns.tolist())
    raise SystemExit("column not found")

def aa_ok(v):
    s = str(v).strip().lower()
    return s in ("aa", "avoidance", "avoid")

def hop_table(df, hopcol, aacol):
    ct = pd.crosstab(df[hopcol], df[aacol])
    for h in [0, 1, 2, 3]:
        if h not in ct.index:
            ct.loc[h] = 0
    ct = ct.sort_index()
    if 1 not in ct.columns:
        ct[1] = 0
    if 0 not in ct.columns:
        ct[0] = 0
    ct = ct[[1, 0]]
    ct.columns = ["AA", "not"]
    return ct

def print_stat(ct, tag):
    print(tag)
    print(ct)
    print((ct["AA"] / ct.sum(axis=1) * 100).round(1))
    chi2, p, dof, exp = chi2_contingency(ct.values)
    print("chi2=", round(chi2, 2), "p=", p)
    print("min_exp=", exp.min())
    tab = sm.stats.Table(ct.values, shift_zeros=False)
    tr = tab.test_ordinal_association(
        row_scores=np.array(ct.index.tolist(), dtype=float),
        col_scores=np.array([1.0, 0.0]),
    )
    print("trend Z=", round(tr.zscore, 2), "p=", tr.pvalue)

def run(path, name):
    df = pd.read_csv(path)
    lab = find_col(df, lab_names)
    hop = find_col(df, hop_names)

    df["hop"] = pd.to_numeric(df[hop], errors="coerce")
    df = df.dropna(subset=["hop"])
    df["hop"] = df["hop"].astype(int)
    df["aa"] = df[lab].map(aa_ok).astype(int)

    print("====", name, "====")
    print("n=", len(df), lab, hop)
    print(df[lab].value_counts())

    ct = hop_table(df, "hop", "aa")
    print_stat(ct, "[response unit]")

    qcol = None
    for c in ["question_id", "qid", "id"]:
        if c in df.columns:
            qcol = c
            break
    if qcol is None:
        print("No item ID")
        return

    g = df.groupby([qcol, "hop"], as_index=False)["aa"].mean()
    g["maj"] = (g["aa"] >= 0.6).astype(int)  # 3/5
    ct2 = hop_table(g, "hop", "maj")
    print_stat(ct2, "[Majority of questions n=" + str(len(g)) + "]")

for name, path in files.items():
    if not os.path.exists(path):
        print("no", path)
        continue
    run(path, name)