"""
step3 just combine step1 and step2 result and attach label only. does not run
code again.
what it does:

- combine step1 + step2 by question·repeat number
- attach label_exec (main criteria, exec first)
- attach label_static (sensitivity, avoidance first)
- write why it turned out that way
- save AA/FA/SA table, AA% by hop, trend test

result comes out as all_labeled.csv, summary_*.csv, summary_why.csv


"""



import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps"))
from step1_common import FILES, OUT_DIR

S1 = os.path.join(OUT_DIR, "step1")
S2 = os.path.join(OUT_DIR, "step2")

# label_exec: main criteria. exec first (decided in advance at methodology)
# label_static: sensitivity. avoidance first


def to_bool(x):
    return str(x) == "True"


def label(r, exec_first):
    if r.step1_status != "ok":
        return "FA"
    avoid = int(r.n_merge) < int(r.hop_diameter)
    ok = to_bool(r.exec_ok)
    if exec_first:
        if not ok:
            return "FA"
        if avoid:
            return "AA"
    else:
        if avoid:
            return "AA"
        if not ok:
            return "FA"
    if to_bool(r.value_match):
        return "SA"
    return "FA"


def why(r):
    if r.step1_status != "ok":
        return "step1_" + r.step1_status
    if not to_bool(r.exec_ok):
        e = str(r.err)
        return e.split(":")[0]
    if to_bool(r.value_match):
        return "success"
    return "semantic"


def trend(sub, lab):
    t = sub.assign(A=sub[lab].eq("AA")).groupby("hop_diameter")["A"].agg(["sum", "count"])
    s = t.index.values.astype(float)
    a = t["sum"].values
    n = t["count"].values
    p = a.sum() / n.sum()
    sb = np.average(s, weights=n)
    z = np.sum(n * (s - sb) * (a / n - p)) / np.sqrt(p * (1 - p) * np.sum(n * (s - sb) ** 2))
    chi2 = chi2_contingency(np.vstack([a, n - a]))[0]
    return round(chi2, 2), round(z, 2), 2 * (1 - norm.cdf(abs(z)))


if __name__ == "__main__":
    parts = []
    for name, *_ in FILES:
        a = pd.read_csv(os.path.join(S1, name + ".csv"), keep_default_na=False)
        b = pd.read_csv(os.path.join(S2, name + ".csv"), keep_default_na=False)
        m = a.merge(b, on=["question_id", "repeat_idx"], how="left", validate="one_to_one")
        if len(m) != 530:
            raise RuntimeError(name + " add rows " + str(len(m)))
        parts.append(m)

    allr = pd.concat(parts, ignore_index=True)
    allr["n_merge"] = pd.to_numeric(allr["n_merge"], errors="coerce").fillna(0).astype(int)
    allr["label_exec"] = [label(r, True) for r in allr.itertuples(index=False)]
    allr["label_static"] = [label(r, False) for r in allr.itertuples(index=False)]
    allr["why"] = [why(r) for r in allr.itertuples(index=False)]
    allr.drop(columns=["code"]).to_csv(os.path.join(OUT_DIR, "all_labeled.csv"), index=False)

    for lab in ["label_exec", "label_static"]:
        summ = (allr.groupby(["model", "condition", "phase"])[lab]
                .value_counts().unstack(fill_value=0)
                .reindex(columns=["AA", "FA", "SA"], fill_value=0))
        hop = (allr.assign(A=allr[lab].eq("AA"))
               .groupby(["model", "condition", "phase", "hop_diameter"])["A"]
               .mean().mul(100).round(1).unstack())
        print("\n=", lab, "AA/FA/SA ==")
        print(summ)
        print("\n==", lab, "hop AA% ==")
        print(hop)
        print("\n=", lab, "baseline trend =")
        for (mdl, ph), sub in allr[allr.condition == "baseline"].groupby(["model", "phase"]):
            c, z, p = trend(sub, lab)
            print(mdl, ph, "chi2", c, "z", z, "p", "%.3g" % p)
        summ.to_csv(os.path.join(OUT_DIR, "summary_" + lab + ".csv"))
        hop.to_csv(os.path.join(OUT_DIR, "summary_hop_AA_pct_" + lab + ".csv"))

    allr.groupby(["model", "condition", "phase"])["why"].value_counts().unstack(fill_value=0) \
        .to_csv(os.path.join(OUT_DIR, "summary_why.csv"))
    print("\nsaved", OUT_DIR)