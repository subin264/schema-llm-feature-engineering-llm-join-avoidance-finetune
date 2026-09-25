""" step1 — empty code / parse fail / only concat used
- empty code: case LLM give no code at all
- parse fail: case LLM give code, but as python
- use only concat without merge/join: case LLM give code, parse as python ok, but
  no merge/join at all, just have concat(simple attach) only
+ after check against CAAFE actual implementation(run_llm_code.py) last time,
confirm as recursive version(later record context info together for h_actual track)
"""

import ast
import os
import re
import pandas as pd
import numpy as np
import sqlglot
from sqlglot import exp
from scipy.stats import chi2_contingency

# only change path here
DATA_DIR = '/Users/baesubin/Schema_LLM_FE/RO2/dtat'
GOLD_CSV = os.path.join(DATA_DIR, 'financial_hop_count_results.csv')
QWEN_CSV = os.path.join(DATA_DIR, 'qwen_classified_v3_n5.csv')
LLAMA_CSV = os.path.join(DATA_DIR, 'llama_classified_v3_n5.csv')

gold_df = pd.read_csv(GOLD_CSV)
qwen_df = pd.read_csv(QWEN_CSV)
llama_df = pd.read_csv(LLAMA_CSV)

print("gold_df:", gold_df.shape)
print("qwen_df:", qwen_df.shape)
print("llama_df:", llama_df.shape)

#------------------------------------------------


# pull out only code block
def get_code(res):
    """
    extract only actual python from whole LLM response text
    if explain text goes into ast.parse() as is, parse fail happen, so extract
    only code block
    - if no code mark at all, return whole response as code / not throw data away
    """
    if res is None or (type(res) == float and pd.isna(res)):
        return ""
    res = str(res)
    m = re.search(r"```python(.*?)```", res, re.S)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```(.*?)```", res, re.S)
    if m2:
        return m2.group(1).strip()
    return res.strip()

# look for merge/join (leave line number too, plan to use later for count hop)
def find_mj(node, lst=None):
    """
    go around with ast.iter_child_nodes() recursive, find merge/join call
    implement recursive because plan to use ast.walk later for hop track
    """
    if lst is None:
        lst = []
    if isinstance(node, ast.Attribute):
        if node.attr == "merge" or node.attr == "join":
            nm = None
            if isinstance(node.value, ast.Name):
                nm = node.value.id
            lst.append({"attr": node.attr, "var": nm, "line": getattr(node, "lineno", None)})
    for c in ast.iter_child_nodes(node):
        #current node children call one by one
        find_mj(c, lst)
    return lst

def has_concat(tree):
    """
    check whether concat call exist inside code
    """
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "concat":
            return True
        if isinstance(n, ast.Name) and n.id == "concat":
            return True
    return False

def step1(res, hop=None):
    """
    step1 — empty code / parse fail / only one concat, check later at avoidance part

    """
    code = get_code(res)

    if code == "":
        return {"ok": False, "why": "empty", "calls": []}

    try:
        tree = ast.parse(code)
    except:#approach broadly, return None if parse fail
        return {"ok": False, "why": "parse_fail", "calls": []}

    calls = find_mj(tree)

    # cut here only if concat-only case
    if len(calls) == 0 and has_concat(tree):
        return {"ok": False, "why": "concat_only", "calls": []}

    # hop 0 or no merge just pass for now (avoidance check later)
    #if no calls then question with hop 0 = becomes true
    return {"ok": True, "why": None, "calls": calls}

if __name__ == "__main__":
    import sys
    path = QWEN_CSV
    col = "qwen_response"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if len(sys.argv) > 2:
        col = sys.argv[2]

    df = pd.read_csv(path)

if __name__ == "__main__":
    targets = [
        (QWEN_CSV, "qwen_response"),
        (LLAMA_CSV, "qwen_response"),  
    ]

    for path, col in targets:
        df = pd.read_csv(path)
        print(path)

        if col not in df.columns:
            if "qwen_response" in df.columns:
                col = "qwen_response"
            elif "llama_response" in df.columns:
                col = "llama_response"
            else:
                print(df.columns.tolist())
                continue

        hopcol = None
        if "hop_diameter" in df.columns:
            hopcol = "hop_diameter"

        whys = []
        nmerge = []
        for i, row in df.iterrows():
            hp = None
            if hopcol is not None:
                hp = row[hopcol]
            r = step1(row[col], hp)
            if r["ok"]:
                whys.append(None)
            else:
                whys.append(r["why"])
            nmerge.append(len(r["calls"]))

        df["step1_status"] = whys
        df["n_merge"] = nmerge
        print(df["step1_status"].value_counts(dropna=False))
        print(df["step1_status"].isna().mean())

        name = os.path.basename(path)
        if "llama" in name:
            out = "q2_step1_llama.csv"
        else:
            out = "q2_step1_qwen.csv"
        out_path = os.path.join(DATA_DIR, out)
        df.to_csv(out_path, index=False)
        print(out_path)