"""
RQ1 classify script - local run version

Part changed from Colab original:
- google.colab import, drive.mount removed (local so not need)
- all path changed to local path(/Users/baesubin/Schema_LLM_FE/RQ1/RQ1_data)
- output file also save to same folder

"""

import pandas as pd
import sqlite3
import ast
import re
import inspect
import sqlglot
from sqlglot import exp
import os

# path setting, only change here rest follow automatic
DATA_DIR = "/Users/baesubin/Schema_LLM_FE/RQ1/RQ1_data"

conn = sqlite3.connect(os.path.join(DATA_DIR, "financial.sqlite"))
gold_df = pd.read_csv(os.path.join(DATA_DIR, "financial_hop_count_results.csv"))
print("gold", gold_df.shape)

tables = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]
dfs = {}
for t in tables:
    dfs[t] = pd.read_sql('SELECT * FROM "{}"'.format(t), conn)
    print(t, dfs[t].shape)


def get_code(res):
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

def get_used_tables_ast(code, tables):
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    used = set()

    def matches(name):
        if name is None:
            return None
        parts = name.lower().split('_')
        for t in tables:
            if t in parts:
                return t
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            t = matches(node.id)
            if t:
                used.add(t)
        elif isinstance(node, ast.arg):
            t = matches(node.arg)
            if t:
                used.add(t)

    return used

def get_gold_tables(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return set(tb.name.lower() for tb in tree.find_all(exp.Table))
    except Exception:
        return set()


def get_gold_value(sql):
    try:
        r = pd.read_sql(sql, conn)
        return r.iloc[0, 0]
    except Exception:
        return None


def try_run(code):
    ns = dict(dfs)
    ns["pd"] = pd
    try:
        exec(code, ns)
    except Exception as e:
        return None, "exec_error: " + str(e)

    matches = re.findall(r"def (\w+)\(", code)   # space bug fixed state
    if not matches:
        return None, "no_function"
    fn_name = matches[-1]
    if fn_name not in ns:
        return None, "no_function"

    fn = ns[fn_name]
    sig = inspect.signature(fn)
    if len(sig.parameters) == 0:
        try:
            return fn(), None
        except Exception as e:
            return None, "call_error: " + str(e)

    args = []
    for pname in sig.parameters:
        matched = None
        for t in tables:
            if t in pname.lower():
                matched = dfs[t]
                break
        args.append(matched if matched is not None else list(dfs.values())[0])
    try:
        return fn(*args), None
    except Exception as e:
        return None, "call_error: " + str(e)


def classify(q_row, gold_row, col):
    code = get_code(q_row[col])
    if code == "":
        return "failed", "empty"

    used = get_used_tables_ast(code, tables)
    gold_tables = get_gold_tables(gold_row["gold_sql"])

    if gold_tables and (not gold_tables.issubset(used)):
        return "avoidance", None

    result, err = try_run(code)
    if err:
        return "failed", err

    gold_val = get_gold_value(gold_row["gold_sql"])
    if gold_val is None:
        return "failed", "gold_query_failed"

    try:
        if round(float(result), 2) == round(float(gold_val), 2):
            return "success", None
        return "failed", "wrong_value"
    except Exception:
        return "failed", "cant_compare"


#-- file list to process: (input file, response column name, output file)
jobs = [
    (
        os.path.join(DATA_DIR, "qwen_pilot_results_v3_n5.csv"),
        "qwen_response",
        os.path.join(DATA_DIR, "qwen_classified_v3_n5.csv"),
    ),
    (
        os.path.join(DATA_DIR, "llama_pilot_results_v3_n5.csv"),
        "llama_response",   # Llama file column name state clearly (below safety net also have)
        os.path.join(DATA_DIR, "llama_classified_v3_n5.csv"),
    ),
]

for in_path, col, out_path in jobs:
    df = pd.read_csv(in_path)
    print(in_path, df.shape)

    if col not in df.columns:
        if "qwen_response" in df.columns:
            col = "qwen_response"
        elif "llama_response" in df.columns:
            col = "llama_response"
        else:
            print(df.columns.tolist())
            continue

    labels = []
    errs = []
    for i, row in df.iterrows():
        qid = row["question_id"]
        print(i, qid)

        hit = gold_df[gold_df["question_id"] == qid]
        if len(hit) == 0:
            labels.append("failed")
            errs.append("no_gold")
            continue

        gold_row = hit.iloc[0]
        try:
            label, err = classify(row, gold_row, col)
        except Exception as e:
            label, err = "failed", "classify_error: " + str(e)

        labels.append(label)
        errs.append(err)

    df["label"] = labels
    df["error"] = errs
    df.to_csv(out_path, index=False)
    print(df["label"].value_counts())
    print(out_path)
