"""
debug v2 - test new detect function(get_used_tables_v4) based on Name/arg node.
not limited to merge/join call, look at only identifier(variable name, function
parameter) that actually used across whole code. comment/string automatic
excluded because AST don't catch them.

not doing whole reclassify, just apply new function to few existing avoidance
sample and compare.
"""

import pandas as pd
import ast
import re
import sqlite3
import sqlglot
from sqlglot import exp
import os

DATA_DIR = "/Users/baesubin/Schema_LLM_FE/RQ1/RQ1_data"

conn = sqlite3.connect(os.path.join(DATA_DIR, "financial.sqlite"))
gold_df = pd.read_csv(os.path.join(DATA_DIR, "financial_hop_count_results.csv"))
tables = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]


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


def get_gold_tables(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return set(tb.name.lower() for tb in tree.find_all(exp.Table))
    except Exception:
        return set()


# ---old version (only inside merge/join call)
def var_matches_table(var_name, table_name):
    if var_name is None:
        return False
    parts = var_name.lower().split("_")
    return table_name in parts


def get_used_tables_ast_old(code, tables):
    try:
        tree = ast.parse(code)
    except Exception:
        return set()
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("merge", "join"):
                if isinstance(node.func.value, ast.Name):
                    for t in tables:
                        if var_matches_table(node.func.value.id, t):
                            used.add(t)
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        for t in tables:
                            if var_matches_table(arg.id, t):
                                used.add(t)
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Name):
                        for t in tables:
                            if var_matches_table(kw.value.id, t):
                                used.add(t)
    return used


# new version, scan whole Name/arg node
def get_used_tables_v4(code, tables):
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


#### apply both new/old version to Q90, Q91, Q92 and compare
qwen_v3 = pd.read_csv(os.path.join(DATA_DIR, "qwen_classified_v3.csv"))

for qid in [90, 91, 92]:
    row = qwen_v3[qwen_v3['question_id'] == qid].iloc[0]
    code = get_code(row['qwen_response'])
    gold_row = gold_df[gold_df['question_id'] == qid].iloc[0]
    needed = get_gold_tables(gold_row['gold_sql'])

    old_result = get_used_tables_ast_old(code, tables)
    new_result = get_used_tables_v4(code, tables)

    print(f"\n===== Q{qid} =====")
    print("table that gold need:", needed)
    print("old version(only inside merge) detect:", old_result, "-> need condition satisfy:", needed.issubset(old_result))
    print("new version(Name/arg whole) detect:", new_result, "-> need condition satisfy:", needed.issubset(new_result))