import ast
import os
import re
import inspect
import sqlite3
import pandas as pd

ROOT = "/Users/baesubin/Schema_LLM_FE"
INPUT_DIR = os.path.join(ROOT, "RQ3", "RQ3_data", "final")
OUTPUT_DIR = os.path.join(ROOT, "RQ3", "RQ3_data", "new")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GOLD_CSV = os.path.join(ROOT, "RO2", "dtat", "financial_hop_count_results.csv")
DB_PATH = os.path.join(ROOT, "RQ1", "RQ1_data", "financial.sqlite")

gold_df = pd.read_csv(GOLD_CSV)
tables = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]
conn = sqlite3.connect(DB_PATH)
dfs = {}
for t in tables:
    dfs[t] = pd.read_sql('SELECT * FROM "{}"'.format(t), conn)

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

def find_mj(node, lst=None):
    if lst is None:
        lst = []
    if isinstance(node, ast.Attribute):
        if node.attr == "merge" or node.attr == "join":
            nm = None
            if isinstance(node.value, ast.Name):
                nm = node.value.id
            lst.append({"attr": node.attr, "var": nm, "line": getattr(node, "lineno", None)})
    for c in ast.iter_child_nodes(node):
        find_mj(c, lst)
    return lst

def has_concat(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "concat":
            return True
        if isinstance(n, ast.Name) and n.id == "concat":
            return True
    return False

def get_used_tables(code):
    code_lower = code.lower()
    return set(t for t in tables if t in code_lower)

def step1(res, hop=None):
    code = get_code(res)
    if code == "":
        return {"ok": False, "why": "empty", "calls": []}
    try:
        tree = ast.parse(code)
    except:
        return {"ok": False, "why": "parse_fail", "calls": []}
    calls = find_mj(tree)
    if len(calls) == 0 and has_concat(tree):
        return {"ok": False, "why": "concat_only", "calls": []}
    return {"ok": True, "why": None, "calls": calls}

def make_ns():
    ns = {"pd": pd}
    for t in tables:
        ns[t] = dfs[t]
        ns["df_" + t] = dfs[t]
        ns[t + "_df"] = dfs[t]
    ns["df_distr"] = dfs["district"]
    return ns

def step2(code):
    ns = make_ns()
    try:
        tree = ast.parse(code)
    except Exception as e:
        return {"ok": False, "why": "exec_error: " + str(e), "result": None}
    matches = re.findall(r"def (\w+)\(", code)
    if not matches:
        return {"ok": False, "why": "no_function", "result": None}
    fn_name = matches[-1]
    keep = []
    found = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            keep.append(node)
            found = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
    if not found:
        return {"ok": False, "why": "no_function", "result": None}
    try:
        exec(compile(ast.Module(body=keep, type_ignores=[]), "<ast>", "exec"), ns)
    except Exception as e:
        return {"ok": False, "why": "exec_error: " + str(e), "result": None}
    if fn_name not in ns:
        return {"ok": False, "why": "no_function", "result": None}
    fn = ns[fn_name]
    sig = inspect.signature(fn)
    if len(sig.parameters) == 0:
        try:
            return {"ok": True, "why": None, "result": fn()}
        except Exception as e:
            return {"ok": False, "why": "call_error: " + str(e), "result": None}
    args = []
    for pname in sig.parameters:
        matched = None
        for t in tables:
            if t in pname.lower():
                matched = dfs[t]
                break
        if matched is None:
            matched = list(dfs.values())[0]
        args.append(matched)
    try:
        return {"ok": True, "why": None, "result": fn(*args)}
    except Exception as e:
        return {"ok": False, "why": "call_error: " + str(e), "result": None}

def step3(n_merge, hop_required):
    if pd.isna(n_merge) or pd.isna(hop_required):
        return {"avoidance": None, "overjoin": None}
    n_merge = int(n_merge)
    hop_required = int(hop_required)
    if n_merge < hop_required:
        return {"avoidance": True, "overjoin": False}
    if n_merge == hop_required:
        return {"avoidance": False, "overjoin": False}
    return {"avoidance": False, "overjoin": True}

if __name__ == "__main__":
    targets = [
        (os.path.join(INPUT_DIR, "qwen_uniform_results_n5.csv"), "qwen_uniform"),
        (os.path.join(INPUT_DIR, "qwen_adaptive_results_n5.csv"), "qwen_adaptive"),
        (os.path.join(INPUT_DIR, "llama_uniform_results_n5.csv"), "llama_uniform"),
        (os.path.join(INPUT_DIR, "llama_adaptive_results_n5.csv"), "llama_adaptive"),
    ]
    for path, name in targets:
        if not os.path.exists(path):
            print("no file", path)
            continue
        df = pd.read_csv(path)
        print(name, df.shape)
        col = "response"
        if col not in df.columns:
            print(df.columns.tolist())
            continue
        if "hop_diameter" not in df.columns:
            df = df.merge(gold_df[["question_id", "hop_diameter"]], on="question_id", how="left")

        whys1 = []
        nmerge = []
        used_dbg = []
        for i, row in df.iterrows():
            r1 = step1(row[col], row.get("hop_diameter"))
            whys1.append(None if r1["ok"] else r1["why"])
            nmerge.append(len(r1["calls"]))
            used_dbg.append(sorted(get_used_tables(get_code(row[col]))))
        df["step1_status"] = whys1
        df["n_merge"] = nmerge
        df["used_tables_debug"] = used_dbg

        whys2 = [None] * len(df)
        res2 = [None] * len(df)
        for i, row in df.iterrows():
            if pd.notna(df.loc[i, "step1_status"]):
                continue
            r2 = step2(get_code(row[col]))
            whys2[i] = r2["why"]
            res2[i] = None if r2["result"] is None else str(r2["result"])[:200]
        df["step2_status"] = whys2
        df["step2_result"] = res2

        avoid_list = []
        overjoin_list = []
        for i, row in df.iterrows():
            if pd.notna(row.get("step1_status")) or pd.notna(row.get("step2_status")):
                avoid_list.append(None)
                overjoin_list.append(None)
                continue
            r3 = step3(row["n_merge"], row["hop_diameter"])
            avoid_list.append(r3["avoidance"])
            overjoin_list.append(r3["overjoin"])
        df["step3_avoidance"] = avoid_list
        df["step3_overjoin_tag"] = overjoin_list

        print(df["step1_status"].value_counts(dropna=False))
        print(df["step2_status"].value_counts(dropna=False))
        print(df["step3_avoidance"].value_counts(dropna=False))
        print(df["step3_overjoin_tag"].value_counts(dropna=False))

        out_path = os.path.join(OUTPUT_DIR, name + "_step3.csv")
        df.to_csv(out_path, index=False)
        print(out_path)