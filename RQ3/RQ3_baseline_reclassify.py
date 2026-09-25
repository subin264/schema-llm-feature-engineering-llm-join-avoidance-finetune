"""
plan A: reclassify baseline original response(qwen/llama_pilot_results_v3_n5.csv)
with exactly same logic as RQ3's step1~4(get_code, find_mj, step1, step2, step3, step4).
purpose: unify baseline·uniform·adaptive three condition to same judge criteria
so that Cochran's Q comparison stand fair.
as side thing, compare reclassify result against RQ1's original label(legacy)
with agreement rate.
"""
import ast
import os
import re
import inspect
import sqlite3
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)  # submission/
SHARED_DIR = os.path.join(ROOT, "data")  # financial.sqlite, financial_hop_count_results.csv
RQ1_DIR = os.path.join(ROOT, "RQ1", "data")  # qwen/llama_pilot_results_v3_n5.csv (raw model responses)
GOLD_CSV = os.path.join(SHARED_DIR, "financial_hop_count_results.csv")
DB_PATH = os.path.join(SHARED_DIR, "financial.sqlite")
OUT_DIR = os.path.join(_HERE, "data")
os.makedirs(OUT_DIR, exist_ok=True)

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

def step1(res):
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

def get_gold_table(sql):
    try:
        return pd.read_sql(sql, conn)
    except:
        return None

def to_df(result):
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame()
    try:
        return pd.DataFrame([result])
    except:
        return None

def values_match(result, gold):
    if gold is None:
        return None
    if gold.shape[0] == 1 and gold.shape[1] == 1:
        g = gold.iloc[0, 0]
        try:
            r = result.iloc[0] if hasattr(result, "iloc") else result
            if hasattr(r, "iloc"):
                r = r.iloc[0]
            return round(float(r), 2) == round(float(g), 2)
        except:
            return str(result).strip() == str(g).strip()
    pred = to_df(result)
    if pred is None:
        return False
    try:
        g2 = gold.copy()
        p2 = pred.copy()
        g2.columns = range(len(g2.columns))
        p2.columns = range(len(p2.columns))
        gs = set(tuple(x) for x in g2.astype(str).values.tolist())
        ps = set(tuple(x) for x in p2.astype(str).values.tolist())
        return gs == ps
    except:
        return False

def reclassify(path, resp_col):
    df = pd.read_csv(path)
    if resp_col not in df.columns:
        if "qwen_response" in df.columns:
            resp_col = "qwen_response"
        else:
            resp_col = "llama_response"
    if "hop_diameter" not in df.columns:
        df = df.merge(gold_df[["question_id", "hop_diameter"]], on="question_id", how="left")
    labels = []
    for _, row in df.iterrows():
        code = get_code(row[resp_col])
        r1 = step1(row[resp_col])
        if not r1["ok"]:
            labels.append("FA")
            continue
        n_merge = len(r1["calls"])
        r2 = step2(code)
        if not r2["ok"]:
            labels.append("FA")
            continue
        r3 = step3(n_merge, row["hop_diameter"])
        if r3["avoidance"]:
            labels.append("AA")
            continue
        hit = gold_df[gold_df["question_id"] == row["question_id"]]
        if hit.empty:
            labels.append("FA")
            continue
        gtab = get_gold_table(hit.iloc[0]["gold_sql"])
        match = values_match(r2["result"], gtab)
        if match is None:
            labels.append("FA")
        elif match:
            labels.append("SA")
        else:
            labels.append("semantic")
    df["final_label"] = labels
    return df

def compare_legacy(df_re, model):
    path = os.path.join(RQ1_DIR, model + "_classified_v3_n5.csv")
    if not os.path.exists(path):
        return None
    old = pd.read_csv(path)[["question_id", "label"]]
    m = df_re[["question_id", "final_label"]].merge(old, on="question_id", how="left")
    a = (m["final_label"] == "AA")
    b = (m["label"] == "avoidance")
    return (a == b).mean()

if __name__ == "__main__":
    for model, col in [("qwen", "qwen_response"), ("llama", "llama_response")]:
        path = os.path.join(RQ1_DIR, model + "_pilot_results_v3_n5.csv")
        df = reclassify(path, col)
        out = os.path.join(OUT_DIR, model + "_baseline_reclassified.csv")
        df.to_csv(out, index=False)
        print(model, len(df))
        print(df["final_label"].value_counts())
        agr = compare_legacy(df, model)
        if agr is not None:
            print(model, "agree", round(agr, 3))
        print()