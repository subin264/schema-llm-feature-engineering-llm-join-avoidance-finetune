"""
step1: filter out only case of empty code / parse fail / only concat
case caught here not seen as attempt, only passed one go to step2
"""
import ast
import os
import re
import inspect
import sqlite3
import pandas as pd

DATA_DIR = "/Users/baesubin/Schema_LLM_FE/RO2/dtat"          # remove Desktop/
GOLD_CSV = os.path.join(DATA_DIR, "financial_hop_count_results.csv")
QWEN_CSV = os.path.join(DATA_DIR, "qwen_classified_v3_n5.csv")    # add _n5
LLAMA_CSV = os.path.join(DATA_DIR, "llama_classified_v3_n5.csv")  # add _n5
DB_PATH = os.path.join(DATA_DIR, "financial.sqlite")

gold_df = pd.read_csv(GOLD_CSV)
qwen_df = pd.read_csv(QWEN_CSV)
llama_df = pd.read_csv(LLAMA_CSV)
print("gold_df:", gold_df.shape)
print("qwen_df:", qwen_df.shape)
print("llama_df:", llama_df.shape)

tables = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]
conn = sqlite3.connect(DB_PATH)
dfs = {}
for t in tables:
    dfs[t] = pd.read_sql('SELECT * FROM "{}"'.format(t), conn)

# ------------------------------------------------

def get_code(res):
    """
    here not check whether SQL statement is correct or not. just extract code block
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
    # if no fence, treat whole thing as code. if broken, goes to parse_fail anyway
    return res.strip()

def find_mj(node, lst=None):
    """
    actual h_actual just need to track join/merge that actually got called.
    concat is just wrong function trying to attach, so cut at step1
    could use walk too, but wrote recursive to leave variable/line number later
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
        find_mj(c, lst)
    return lst

def has_concat(tree):
    """
    case of only concat is not seen as attempt
    on other hand, if neither merge nor concat exist:
    - if hop 0, could be normal
    - if hop 1 or more, avoidance is seen at step3
    """
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "concat":
            return True
        if isinstance(n, ast.Name) and n.id == "concat":
            return True
    return False

def step1(res, hop=None):
    """
    hop not used for now. spot to put h_required at step3
    """
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
    # pass even with no merge. could be hop 0, avoidance seen at step3
    return {"ok": True, "why": None, "calls": calls}

def make_ns():
    # so it doesn't die even using df_account, account_df like this
    ns = {"pd": pd}
    for t in tables:
        ns[t] = dfs[t]
        ns["df_" + t] = dfs[t]
        ns[t + "_df"] = dfs[t]
    ns["df_distr"] = dfs["district"]
    return ns

def step2(code):
    # run only function definition and import
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
            result = fn()
            return {"ok": True, "why": None, "result": result}
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
        result = fn(*args)
        return {"ok": True, "why": None, "result": result}
    except Exception as e:
        return {"ok": False, "why": "call_error: " + str(e), "result": None}



if __name__ == "__main__":
    targets = [
        (QWEN_CSV, "qwen_response"),
        (LLAMA_CSV, "llama_response"),
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
        if "hop_diameter" not in df.columns:
            df = df.merge(gold_df[["question_id", "hop_diameter"]], on="question_id", how="left")
        hopcol = "hop_diameter"

        whys1 = []
        nmerge = []
        for i, row in df.iterrows():
            r1 = step1(row[col], row.get(hopcol))
            if r1["ok"]:
                whys1.append(None)
            else:
                whys1.append(r1["why"])
            nmerge.append(len(r1["calls"]))

        df["step1_status"] = whys1
        df["n_merge"] = nmerge

        whys2 = [None] * len(df)
        res2 = [None] * len(df)
        for i, row in df.iterrows():
            if pd.notna(df.loc[i, "step1_status"]):
                continue
            r2 = step2(get_code(row[col]))
            whys2[i] = r2["why"]
            if r2["result"] is None:
                res2[i] = None
            else:
                res2[i] = str(r2["result"])[:200]

        df["step2_status"] = whys2
        df["step2_result"] = res2

        print(df["step1_status"].value_counts(dropna=False))
        print(df["step2_status"].value_counts(dropna=False))

        name = os.path.basename(path)
        if "llama" in name:
            tag = "llama"
        else:
            tag = "qwen"
        out_path = os.path.join(DATA_DIR, "q2_step2_" + tag + "_n5.csv")
        df.to_csv(out_path, index=False)
        print(out_path)