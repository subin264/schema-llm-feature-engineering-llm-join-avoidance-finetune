import ast
import os
import re
import inspect
import sqlite3
import pandas as pd

DATA_DIR = "/Users/baesubin/Schema_LLM_FE/RO2/dtat"
GOLD_CSV = os.path.join(DATA_DIR, "financial_hop_count_results.csv")
DB_PATH = os.path.join(DATA_DIR, "financial.sqlite")

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

def make_ns():
    """
    not much data to make into table,
    so put in alias so we don't lose RQ2 sample because of that
    """
    ns = {"pd": pd}
    for t in tables:
        ns[t] = dfs[t]
        ns["df_" + t] = dfs[t]
        ns[t + "_df"] = dfs[t]
    ns["df_distr"] = dfs["district"]
    return ns

def run_and_get_result(code):
    """
    run code and return result. return None if error
    - if there is function definition inside code, run only that function and return result
    - if no function definition, run whole code and return last value
    - even if function definition exist, if there is parameter, find parameter that
      has table name in it, and put that table in
    """
    ns = make_ns()
    try:
        tree = ast.parse(code)
    except Exception as e:
        return None, "exec_error: " + str(e)

    matches = re.findall(r"def (\w+)\(", code)
    if not matches:
        return None, "no_function"
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
        return None, "no_function"

    try:
        exec(compile(ast.Module(body=keep, type_ignores=[]), "<ast>", "exec"), ns)
    except Exception as e:
        return None, "exec_error: " + str(e)

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
        if matched is None:
            matched = list(dfs.values())[0]
        args.append(matched)
    try:
        return fn(*args), None
    except Exception as e:
        return None, "call_error: " + str(e)

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
    """
    check whether result value and gold value same
    """
    if gold is None:
        return None
    # single number cell
    if gold.shape[0] == 1 and gold.shape[1] == 1:
        g = gold.iloc[0, 0]
        try:
            if hasattr(result, "iloc"):
                r = result.iloc[0]
                if hasattr(r, "iloc"):
                    r = r.iloc[0]
            else:
                r = result
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

def has_disp(sql):
    if sql is None or (type(sql) == float and pd.isna(sql)):
        return False
    return "disp" in str(sql).lower()

if __name__ == "__main__":
    targets = [
        (os.path.join(DATA_DIR, "q2_step3_qwen.csv"), "qwen_response", "qwen"),
        (os.path.join(DATA_DIR, "q2_step3_llama.csv"), "llama_response", "llama"),
    ]

    for path, col, name in targets:
        df = pd.read_csv(path)
        if col not in df.columns:
            if "qwen_response" in df.columns:
                col = "qwen_response"
            else:
                col = "llama_response"

        idxs = df[df["step3_avoidance"] == False].index
        print(name, "n", len(idxs))

        labels = [None] * len(df)
        pred_s = [None] * len(df)
        gold_s = [None] * len(df)
        disp_s = [None] * len(df)

        for i in idxs:
            qid = df.loc[i, "question_id"]
            hit = gold_df[gold_df["question_id"] == qid]
            if len(hit) == 0:
                labels[i] = "no_gold"
                continue
            grow = hit.iloc[0]
            disp_s[i] = has_disp(grow["gold_sql"])

            result, err = run_and_get_result(get_code(df.loc[i, col]))
            if err:
                labels[i] = "rerun_error: " + err
                continue

            gtab = get_gold_table(grow["gold_sql"])
            match = values_match(result, gtab)
            pred_s[i] = str(result)[:200]
            gold_s[i] = str(gtab.values.tolist()[:5] if gtab is not None else None)

            if match is None:
                labels[i] = "cant_compare"
            elif match:
                labels[i] = "success"
            else:
                labels[i] = "semantic"

        df["step4_label"] = labels
        df["step4_result"] = pred_s
        df["step4_gold"] = gold_s
        df["gold_has_disp"] = disp_s
        print(df["step4_label"].value_counts(dropna=False))

        out_path = os.path.join(DATA_DIR, "q2_step4_" + name + "_n5.csv")
        df.to_csv(out_path, index=False)
        print(out_path)