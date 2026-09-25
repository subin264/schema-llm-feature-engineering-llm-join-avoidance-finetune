"""
setting and function collection that step1·2·3 use together

- folder path, list of 12 CSV
- gold SQL / hop / disp
- pull out code (get_code)
- join count (count_merge)
- run code (run_code)
- compare value (same_value)

"""

import os
import re
import ast
import signal
import sqlite3
import inspect
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # submission/finetuning
_ROOT = os.path.dirname(_HERE)  # submission/
SHARED_DIR = os.path.join(_ROOT, "data")
DB_PATH = os.path.join(SHARED_DIR, "financial.sqlite")
GOLD_CSV = os.path.join(SHARED_DIR, "financial_hop_count_results.csv")
DATA_DIR = os.path.join(_HERE, "data")
OUT_DIR = os.path.join(_HERE, "data")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    ("qwen_before_baseline", os.path.join(DATA_DIR, "qwen_before_baseline.csv"), "qwen_response", "before", "baseline", "qwen"),
    ("qwen_before_uniform", os.path.join(DATA_DIR, "qwen_before_uniform.csv"), "response", "before", "uniform", "qwen"),
    ("qwen_before_adaptive", os.path.join(DATA_DIR, "qwen_before_adaptive.csv"), "response", "before", "adaptive", "qwen"),
    ("llama_before_baseline", os.path.join(DATA_DIR, "llama_before_baseline.csv"), "llama_response", "before", "baseline", "llama"),
    ("llama_before_uniform", os.path.join(DATA_DIR, "llama_before_uniform.csv"), "response", "before", "uniform", "llama"),
    ("llama_before_adaptive", os.path.join(DATA_DIR, "llama_before_adaptive.csv"), "response", "before", "adaptive", "llama"),
    ("qwen_after_baseline", os.path.join(DATA_DIR, "qwen_after_baseline_n5.csv"), "qwen_response", "after", "baseline", "qwen"),
    ("qwen_after_uniform", os.path.join(DATA_DIR, "qwen_after_uniform_n5.csv"), "qwen_response", "after", "uniform", "qwen"),
    ("qwen_after_adaptive", os.path.join(DATA_DIR, "qwen_after_adaptive_n5.csv"), "qwen_response", "after", "adaptive", "qwen"),
    ("llama_after_baseline", os.path.join(DATA_DIR, "llama_after_baseline_n5.csv"), "llama_response", "after", "baseline", "llama"),
    ("llama_after_uniform", os.path.join(DATA_DIR, "llama_after_uniform_n5.csv"), "llama_response", "after", "uniform", "llama"),
    ("llama_after_adaptive", os.path.join(DATA_DIR, "llama_after_adaptive_n5.csv"), "llama_response", "after", "adaptive", "llama"),
]

TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]
EXEC_TIMEOUT = 60


def load_gold():
    g = pd.read_csv(GOLD_CSV)
    g["question_id"] = g["question_id"].astype(int)
    sql = dict(zip(g.question_id, g.gold_sql))
    hop = dict(zip(g.question_id, g.hop_diameter.astype(int)))
    disp = {}
    for q, s in sql.items():
        disp[q] = bool(re.search(r"\bdisp\b", str(s).lower()))
    return sql, hop, disp


def load_input(path, col):
    df = pd.read_csv(path)
    if len(df) != 530:
        raise RuntimeError(path + " row count " + str(len(df)))
    if col not in df.columns:
        raise RuntimeError(path + " no column: " + col)
    df["question_id"] = df["question_id"].astype(int)
    df["repeat_idx"] = df["repeat_idx"].astype(int)
    return df[["question_id", "repeat_idx", col]].rename(columns={col: "response"})


def get_code(res):
    if res is None or (isinstance(res, float) and pd.isna(res)):
        return ""
    res = str(res)
    for pat in (r"```python(.*?)```", r"```(.*?)```", r"```python(.*)"):
        m = re.search(pat, res, re.S)
        if m:
            return m.group(1).strip()
    return res.strip()


def count_merge(tree):
    # exclude '-'.join, os.path.join
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("merge", "join"):
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                continue
            if isinstance(v, ast.Attribute) and v.attr == "path":
                continue
            n += 1
    return n


def has_concat(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "concat":
            return True
    return False


_BASE_DFS = None


def base_dfs():
    global _BASE_DFS
    if _BASE_DFS is None:
        conn = sqlite3.connect(DB_PATH)
        _BASE_DFS = {}
        for t in TABLES:
            df = pd.read_sql('SELECT * FROM "{}"'.format(t), conn)
            for c in df.columns:
                if "date" in c.lower():
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            _BASE_DFS[t] = df
        conn.close()
    return _BASE_DFS


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _make_ns():
    ns = {"pd": pd, "np": np}
    for t, df in base_dfs().items():
        c = df.copy()
        ns[t] = c
        ns["df_" + t] = c
        ns[t + "_df"] = c
    return ns


def _pick_arg(pname, ns):
    p = pname.lower()
    if p in ns and isinstance(ns[p], pd.DataFrame):
        return ns[p]
    for t in sorted(TABLES, key=len, reverse=True):
        if t in p:
            return ns[t]
    return ns[TABLES[0]]


def run_code(code):
    signal.signal(signal.SIGALRM, _alarm)
    ns = _make_ns()
    signal.alarm(EXEC_TIMEOUT)
    try:
        exec(code, ns)
        names = re.findall(r"^def (\w+)\(", code, re.M)
        if not names or ns.get(names[-1]) is None:
            return None, "exec_no_function"
        fn = ns[names[-1]]
        out = fn(*[_pick_arg(p, ns) for p in inspect.signature(fn).parameters])
        return out, None
    except _Timeout:
        return None, "exec_timeout"
    except Exception as e:
        return None, "exec_error: " + type(e).__name__ + ": " + str(e)[:150]
    finally:
        signal.alarm(0)


def _norm(v):
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return round(float(v), 2)
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    return str(v).strip()


def _to_set(x):
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return set(tuple(_norm(v) for v in r) for r in x.itertuples(index=False))
    if isinstance(x, pd.Series):
        return set((_norm(v),) for v in x)
    if isinstance(x, (list, tuple, set, np.ndarray)):
        out = set()
        for it in x:
            if isinstance(it, (list, tuple)):
                out.add(tuple(_norm(v) for v in it))
            else:
                out.add((_norm(it),))
        return out
    if isinstance(x, dict):
        return set((_norm(v),) for v in x.values())
    return {(_norm(x),)}


def same_value(gold, pred):
    try:
        a, b = _to_set(gold), _to_set(pred)
    except Exception:
        return False
    return a is not None and b is not None and a == b