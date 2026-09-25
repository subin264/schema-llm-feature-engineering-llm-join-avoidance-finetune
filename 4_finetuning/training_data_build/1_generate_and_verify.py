"""
for superhero + student_club's hop_diameter >= 2 question(80 count), process
each question one by one in order:
  1) run gold SQL on actual DB and get correct answer value
  2) generate equivalent pandas code with Claude(teacher model)
  3) actually run that code
  4) keep only the one where value match as training candidate
does not make separate intermediate file(gold value CSV etc).

prompt format is matched character-for-character with RQ1 baseline pilot
(run_qwen in RQ1/ro_1.py). but that template's "loan status prediction model"
phrase is financial-only wording, doesn't fit superhero/student_club question
in meaning — still, kept it as is on purpose to keep "prompt format identical
character-for-character with financial eval". if you don't want that, just
change the first sentence of build_prompt().

need condition:
  - ANTHROPIC_API_KEY must be exported in terminal
  - BIRD dev_databases/superhero/superhero.sqlite, dev_databases/student_club/student_club.sqlite must exist
    (BIRD dataset itself too large so not included in this repo — download dev set from
    https://bird-bench.github.io then change ROOT below to the local path where that data is)
  - reuse analyze() from RQ1/hop_count.py (same file as this repo's RQ1/hop_count.py)

note: this script already ran once and its result(train_candidates_genuine_v4/v5.csv,
train_data.jsonl) is already in this folder together. no need to run again unless
for reproduction purpose.
"""
import os
import re
import ast
import json
import time
import inspect
import sqlite3
import sys

import numpy as np
import pandas as pd
import anthropic

ROOT = "/Users/baesubin/Schema_LLM_FE"
sys.path.insert(0, os.path.join(ROOT, "RQ1"))
from hop_count import analyze  # noqa: E402

DEV_JSON = os.path.join(ROOT, "dev.json")
OUT_DIR = os.path.join(ROOT, "모든질문실험이후_데이터분석", "학습_데이터_선정")
os.makedirs(OUT_DIR, exist_ok=True)

# v1: version where arg matching was substring so hero_attribute got mismatched to attribute.
# v2: fixed to exact match first + longer table name match first, added raw_response/code logging.
# v3: raised MAX_TOKENS 500 -> 2048 to fix response truncation problem.
# v4: instruct merge() use only at generation time(saved prompt keeps baseline format as is),
#     added n_merge/genuine_join column to filter out "value correct but avoidance type
#     (no merge used)" code.
#     -> secured superhero+student_club 80 question, 35 genuine.
# v5: fell short of 100 training data target so added 6 more DB
#     (formula_1, codebase_community, thrombosis_prediction, toxicology, card_games,
#      european_football_2). debit_card_specializing excluded because its FK graph is
#      disconnected into 4 pieces, california_schools excluded from start because it
#      has 0 hop2+ question itself.
#     combine with v4's genuine 35 to build final training set (separate merge after
#     running main() below).
VERSION = "v5"
ATTEMPTS_CSV = os.path.join(OUT_DIR, "gen_attempts_log_{}.csv".format(VERSION))
PASSED_CSV = os.path.join(OUT_DIR, "train_candidates_passed_{}.csv".format(VERSION))
GENUINE_CSV = os.path.join(OUT_DIR, "train_candidates_genuine_{}.csv".format(VERSION))

TARGET_DBS = ["formula_1", "codebase_community", "thrombosis_prediction",
              "toxicology", "card_games", "european_football_2"]

# db_id -> domain name to put in prompt (corresponds to financial's "loan status prediction" spot)
DOMAIN_NAME = {
    "superhero": "superhero",
    "student_club": "student club",
    "formula_1": "Formula 1 racing",
    "codebase_community": "community Q&A site",
    "thrombosis_prediction": "thrombosis prediction",
    "toxicology": "toxicology",
    "card_games": "card game",
    "european_football_2": "European football",
}

# teacher model. raise to claude-opus-5 if pass rate too low.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
SLEEP_SEC = 0.5  # API rate limit margin

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY env var automatic


# schema text, same as get_schema in RQ1/ro_1.py

def get_schema(conn):
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()
    txt = ""
    for t in tables:
        cols = pd.read_sql('PRAGMA table_info("{}");'.format(t), conn)
        col_str = ", ".join(cols["name"].tolist())
        txt += "- {}({})\n".format(t, col_str)
    return txt, tables


#### prompt, same structure as run_qwen in RQ1/ro_1.py, just swap domain

def build_prompt(q, schema_text, domain):
    return """You are working on a feature engineering task for a {domain} analysis model.

Database schema:
{schema_text}

Question: {q}

Write a Python function using pandas that computes a feature answering this question.
Assume the tables are already loaded as pandas DataFrames with the same names as above.
Only output the Python code, no explanation.""".format(domain=domain, schema_text=schema_text, q=q)


# instruction added only at generation time. does not go into the input prompt that
# gets "saved" as training data(build_prompt result) — this is to keep eval prompt
# and training data's input format identical.
# ("how" the correct code got made is separate from the input format, so fine to
# change only here.)
GEN_ONLY_INSTRUCTION = (
    "\n\nUse pandas merge() to join tables. "
    "Do not use isin() or manual filtering as a substitute for joins."
)


def build_gen_prompt(q, schema_text, domain):
    return build_prompt(q, schema_text, domain) + GEN_ONLY_INSTRUCTION


def get_code(res):
    if res is None:
        return ""
    res = str(res)
    m = re.search(r"```python(.*?)```", res, re.S)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```(.*?)```", res, re.S)
    if m2:
        return m2.group(1).strip()
    return res.strip()


def generate_code(question, schema_text, domain, retries=3):
    # use the prompt with merge instruction attached for API call,
    # return the prompt to save as training data in exact same format as baseline(eval).
    saved_prompt = build_prompt(question, schema_text, domain)
    gen_prompt = build_gen_prompt(question, schema_text, domain)
    last_err = None
    for _ in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": gen_prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text, saved_prompt
        except Exception as e:
            last_err = e
            time.sleep(2)
    return "ERROR: {}".format(last_err), saved_prompt


def run_code(code, dfs):
    ns = dict(dfs)
    ns["pd"] = pd
    try:
        exec(code, ns)
    except Exception as e:
        return None, "exec_error: " + str(e)

    matches = re.findall(r"def (\w+)\(", code)
    if not matches:
        return None, "no_function"
    fn_name = matches[-1]
    if fn_name not in ns:
        return None, "no_function"
    fn = ns[fn_name]

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        try:
            return fn(), None
        except Exception as e:
            return None, "call_error: " + str(e)

    if len(sig.parameters) == 0:
        try:
            return fn(), None
        except Exception as e:
            return None, "call_error: " + str(e)

    args = []
    for pname in sig.parameters:
        pname_l = pname.lower()
        matched = None
        # priority 1: table name and parameter name exact match
        for t in dfs:
            if t.lower() == pname_l:
                matched = dfs[t]
                break
        # priority 2: substring contain, check longer table name first
        # (e.g. prevent "hero_attribute" parameter matching "attribute" first)
        if matched is None:
            for t in sorted(dfs, key=len, reverse=True):
                if t.lower() in pname_l:
                    matched = dfs[t]
                    break
        args.append(matched if matched is not None else list(dfs.values())[0])
    try:
        return fn(*args), None
    except Exception as e:
        return None, "call_error: " + str(e)


# --count merge/join calls (same as find_mj, RQ3 step1)

def count_merge(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("merge", "join"))


#compare value - same_value, same as before

def norm_val(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return round(float(v), 2)
    return str(v).strip()


def to_set(x):
    if x is None:
        return None

    if isinstance(x, pd.DataFrame):
        if x.empty:
            return set()
        return set(
            tuple(norm_val(v) for v in row)
            for row in x.itertuples(index=False)
        )

    if isinstance(x, pd.Series):
        return set((norm_val(v),) for v in x)

    if isinstance(x, (list, tuple, set)):
        out = set()
        for item in x:
            if isinstance(item, (list, tuple)):
                out.add(tuple(norm_val(v) for v in item))
            else:
                out.add((norm_val(item),))
        return out

    if pd.isna(x):
        return set()
    return set([(norm_val(x),)])


def same_value(gold, pred):
    a = to_set(gold)
    b = to_set(pred)
    if a is None or b is None:
        return False
    return a == b


# ----- pull hop2+ question, reuse analyze() from RQ1/hop_count.py

def get_hop2_questions(db_id, dev_rows):
    out = []
    for r in dev_rows:
        if r["db_id"] != db_id:
            continue
        try:
            res = analyze(r["SQL"])
        except Exception:
            continue
        if res.get("hop_diameter", 0) >= 2:
            out.append((r, res["hop_diameter"]))
    return out


def main():
    dev_rows = json.load(open(DEV_JSON, encoding="utf-8"))

    passed = []
    attempts = []

    for db_id in TARGET_DBS:
        db_path = os.path.join(ROOT, "dev_databases", db_id, "{}.sqlite".format(db_id))
        conn = sqlite3.connect(db_path)
        schema_text, tables = get_schema(conn)
        dfs = {t: pd.read_sql('SELECT * FROM "{}"'.format(t), conn) for t in tables}

        qs = get_hop2_questions(db_id, dev_rows)
        print(db_id, "hop2+ question count:", len(qs))

        for i, (r, hop) in enumerate(qs):
            qid = r["question_id"]
            question = r["question"]
            sql = r["SQL"]

            # 1) run gold
            try:
                gold_result = pd.read_sql(sql, conn)
            except Exception as e:
                attempts.append({"db_id": db_id, "question_id": qid, "hop": hop,
                                  "status": "gold_exec_fail", "detail": str(e), "raw_response": "", "code": ""})
                print("[{}] {}/{} qid={} hop={} -> gold_exec_fail".format(db_id, i + 1, len(qs), qid, hop))
                continue

            # 2) generate pandas code with Claude
            raw_response, prompt = generate_code(question, schema_text, DOMAIN_NAME[db_id])
            code = get_code(raw_response)

            if not code:
                attempts.append({"db_id": db_id, "question_id": qid, "hop": hop,
                                  "status": "empty_code", "detail": raw_response[:200],
                                  "raw_response": raw_response, "code": ""})
                print("[{}] {}/{} qid={} hop={} -> empty_code".format(db_id, i + 1, len(qs), qid, hop))
                time.sleep(SLEEP_SEC)
                continue

            # 3) run
            pred_result, err = run_code(code, dfs)
            if err:
                attempts.append({"db_id": db_id, "question_id": qid, "hop": hop,
                                  "status": "run_fail", "detail": err,
                                  "raw_response": raw_response, "code": code})
                print("[{}] {}/{} qid={} hop={} -> run_fail ({})".format(db_id, i + 1, len(qs), qid, hop, err))
                time.sleep(SLEEP_SEC)
                continue

            # 4) compare
            try:
                match = same_value(gold_result, pred_result)
            except Exception as e:
                attempts.append({"db_id": db_id, "question_id": qid, "hop": hop,
                                  "status": "compare_error", "detail": str(e),
                                  "raw_response": raw_response, "code": code,
                                  "n_merge": None, "genuine_join": False})
                print("[{}] {}/{} qid={} hop={} -> compare_error ({})".format(db_id, i + 1, len(qs), qid, hop, e))
                time.sleep(SLEEP_SEC)
                continue
            n_merge = count_merge(code)
            genuine_join = (n_merge is not None) and (n_merge >= hop)
            status = "pass" if match else "value_mismatch"
            attempts.append({"db_id": db_id, "question_id": qid, "hop": hop, "status": status, "detail": "",
                              "raw_response": raw_response, "code": code,
                              "n_merge": n_merge, "genuine_join": genuine_join})

            if match:
                passed.append({
                    "db_id": db_id,
                    "question_id": qid,
                    "question": question,
                    "hop_diameter": hop,
                    "schema_text": schema_text,
                    "prompt": prompt,
                    "code": code,
                    "n_merge": n_merge,
                    "genuine_join": genuine_join,
                })

            print("[{}] {}/{} qid={} hop={} -> {}".format(db_id, i + 1, len(qs), qid, hop, status))
            time.sleep(SLEEP_SEC)

        conn.close()

    att_df = pd.DataFrame(attempts)
    pass_df = pd.DataFrame(passed)
    att_df.to_csv(ATTEMPTS_CSV, index=False)
    pass_df.to_csv(PASSED_CSV, index=False)

    genuine_df = pass_df[pass_df["genuine_join"] == True] if len(pass_df) > 0 else pass_df
    genuine_df.to_csv(GENUINE_CSV, index=False)

    print("saved:", ATTEMPTS_CSV)
    print("saved:", PASSED_CSV, "(all value match, avoidance type included)")
    print("saved:", GENUINE_CSV, "(genuine_join == True only, will actually use for training)")

    print()
    print("=== summary ===")
    print("total attempt:", len(att_df))
    print("passed(value match):", len(pass_df))
    print("  of which genuine_join(used merge as much as hop):", len(genuine_df))
    print("  of which avoidance type(merge short, value only matched):", len(pass_df) - len(genuine_df))
    print(att_df["status"].value_counts())
    if len(pass_df) > 0:
        print()
        print("passed(whole) hop distribution (db_id x hop_diameter):")
        print(pass_df.groupby(["db_id", "hop_diameter"]).size())
    if len(genuine_df) > 0:
        print()
        print("genuine_join only hop distribution (db_id x hop_diameter):")
        print(genuine_df.groupby(["db_id", "hop_diameter"]).size())


if __name__ == "__main__":
    main()
