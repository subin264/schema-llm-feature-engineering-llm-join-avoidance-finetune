"""
step2 is the stage that actually run only the code that passed step1
what it does:
- read step1 result CSV
- if step1_status not ok, don't run
- if ok, put in financial DB table and exec the code
- compare the value that came out with gold SQL answer result
- save only exec_ok, err, value_match

result saved to step2/<name>.csv

"""

import os
import sqlite3
import pandas as pd
from step1_common import FILES, OUT_DIR, DB_PATH, load_gold, run_code, same_value

S1 = os.path.join(OUT_DIR, "step1")
D = os.path.join(OUT_DIR, "step2")


def gold_result(qid, sql_map, cache):
    if qid not in cache:
        try:
            cache[qid] = pd.read_sql(sql_map[qid], _conn)
        except Exception:
            cache[qid] = None
    return cache[qid]


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)

    SQL, _, _ = load_gold()
    _conn = sqlite3.connect(DB_PATH)
    _gold = {}

    for name, *_ in FILES:
        s1 = pd.read_csv(os.path.join(S1, name + ".csv"), keep_default_na=False)
        out_path = os.path.join(D, name + ".csv")
        done = {}
        if os.path.exists(out_path):
            prev = pd.read_csv(out_path, keep_default_na=False)
            for rec in prev.to_dict("records"):
                done[(int(rec["question_id"]), int(rec["repeat_idx"]))] = rec

        rows = []
        for i, r in enumerate(s1.itertuples(index=False)):
            key = (int(r.question_id), int(r.repeat_idx))
            if key in done:
                rows.append(done[key])
                continue
            if r.step1_status != "ok":
                exec_ok, err, match = "", "", ""
            else:
                res, e = run_code(r.code)
                exec_ok = e is None
                err = e or ""
                match = same_value(gold_result(key[0], SQL, _gold), res) if exec_ok else ""
            rows.append({
                "question_id": key[0],
                "repeat_idx": key[1],
                "exec_ok": exec_ok,
                "err": err,
                "value_match": match,
            })
            if (i + 1) % 25 == 0:
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(name, i + 1, "/ 530")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(name, "done")

    print("saved", D)