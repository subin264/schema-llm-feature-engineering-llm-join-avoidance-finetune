"""
step1.py is the stage that pulls out code and only look at syntax, doesn't run it

what it does:

- read response from 12 CSV
- cut only code block
- parse as python
- count merge/join(n_merge)
- if hop bigger than 0 but no join, only concat, mark concat_only

result saved to step1/<name>.csv

"""

import os
import ast
import pandas as pd
from step1_common import FILES, OUT_DIR, load_gold, load_input, get_code, count_merge, has_concat

if __name__ == "__main__":
    D = os.path.join(OUT_DIR, "step1")
    os.makedirs(D, exist_ok=True)
    _, HOP, DISP = load_gold()

    for name, path, col, phase, cond, model in FILES:
        df = load_input(path, col)
        rows = []
        for r in df.itertuples(index=False):
            code = get_code(r.response)
            status = "ok"
            n = None
            if not code:
                status = "no_code"
            else:
                try:
                    tree = ast.parse(code)
                    n = count_merge(tree)
                    if HOP[r.question_id] > 0 and n == 0 and has_concat(tree):
                        status = "concat_only"
                except SyntaxError:
                    status = "parse_fail"
            rows.append({
                "file": name,
                "phase": phase,
                "condition": cond,
                "model": model,
                "question_id": r.question_id,
                "repeat_idx": r.repeat_idx,
                "hop_diameter": HOP[r.question_id],
                "gold_has_disp": DISP[r.question_id],
                "step1_status": status,
                "n_merge": n,
                "code": code,
            })
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(D, name + ".csv"), index=False)
        print(name, out.step1_status.value_counts().to_dict())

    print("saved", D)