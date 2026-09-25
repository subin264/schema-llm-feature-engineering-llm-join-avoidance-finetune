"""
combine train_candidates_genuine_v4.csv + train_candidates_genuine_v5.csv to
make JSONL(train_data.jsonl) for LoRA training.

each line format:
  {"messages": [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<code, wrapped in ```python fence>"}]}

note:
  - prompt column is already saved in genuine_v4/v5.csv as "same baseline format as eval"
    (because at v4/v5 generation, merge instruction was added only to API call, and
    saving used build_prompt()'s result as is).
    means no need to remove the phrase separately here — check once more with assert
    in case of contamination.
  - assistant content wrapped in same ```python ... ``` fence that get_code() expects at eval time.
"""
import os
import json

import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_CSVS = [
    os.path.join(OUT_DIR, "train_candidates_genuine_v4.csv"),
    os.path.join(OUT_DIR, "train_candidates_genuine_v5.csv"),
]
OUT_JSONL = os.path.join(OUT_DIR, "train_data.jsonl")

BAD_MARKER = "Use pandas merge()"  # generation-only instruction. must not exist in saved prompt.


def main():
    dfs = [pd.read_csv(p) for p in SOURCE_CSVS]
    df = pd.concat(dfs, ignore_index=True)
    print("combined original row count:", len(df))

    # filter genuine_join once more (defense just in case. genuine csv should already have only True)
    if "genuine_join" in df.columns:
        before = len(df)
        df = df[df["genuine_join"] == True]
        if len(df) != before:
            print("warning: genuine_join False mixed in, excluded {} case".format(before - len(df)))

    # prompt contamination defense check
    contaminated = df["prompt"].astype(str).str.contains(BAD_MARKER, regex=False)
    if contaminated.any():
        raise RuntimeError(
            "merge instruction mixed into prompt ({} case) — recheck the save logic".format(contaminated.sum())
        )
    print("prompt contamination check passed (no merge instruction)")

    # remove duplicate by question_id + db_id (v4/v5 have different db_id so expect no dup, but defensive)
    before = len(df)
    df = df.drop_duplicates(subset=["db_id", "question_id"])
    if len(df) != before:
        print("dup removed:", before - len(df), "case")

    rows = []
    for _, r in df.iterrows():
        code = str(r["code"]).strip()
        assistant_content = "```python\n{}\n```".format(code)
        rows.append({
            "messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print("saved:", OUT_JSONL)
    print("total training sample count:", len(rows))
    print()
    print("db_id x hop_diameter distribution:")
    print(df.groupby(["db_id", "hop_diameter"]).size())


if __name__ == "__main__":
    main()
