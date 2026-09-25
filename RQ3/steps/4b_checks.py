import os
import pandas as pd

ROOT = "/Users/baesubin/Schema_LLM_FE"
DATA_DIR = os.path.join(ROOT, "RQ3", "RQ3_data", "new")
os.makedirs(DATA_DIR, exist_ok=True)

files = {
    "qwen_uniform": "qwen_uniform_step4.csv",
    "qwen_adaptive": "qwen_adaptive_step4.csv",
    "llama_uniform": "llama_uniform_step4.csv",
    "llama_adaptive": "llama_adaptive_step4.csv",
}

def make_label(row):
    if row["step3_avoidance"] == True:
        return "AA"
    if row.get("step4_label") == "success":
        return "SA"
    if row.get("step4_label") == "semantic":
        return "semantic"
    return "FA"

if __name__ == "__main__":
    dfs = []
    for cond, f in files.items():
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        df["condition"] = cond
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    print("rows", len(all_df), "qid", all_df["question_id"].nunique(), "(106 x 5)")

    all_df["final_label"] = all_df.apply(make_label, axis=1)

    minor = all_df[all_df["step4_label"].astype(str).str.contains(
        "rerun_error|cant_compare|no_gold", na=False)]
    print(minor.groupby("condition")["step4_label"].apply(
        lambda x: x.str.split(":").str[0].value_counts()))

    main_table = all_df.groupby(["condition", "hop_diameter", "final_label"]).size().unstack(fill_value=0)
    print(main_table)
    main_table.to_csv(os.path.join(DATA_DIR, "rq3_main_table.csv"))

    all_df["is_col_keyerror"] = all_df["step2_status"].astype(str).str.match(
        r"^call_error: '[\w]+'$", na=False)
    keyerr_table = all_df.groupby(["condition", "hop_diameter"])["is_col_keyerror"].agg(["sum", "count"])
    keyerr_table["rate"] = (keyerr_table["sum"] / keyerr_table["count"]).round(3)
    print(keyerr_table)
    keyerr_table.to_csv(os.path.join(DATA_DIR, "rq3_keyerror_mechanism.csv"))

    sa_rows = all_df[all_df["final_label"] == "SA"][
        ["condition", "question_id", "hop_diameter", "step3_overjoin_tag"]
    ]
    print(sa_rows.to_string(index=False))
    sa_rows.to_csv(os.path.join(DATA_DIR, "rq3_sa_cases.csv"), index=False)

    reached = all_df[all_df["step3_avoidance"] == False]
    disp_n = reached.groupby(["condition", "gold_has_disp"]).size().rename("n")
    disp_rate = reached.groupby(["condition", "gold_has_disp"])["final_label"].apply(
        lambda x: (x == "semantic").mean()
    ).rename("semantic_rate")
    disp_summary = pd.concat([disp_n, disp_rate], axis=1)
    print(disp_summary)
    disp_summary.to_csv(os.path.join(DATA_DIR, "rq3_disp_semantic_rate.csv"))

    ### keep hop same, compare only question with disp vs without disp, look more into result that came out about question2
    disp_hop_n = reached.groupby(["condition", "hop_diameter", "gold_has_disp"]).size().rename("n")
    disp_hop_rate = reached.groupby(["condition", "hop_diameter", "gold_has_disp"])["final_label"].apply(
        lambda x: (x == "semantic").mean()
    ).rename("semantic_rate")
    disp_hop_summary = pd.concat([disp_hop_n, disp_hop_rate], axis=1)
    print(disp_hop_summary)
    disp_hop_summary.to_csv(os.path.join(DATA_DIR, "rq3_disp_semantic_rate_by_hop.csv"))
