import os
import pandas as pd

DATA_DIR = "/Users/baesubin/Schema_LLM_FE/RO2/dtat"

def step3(n_merge, hop_required):
    """
    step3 core is check whether join was done less, or done more !
    into 3
    1. hop_required > n_merge : avoidance = True, overjoin = False
    2. hop_required == n_merge : no avoidance, no overjoin
    3. hop_required < n_merge : no avoidance, overjoin = True

    """
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
        (os.path.join(DATA_DIR, "q2_step2_qwen.csv"), "qwen"),
        (os.path.join(DATA_DIR, "q2_step2_llama.csv"), "llama"),
    ]

    for path, name in targets:
        df = pd.read_csv(path)
        avoid_list = []
        overjoin_list = []

        for i, row in df.iterrows():
            if pd.notna(row.get("step1_status")) or pd.notna(row.get("step2_status")):
                avoid_list.append(None)
                overjoin_list.append(None)
                continue
            r = step3(row["n_merge"], row["hop_diameter"])
            avoid_list.append(r["avoidance"])
            overjoin_list.append(r["overjoin"])

        df["step3_avoidance"] = avoid_list
        df["step3_overjoin_tag"] = overjoin_list
        print(name)
        print(df["step3_avoidance"].value_counts(dropna=False))
        print(df["step3_overjoin_tag"].value_counts(dropna=False))

        out_path = os.path.join(DATA_DIR, "q2_step3_" + name + "_n5.csv")
        df.to_csv(out_path, index=False)
        print(out_path)