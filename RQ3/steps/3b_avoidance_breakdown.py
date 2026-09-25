# 3b - just look at step3 uniform/adaptive files, count how many dropped at
# each step, and check how many of step2's errors are KeyError kind
import os
import pandas as pd

ROOT = "/Users/baesubin/Schema_LLM_FE"
DATA_DIR = os.path.join(ROOT, "RQ3", "RQ3_data", "new")
os.makedirs(DATA_DIR, exist_ok=True)

files = [
    "qwen_uniform_step3.csv",
    "qwen_adaptive_step3.csv",
    "llama_uniform_step3.csv",
    "llama_adaptive_step3.csv",
]

if __name__ == "__main__":
    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        n = len(df)
        print(f, n)

        n1 = df["step1_status"].notna().sum()
        n2 = df["step2_status"].notna().sum()
        n3 = df["step3_avoidance"].notna().sum()
        print("step1 drop", n1, round(n1 / n, 3))
        print("step2 drop", n2, round(n2 / n, 3))
        print("step3 done", n3, round(n3 / n, 3))
        print("sum", n1 + n2 + n3, "expect", n)

        print(df["step1_status"].value_counts(dropna=True))

        err = df["step2_status"].dropna()
        print(err.value_counts().head(10))

        # just roughly catch KeyError pattern, not a perfect regex
        key_err = err[err.str.contains(r"call_error: '[A-Za-z0-9_]+'$", na=False)]
        if len(err) == 0:
            print("keyerr", 0)
        else:
            print("keyerr", len(key_err), len(err), round(len(key_err) / len(err), 3))
        print()