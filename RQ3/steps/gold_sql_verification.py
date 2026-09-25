import sqlite3
import json
import pandas as pd

# ===== path setting (local — based on ~/Schema_LLM_FE/RQ1/RQ1_data) =====
DB_PATH = '~/Schema_LLM_FE/RQ1/RQ1_data/financial.sqlite'
DEV_JSON = '~/Schema_LLM_FE/dev.json'
HOP_CSV = '~/Schema_LLM_FE/RQ1/RQ1_data/qwen_classified_v3.csv'

import os
DB_PATH = os.path.expanduser(DB_PATH)
DEV_JSON = os.path.expanduser(DEV_JSON)
HOP_CSV = os.path.expanduser(HOP_CSV)

# ===== pull only financial 106 from dev.json, secure gold SQL =====
with open(DEV_JSON) as f:
    dev = json.load(f)
dev_df = pd.DataFrame(dev)
fin_df = dev_df[dev_df['db_id'] == 'financial'][['question_id', 'question', 'SQL']]
fin_df = fin_df.rename(columns={'SQL': 'gold_sql'})
print("financial question count:", len(fin_df))

# ===== attach hop_diameter =====
hop_df = pd.read_csv(HOP_CSV)[['question_id', 'hop_diameter']]
df = fin_df.merge(hop_df, on='question_id', how='inner')
print("after merge:", len(df), "/ 106")  # if not 106, some row's question_id didn't match

# ===== pull sample by hop_diameter bracket (5 per bracket, random) =====
N_PER_HOP = 5
samples = []
for h in sorted(df['hop_diameter'].unique()):
    sub = df[df['hop_diameter'] == h]
    n = min(N_PER_HOP, len(sub))
    samples.append(sub.sample(n=n, random_state=42))
sample_df = pd.concat(samples).reset_index(drop=True)
print(f"\ntotal {len(sample_df)} sample (max {N_PER_HOP} per hop)")

# ===== actually run gold SQL =====
con = sqlite3.connect(DB_PATH)

def run_gold(sql):
    # run and return result/error, cut to only 20 row for preview purpose
    try:
        res = pd.read_sql_query(sql, con)
        return {'exec_ok': True, 'n_rows': len(res), 'preview': res.head(5).to_dict('records'), 'error': None}
    except Exception as e:
        return {'exec_ok': False, 'n_rows': None, 'preview': None, 'error': str(e)}

results = []
for i, row in sample_df.iterrows():
    r = run_gold(row['gold_sql'])
    results.append({
        'question_id': row['question_id'],
        'question': row['question'],
        'hop_diameter': row['hop_diameter'],
        'gold_sql': row['gold_sql'],
        **r
    })
    print(f"{i+1}/{len(sample_df)} hop={row['hop_diameter']} exec_ok={r['exec_ok']}")

con.close()

# ===== save result — for eyeball check =====
out_df = pd.DataFrame(results)
out_df.to_csv(os.path.expanduser('~/Schema_LLM_FE/RQ3/gold_sql_verification_sample.csv'), index=False)

# case where execution itself failed (obvious error)
n_exec_fail = (~out_df['exec_ok']).sum()
print(f"\nexec fail: {n_exec_fail}/{len(out_df)}")

# empty result(0 row) — need to check by hand whether this is weird for the question's intent
n_empty = (out_df['n_rows'] == 0).sum()
print(f"0 row result: {n_empty}/{len(out_df)}")

print("\n-> open gold_sql_verification_sample.csv and put question/gold_sql/preview side by side")
print("   and check one by one by hand whether result match question intent")
