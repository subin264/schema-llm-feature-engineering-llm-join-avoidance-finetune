# Beyond the Boundary — Code & Data

Code and data for Subin Bae's bachelor's thesis (Gisma, September 2026) — this exact URL is cited in Section 4.4.5.


## Overview

On the financial domain of the BIRD dev set (106 questions), this measures whether
Qwen2.5-Coder-7B-Instruct and Llama-3.1-8B-Instruct avoid JOINs when reassembling a
normalized schema into pandas feature code, or fail on the executed values instead.

## Requirements

These are the minimum packages used across the pipeline (RQ1 → RQ2 → RQ3 → fine-tuning).

```
pip install pandas numpy scipy networkx sqlglot statsmodels transformers sentencepiece peft trl bitsandbytes anthropic
```

* pandas, numpy, scipy, networkx, sqlglot, statsmodels
* transformers, sentencepiece, peft, trl, bitsandbytes — model inference / LoRA
* anthropic — only for `4_finetuning/training_data_build/1_generate_and_verify.py`

Python 3.11+. `1_generate_and_verify.py` also needs an `ANTHROPIC_API_KEY` env var
(never hardcode it) and the full BIRD `dev_databases/` (~1.4GB, not included here —
download from the link below). It is optional: its output (`train_data.jsonl`) is
already included, so it does not need to be rerun to reproduce the paper numbers.

## Data

**Source**: BIRD (financial domain) — https://bird-bench.github.io. `data/financial.sqlite`
and `data/financial_hop_count_results.csv` are derived from it: the 106 financial-domain
questions with gold SQL parsed into a join graph and `hop_diameter` attached.

**Schema**: the financial DB has 8 tables. Each is read with SQLite `PRAGMA table_info`
and given to the model as name + column list only — no descriptions, data types, or FK
statements (what RQ1/RQ2 measure is JOIN steps, not domain knowledge).

```
- account(account_id, district_id, frequency, date)
- card(card_id, disp_id, type, issued)
- client(client_id, gender, birth_date, district_id)
- disp(disp_id, client_id, account_id, type)
- district(district_id, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16)
- loan(loan_id, account_id, date, amount, duration, payments, status)
- order(order_id, account_id, bank_to, account_to, amount, k_symbol)
- trans(trans_id, account_id, date, type, operation, amount, balance, k_symbol, bank, account)
```

## Key artifacts used in each stage

* RQ1 entry point: `RQ1/RQ1_final.py`
* RQ2 entry point: `RQ2/RQ2_final.py`
* RQ3 entry point: `RQ3/RQ3_baseline_reclassify.py` → `RQ3/RQ3_final.py`
* fine-tuning entry point: `4_finetuning/Finetuning_final.py`
* shared input data: `data/financial.sqlite`, `data/financial_hop_count_results.csv`, `data/database_description/`

Each group follows the same layout: `<group>_final.py` (reproduces the thesis result) +
`steps/` (process record) + `data/` (input csv).

### `steps/` — process record, not verified

Each group's `steps/` holds earlier/intermediate scripts kept for traceability (e.g. RQ1's
pre-execution avoidance classifier — the source of the "96 → 329" methodological pitfall in
the abstract). Unlike `*_final.py`, these were **not** re-run or path-fixed for this repo —
most still reference the original project's absolute paths and won't run as-is from here.

### `4_finetuning/`

* `training_data_build/` — how the 49-sample LoRA training set (Table 4.4–4.5) was built.
  `1_generate_and_verify.py` calls the Anthropic API and needs the BIRD dev databases (see
  Requirements/Data above); `2_build_jsonl.py` just merges the verified candidates into
  `train_data.jsonl` and is self-contained.
* `lora_adapters/` — trained LoRA adapter weights (`lora_qwen.zip` ~10MB, `lora_llama.zip`
  ~145MB, Git LFS — over GitHub's 100MB push limit for a normal commit).

## 1) RQ1 (hop count vs avoidance)

* run `RQ1/RQ1_final.py` to reproduce the chi-square and Cochran–Armitage trend test
* hop_diameter is computed by `RQ1/hop_count.py`
* expected key output: Table 5.2–5.3 numbers, printed to console

## 2) RQ2 (failure type by relationship complexity)

* run `RQ2/RQ2_final.py` to reproduce the has_disp chi-square test
* expected key output file: `RQ2/data/qwen_rq2_final.csv`, `llama_rq2_final.csv`

## 3) RQ3 (schema exposure conditions)

* run `RQ3/RQ3_baseline_reclassify.py` first (re-scores RQ1's baseline with the final classifier)
* then run `RQ3/RQ3_final.py` to reproduce Cochran's Q test
* expected key output: Table 5.1 (baseline), Table 5.6

## 4) fine-tuning

* run `4_finetuning/Finetuning_final.py` to reproduce the before/after comparison
* training data build (optional, not needed to reproduce paper numbers): `4_finetuning/training_data_build/`

## Key outputs

* RQ1/RQ2/RQ3 final tables: printed by each `*_final.py` script
* fine-tuning summary: `4_finetuning/data/summary_label_exec.csv`, `summary_hop_AA_pct_label_exec.csv`
* LoRA adapters: `4_finetuning/lora_adapters/lora_qwen.zip`, `lora_llama.zip`

## Table ↔ File Mapping

| Thesis table | Paper value | Source |
|---|---|---|
| Table 5.1 — Overall Label Distribution (n=530) | Qwen: AA 72 (13.6%), FA 423 (79.8%), semantic 35 (6.6%), SA 0 (0%) · Llama: AA 67 (12.6%), FA 385 (72.6%), semantic 70 (13.2%), SA 8 (1.5%) | output csv of `RQ3/RQ3_baseline_reclassify.py` (`qwen/llama_baseline_reclassified.csv`) |
| Table 5.2 — Avoidance Items by Hop | Qwen 0/8/4/1, Llama 0/3/5/0 (hop 0–3) | console output of `RQ1/RQ1_final.py` |
| Table 5.3 — Avoidance × Hop Test | Qwen χ²=2.49 p=0.477, trend z=−0.09 p=0.928 · Llama χ²=4.84 p=0.184, trend z=0.50 p=0.614 | console output of `RQ1/RQ1_final.py` (no file written) |
| Table 5.4 — Avoidance by disp Status | No disp: Qwen 11 (16.4%) / Llama 5 (7.5%) · Has disp: Qwen 2 (5.1%) / Llama 3 (7.7%) | `RQ2/data/qwen_rq2_final.csv`, `llama_rq2_final.csv` |
| Table 5.5 — Error type by disp | Qwen no 21 (7.5%) / has 14 (7.8%) · Llama no 60 (21.5%) / has 10 (5.7%) | same as above — **does not fully reproduce, see Known Gaps** |
| Table 5.6 — Question-Level Avoidance, Cochran's Q | Qwen 13/15/7, Q=4.00 p=0.135 · Llama 8/13/5, Q=4.67 p=0.097 | console output of `RQ3/RQ3_final.py` (no file written) |
| Section 5.0.4 — fine-tuning before/after | Llama baseline AA 66→9, SA 10→5 · Qwen adaptive SA up to 27 | `4_finetuning/data/summary_label_exec.csv`, `summary_hop_AA_pct_label_exec.csv` |
| Table 4.4 — Construction of the Training Set | 185 → 177 → 53 → 49 (hop2/hop3: 44/5) | `4_finetuning/training_data_build/train_data.jsonl` |
| Table 4.5 — Domain of the Final 49 Items | superhero 27, student_club 8, formula_1 7, rest 1–2 each, total 49 | same as above |

Every row above was verified by re-running the corresponding script against the numbers
printed in the thesis, except Table 5.5 (see Known Gaps).

## Known Gaps

RQ2 percentages in this repo (16.6% / 3.5%) are not the same as Table 5.5 (21.5% / 5.7%).
The direction is the same. Use Table 5.5 as the paper result.
The gap is likely from treating date columns as dates versus as text (thesis §6.1).
