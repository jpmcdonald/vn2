# Reconciliation & Validation — VN2

## Reconciliation identities

1. **Week simulation cost** — For `full_L3_simulation`, for each week row: `total_cost ≈ holding_cost + shortage_cost` (floating rounding only). Source: `scripts/full_L3_simulation.py` results DataFrame columns.
2. **Portfolio total** — `sum(weekly total_cost)` equals printed “Total Cost” in summary. **Check:** `results['total_cost'].sum()`.
3. **Eval fold grain** — One row per `(model_name, store, product, fold_idx)` before aggregation; after `aggregate_results`, cost-like columns summed at SKU–model grain should match sum of fold-level rows for that subset. **Check:** `groupby(...).sum()` consistency.
4. **Selector coverage** — Static selector map for full backtest should have **599** rows if every competition SKU is assigned; otherwise `default_model` fills gaps — document missing keys. **Check:** `len(selector_df)` vs unique keys in initial state.
5. **Integer cost multiples** — With integerized units and cu=1, co=0.2, total costs should be representable as exact decimal increments (see `EVALUATION_V3_SUMMARY.md`). **Check:** `(total * 5).round()` integer for 0.2 granularity (heuristic).

## Validation checks (post-step)

### After eval-models

```text
Pseudocode:
  df = read_parquet("models/results/eval_folds_{suffix}.parquet")
  assert {"store","product","model_name","fold_idx"}.issubset(df.columns)
  assert df.groupby(["store","product","model_name"]).size().max() <= max_folds_expected
  if use_sip:
    assert "sip_realized_cost_w2" in df.columns
  assert df["sip_realized_cost_w2"].notna().mean() > threshold  # tune threshold
```

### After selector build

```text
Pseudocode:
  sel = read_parquet(".../selector.parquet")
  assert {"store","product","model_name"}.issubset(sel.columns)
  assert sel.duplicated(subset=["store","product"]).sum() == 0
```

### After full_L3_simulation

```text
Pseudocode:
  r = read_csv(".../full_simulation_results.csv")
  assert (r["holding_cost"] + r["shortage_cost"] - r["total_cost"]).abs().max() < 1e-6
```

## Known data quality limits (boundaries, not bugs)

- **Censored demand** — True demand unknown when stockout; imputation is model-based, not ground truth.
- **Sparse / intermittent SKUs** — High variance in fold-level metrics; aggregate stability requires sufficient fold count.
- **Checkpoint gaps** — Missing pickle → task skipped or default model; totals are **conditional** on available forecasts.
- **Gitignored artifacts** — Reproducibility depends on external storage of parquet/pickle; repo alone may not reproduce numbers.

## Variance localization

When a metric diverges from expectation:

1. **Git / config** — Confirm SHA and `forecast.yaml` paths match reference run.
2. **Input data** — Diff raw file checksums for the same calendar weeks.
3. **Checkpoint subset** — Pick one SKU where cost differs; compare `fold_k.pkl` quantiles to reference.
4. **Selector** — Diff selector maps; count rows per `model_name`.
5. **Eval vs sim** — `sip_realized_cost_w2` is fold-local SIP metric; **8-week `full_L3_simulation` cost is not guaranteed to equal** sum of eval fold costs (different state dynamics). Treat as **related but not identical** — localize which layer changed.

## Cross-references

- `ASSUMPTION_REGISTRY.md` — ASM-001, ASM-009, ASM-011, ASM-021  
- `DATA_CONTRACT.md` — artifact locations  
- `POLICY.md` — P7 evidence, P9 cost consistency  
