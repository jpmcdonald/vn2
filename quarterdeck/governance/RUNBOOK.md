# Runbook — VN2

## Setup (from scratch)

1. **Clone** the repository and `cd` into the project root.
2. **Python 3.11+** — verify `python --version`.
3. **Install** — recommended: `uv sync` or `pip install -e ".[dev]"` from root (see `pyproject.toml`).
4. **Raw data** — place competition CSVs under `data/raw/` per `README.md` / `DATA_CONTRACT.md`.
5. **Config paths** — edit `configs/forecast.yaml` `paths.*` if defaults point to another machine (known issue: absolute paths in repo).
6. **Environment** — optional: `./activate.sh` or project venv per team practice.

## Run procedures (primary entrypoints)

### Imputation (before some training paths)

```bash
uv run go impute --config configs/uncertainty.yaml --n-neighbors 20
```

**Inputs:** Raw + config. **Outputs:** Processed imputation artifacts (paths in config). **Verify:** Logs show completion; expected files exist under `data/processed/` (if not gitignored locally).

### Forecast training

```bash
uv run go forecast --config configs/forecast.yaml
```

**Inputs:** Processed demand, config. **Outputs:** Checkpoints under `paths.checkpoints` in config. **Verify:** Spot-check `models/checkpoints_h3/<model>/<store>_<product>/fold_*.pkl` count.

### Model evaluation (SIP)

```bash
uv run go eval-models --checkpoint-dir models/checkpoints_h3 --use-sip-optimization --out-suffix v4_sip --n-jobs 4 --resume
```

**Inputs:** Checkpoints, `demand_imputed.parquet` or fallback, optional `state.parquet`. **Outputs:** `models/results/eval_folds_v4_sip.parquet` (after full run), progress JSON. **Verify:** Row count ≈ models × SKUs × folds; column `sip_realized_cost_w2` present when SIP enabled.

### Full L3 backtest with selector

```bash
uv run python scripts/full_L3_simulation.py \
  --selector-map reports/dynamic_selector/static_composite_selector.parquet \
  --max-weeks 8 --service-level 0.833 \
  --checkpoints-dir models/checkpoints_h3 \
  --output-dir reports/backtest_L3
```

**Inputs:** Initial state CSV, sales under `data/raw`, selector parquet, checkpoints. **Outputs:** `full_simulation_results.csv`, optional `sku_detail.parquet`. **Verify:** Console “Total Cost”; holding + shortage sum to total.

### Selector comparison

```bash
uv run python scripts/compare_selector_metrics.py --timeout 3600
```

**Inputs:** `models/results/eval_folds_v4_sip.parquet`, composite selector path, checkpoints. **Outputs:** `cost_based_selector.parquet`, `comparison_summary.md` under `reports/selector_comparison/`. **Verify:** Both selector maps non-empty (599 rows); summary table populated.

### Unit tests

```bash
pytest test/
```

**Verify:** Exit code 0.

## Reproduction (given a manifest)

When P3 manifest is available:

1. Checkout recorded **git SHA**.
2. Restore or symlink **inputs** with matching hashes (raw data, checkpoints).
3. Run the **same command line** from manifest.
4. Compare **output hashes** and **total cost** within tolerance (define tolerance per team — not standardized in repo).

**Gap:** Standard manifest schema not yet in repo — use ad-hoc JSON until defined.

## Troubleshooting

| Symptom | Likely cause | Mitigation |
|---------|--------------|------------|
| `FileNotFoundError` on demand parquet | Missing imputation or wrong path | Run `go impute` or fix `cli.py` paths / use `demand_imputed.parquet` fallback |
| `PermissionError` / loky / semaphores | Sandbox or CI limits | `--n-jobs 1` or full OS permissions |
| Empty selector (0 SKUs) | Wrong fold-window logic or partial eval_folds | Ensure full eval_folds; verify `model_selector` last-N-fold fix |
| `sysctl` error on eval-models | Non-macOS host | Pass `--n-jobs` explicitly |
| Cost numbers not comparable | Mixed SL / eval cost bug | Confirm `eval_costs` vs order costs in `full_L3_simulation` |

## Definition of done (major steps)

| Step | Done when |
|------|-----------|
| Ingest | Interim files created; no ingest errors |
| Impute | Downstream `demand_imputed*.parquet` readable by training |
| Train | Checkpoints exist for all enabled models × SKUs (per training logs) |
| Eval | Consolidated `eval_folds_*.parquet` exists; aggregates match sum of parts |
| Backtest | Simulation completes 8 weeks; totals printed; CSV written |
| Selector comparison | Summary markdown lists both selectors and cost delta |

## Cross-references

- `MANIFEST.yaml` — full entrypoint list  
- `DATA_CONTRACT.md` — schema and paths  
- `ASSUMPTION_REGISTRY.md` — preflight checks  
