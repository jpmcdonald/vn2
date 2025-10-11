## Imputation runbook (stockout-censored demand)

This runbook documents how to run the stockout imputation at scale, with sizing, logging, and checkpoint/resume strategy.

### Data inputs

- `data/processed/demand_long.parquet`
- `data/processed/surd_transforms.parquet` (optional; defaults to `log` when absent)
- Quantile levels: from `configs/uncertainty.yaml` (`sip.quantiles`)

### Command

```bash
# inside tmux
source V2env/bin/activate
python -m vn2.cli impute \
  --config configs/uncertainty.yaml \
  --n-neighbors 20 \
  --n-jobs -1
```

Adjust `--n-jobs` per the resource profile in operations guide.

### Logging

- Redirect stdout/stderr to `logs/impute-<date>.log`
- Write heartbeat every 60s to `logs/impute_heartbeat.csv` with: timestamp, items_total, items_done, items_failed, eta_minutes

### Checkpoint/resume design (to implement)

We checkpoint at the unit of work = stockout triplet `(Store, Product, week)`.

Artifacts:

- `artifacts/impute/manifest.parquet`
  - columns: `Store:int`, `Product:int`, `week:int`, `status:str` (done|failed), `duration_s:float`, `ts:str`
- `artifacts/impute/sips.parquet` (append-only, partitioned)
  - columns: `Store`, `Product`, `week`, `quantile:float64`, `value:float64`
  - write mode: append; idempotent by deduping on (Store, Product, week, quantile)

Control flags:

- `--checkpoint-dir artifacts/impute/`
- `--checkpoint-interval 60` (seconds between manifest flushes)
- `--resume` (skip keys present in manifest with `status == 'done'`)

Resume algorithm:

1. Load `manifest.parquet` if present; build a set of completed keys
2. Enumerate stockouts; skip keys in completed set
3. For each key, attempt imputation; on success append SIP rows and record `done` in manifest; on failure record `failed`
4. Flush manifest every `checkpoint-interval` seconds and at job end

### Sizing guidance

- Inputs: ~94k rows, ~10.5k stockouts (~11%)
- Typical full run: 6–10 hours at `--n-jobs -1` on 12 cores; memory ~3–4 GB for workers
- Pilot slice: 2 stores × ~8 recent weeks; extrapolate ETA (items/time × items_total)

### Outputs

- `data/processed/demand_imputed.parquet` (training frame with imputed medians)
- `data/processed/imputed_sips.parquet` (per-stockout SIPs)
- `data/processed/imputation_summary.csv`

### Validation

- Spot-check a few imputed SIPs for monotonicity and non-negativity
- Compare summary coverage vs expectations; investigate extreme lifts
