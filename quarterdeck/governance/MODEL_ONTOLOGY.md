# Model Ontology — VN2

Governance view of **forecast model kinds**, their relationships, data contracts, and downstream consumers. Checkpoint directory names (`model_name`) are the **canonical IDs** used in eval, selectors, and `full_L3_simulation`.

**Related:** `configs/forecast.yaml` (training toggles), `docs/paper/revised_paper.md` §3.2, `scripts/compute_bias_analysis.py` `MODELS` list.

---

## 1. Taxonomy (high level)

| Class | Description | Examples |
|-------|-------------|----------|
| **A. Baseline / naive** | Simple seasonal or structural baselines | `seasonal_naive` |
| **B. Intermittent / classical** | Croston-style and count models (often disabled in default config) | `croston_*`, `zip`, `zinb` |
| **C. Gradient boosting quantile** | LightGBM with pinball loss on multiple quantiles | `lightgbm_quantile`, `lightgbm_surd` |
| **D. SLURP bootstrap** | k-NN conditional resampling → empirical quantiles | `slurp_bootstrap`, `slurp_stockout_aware`, `slurp_surd`, `slurp_surd_stockout_aware` |
| **E. Deep autoregressive** | Neural sequence model → samples → quantiles (separate training scripts; not all entries in `forecast.yaml`) | `deepar`, `deepar_surd`, `deepar_gluonts` variants |
| **F. Other parametric / linear** | Linear quantile, NGBoost, GLM (often disabled) | `linear_quantile`, `ngboost`, `glm_poisson`, `glm_negbin` |
| **G. Challenger / experimental** | Point or forest models for Jensen / ablation studies | `lightgbm_point`, `qrf`, `knn_profile` |

---

## 2. Relationships (`variant_of`, `extends`)

```
slurp_bootstrap
  ├── slurp_stockout_aware      (adds stockout_aware censoring handling)
  ├── slurp_surd                (+ SURD variance-stabilizing transform per series)
  └── slurp_surd_stockout_aware (+ both)

lightgbm_quantile
  └── lightgbm_surd             (+ use_surd: true)

deepar
  ├── deepar_surd               (+ SURD; per paper / bias analysis)
  └── deepar_gluonts            (naming variant from GluonTS training path — verify checkpoints dir)
```

**Orthogonal axes (not subclasses):**

- **SURD** — Per-SKU transform wrapper; pairs with LightGBM, SLURP, DeepAR families.
- **Stockout-aware** — SLURP-only flag in config; affects neighbor/target construction.

---

## 3. Model registry (canonical `model_name`)

Columns: **ID** = checkpoint subdirectory name. **Enabled** reflects `configs/forecast.yaml` as of ontology authoring — **verify before runs**.

| ID | Class | Training data (typical) | Quantile output | Enabled (default yaml) | Notes |
|----|-------|-------------------------|-----------------|------------------------|--------|
| `seasonal_naive` | A | Long history / seasonal decomposition | Yes (13 levels) | false | High bias risk on sparse series (see paper §4.2) |
| `croston_classic` | B | Intermittent pipeline | Per model | false | |
| `croston_sba` | B | | | false | |
| `croston_tsb` | B | | | false | |
| `zip` | B | Count | | false | |
| `zinb` | B | Count | | false | |
| `lightgbm_quantile` | C | Imputed / winsorized demand | Yes, h=1..3 | true | |
| `lightgbm_surd` | C | Same + SURD transform space | Yes | true | `use_surd: true` |
| `lightgbm_point` | G | Point head for challengers | No / point only | false | Jensen study |
| `slurp_bootstrap` | D | Raw demand + `in_stock` | Yes | true | |
| `slurp_stockout_aware` | D | Raw + censoring-aware | Yes | true | |
| `slurp_surd` | D | Raw + SURD | Yes | true | Identity transform bug was fixed per paper §5.3 |
| `slurp_surd_stockout_aware` | D | Raw + SURD + censoring | Yes | true | |
| `ets` | B | statsmodels ETS | | false | |
| `linear_quantile` | F | | | false | |
| `ngboost` | F | | | false | |
| `knn_profile` | G | | | false | |
| `qrf` | G | | | false | |
| `glm_poisson` | F | | | false | TODO pipeline per yaml |
| `glm_negbin` | F | | | false | TODO pipeline per yaml |
| `deepar` | E | Pre-competition window / long series | Yes (from samples) | **not in forecast.yaml** | Often trained via `scripts/train_deepar.py` |
| `deepar_surd` | E | Transformed space | Yes | **not in forecast.yaml** | Listed in `compute_bias_analysis.MODELS` |
| `deepar_gluonts` | E | | | **inferred from checkpoints** | Appears in eval/selector outputs when present |

**Could not determine from static config alone:** Every checkpoint name that exists only on disk; register new IDs here when adding models.

---

## 4. Output contract (checkpoints)

All density models in the main path are expected to produce **pickle** files:

- **Path:** `{checkpoints_dir}/{model_name}/{store}_{product}/fold_{k}.pkl`
- **Payload:** Dict with key `quantiles`: `pandas.DataFrame`, index = horizon **1, 2, 3**, columns = quantile levels (see `configs/forecast.yaml` `quantiles` list, typically 0.01–0.99).

**Consumers:** `vn2.analyze.model_eval.evaluate_one`, `scripts/full_L3_simulation.load_quantiles`, `scripts/compute_bias_analysis.load_checkpoint_quantiles`.

---

## 5. Downstream consumers

| Consumer | Uses |
|----------|------|
| `go eval-models` / `run_evaluation` | All checkpoint models under `--checkpoint-dir`; fold tasks |
| `go forecast` | Only `enabled: true` models in `forecast.yaml` |
| `model_selector.select_per_sku_from_folds` | Rows in `eval_folds` keyed by `model_name` |
| `build_dynamic_selector.py` | Bias detail parquet keyed by `model` column (must align with checkpoint names) |
| `full_L3_simulation` | Selector map `model_name` must match checkpoint subdirectory |
| `compute_bias_analysis` | Fixed `MODELS` list — **must be updated** when adding models to comparative bias reports |

---

## 6. Selection / ensemble context

- **Per-SKU selector** chooses among **IDs present in eval_folds and checkpoints**.
- **Composite metric** (pinball@CF × Wasserstein) and **cost-based** metric (`sip_realized_cost_w2`) are defined over the same `model_name` grain — see `docs/RECOMMENDED_NEXT_STEPS_2026-03-12.md` and selector scripts.

---

## 7. Governance rules

1. **New model** → add row to §3; update `forecast.yaml` if trainable via `go forecast`; update `compute_bias_analysis.MODELS` if part of standard bias suite; register in `MANIFEST.yaml` entrypoints if new training script is blessed.
2. **Renaming a model** → breaking change for selector maps and eval history; requires migration note.
3. **SURD + stockout** combinations are the only **multi-axis** products in the SLURP line — document new combinations the same way.

---

## Cross-references

- `DATA_CONTRACT.md` — checkpoint path layout  
- `ASSUMPTION_REGISTRY.md` — ASM-004, ASM-010, ASM-018  
- `MANIFEST.yaml` — `go forecast`, `go eval-models`  
- `RUNBOOK.md` — training and eval procedures  
