# Quarterdeck Charter — VN2

## Mission

VN2 is a research and engineering codebase for the **VN2 Inventory Planning Challenge**: multi-SKU (599 store–product pairs) inventory planning under asymmetric costs (shortage vs holding), lead time, and censored demand from stockouts. The system delivers **quantile-based probabilistic forecasts** (SLURP, LightGBM, DeepAR, etc.), converts them to decision-relevant distributions (SIP / PMF), and runs **sequential inventory simulations and evaluations** so total realized cost can be minimized and compared to benchmarks, competition winners, and alternative policies. Business value: **decision-valid cost estimates** and reproducible evidence for density-aware optimization versus point-forecast + safety-stock workflows.

## Success criteria

1. **Reproducible backtest cost**: A named configuration (checkpoints dir, selector map, service level, max weeks) produces the same total holding + shortage cost when re-run on the same frozen inputs (raw sales, initial state, checkpoint pickles).
2. **Evaluation integrity**: Model evaluation separates **ordering policy costs** (e.g. service-level–adjusted) from **evaluation costs** (fixed cu/co) when comparing experiments; documented in `docs/paper/revised_paper.md` and `scripts/full_L3_simulation.py` patterns.
3. **Traceable artifacts**: Forecast checkpoints follow `{checkpoints_dir}/{model_name}/{store}_{product}/fold_{k}.pkl`; eval outputs land under `models/results/` with eval_folds naming conventions documented in `src/vn2/analyze/model_eval.py`.
4. **CLI discoverability**: Primary user actions are invocable via `go` (`vn2.cli:main`) with documented subcommands matching `README.md` and `MANIFEST.yaml`.
5. **Test signal**: `pytest test/` passes on the supported Python version (`requires-python >=3.11` in `pyproject.toml`).

## Scope boundaries

**In scope**

- Ingesting and transforming competition-style raw data under `data/raw`, `data/interim`, `data/processed`.
- Training and evaluating density forecasters; per-SKU model selection; ensemble and backtest harnesses.
- Simulation and optimization aligned with competition rules (costs, lead time as implemented in code — see Assumption Registry for L=2 vs L=3 documentation drift).
- Documentation, notebooks, and governance under `docs/` and `quarterdeck/governance/`.

**Out of scope**

- Production deployment, SLA guarantees, or multi-tenant operation (no such implementation in repo).
- Real-time API serving of forecasts (CLI and batch scripts only).
- Modification of competition source data integrity (raw truth is external; this repo consumes copies).
- Legal/compliance certification (no HIPAA/PCI modules present).

## Decision authority

All acceptance criteria for releases, approvals of assumptions in `ASSUMPTION_REGISTRY.md`, and scope changes to this charter require **human authorization**. Autonomous agents must not alter this file without explicit human sign-off.

## Constraints

| Constraint | Source |
|------------|--------|
| Python **≥ 3.11** | `pyproject.toml` |
| Key deps: pandas 2.x, numpy, scipy, scikit-learn, LightGBM, PyTorch, GluonTS, Lightning | `pyproject.toml` |
| Default CLI entrypoint **`go`** | `[project.scripts]` in `pyproject.toml` |
| Large artifacts **gitignored**: `reports/`, `models/checkpoints/`, `models/results/`, `*.parquet`, `*.pkl`, many `data/` subtrees | `.gitignore` |
| **Absolute paths** in `configs/forecast.yaml` (`paths.processed`, etc.) tie runs to a developer machine unless overridden | `configs/forecast.yaml` |
| Competition description in README cites **L=2** wording while post-competition simulators implement **L=3**; operators must align on which semantics apply per run | `README.md` vs `scripts/full_L3_simulation.py` |

**Regulatory:** None identified in codebase; MIT license.

## Related governance artifacts

- **Model ontology** (families, `model_name` IDs, checkpoint contract, consumers): [MODEL_ONTOLOGY.md](MODEL_ONTOLOGY.md)
