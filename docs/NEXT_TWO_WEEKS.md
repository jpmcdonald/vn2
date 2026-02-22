# Next Two Weeks: Action List

Concrete backtest and improvement steps following the post-competition review. See [backtesting_against_competition.md](backtesting_against_competition.md) and [WHY_NOT_BEAT_BENCHMARK.md](WHY_NOT_BEAT_BENCHMARK.md) for context.

---

## Week 1

| Task | Action |
|------|--------|
| **Finalize data** | Ensure Week 1–8 Sales CSVs are in `data/raw/` (see [DATA_INVENTORY.md](DATA_INVENTORY.md)). Add state files if available. |
| **Build demand_long** | Run `python scripts/build_demand_long.py --raw-dir data/raw --out data/processed/demand_long.parquet` so training and eval scripts have long-format demand. |
| **Update docs** | Confirm [backtesting_against_competition.md](backtesting_against_competition.md) and [DATA_INVENTORY.md](DATA_INVENTORY.md) match repo (status, file list). |
| **Leaderboard** | Use `parse_cumulative_leaderboard()` ([src/vn2/competition/leaderboard_parser.py](../src/vn2/competition/leaderboard_parser.py)) on `data/raw/leaderboards/cumulative-leaderboard.txt` to extract our cost, benchmark cost, winner cost; document or save to a small table. |
| **Full 8-week backtest** | Run full simulation with our submissions and actuals (e.g. `scripts/full_L3_simulation.py` with `--weeks 1 2 3 4 5 6 7 8` when data exists); record realized 8-week cost and compare to leaderboard. |
| **Benchmark 8-week cost** | Run official benchmark logic ([notebooks/Official_Benchmark.py](../notebooks/Official_Benchmark.py) or equivalent) to get benchmark orders; run competition sim (L=3) with same actuals; record benchmark 8-week cost. |

---

## Week 2

| Task | Action |
|------|--------|
| **Forecast error report** | Run [scripts/analyze_forecast_metrics.py](../scripts/analyze_forecast_metrics.py) and evaluation (pinball, CRPS, cost-based) per model/selector on holdout; fill [FORECAST_ERROR_SUMMARY.md](FORECAST_ERROR_SUMMARY.md). |
| **Benchmark vs us vs winner** | Produce one summary table: benchmark 8-week cost, our 8-week cost, winner cost; short write-up in [WHY_NOT_BEAT_BENCHMARK.md](WHY_NOT_BEAT_BENCHMARK.md) or a short report. |
| **Optional: agent forecasts** | One run of agent forecasts (from [notebooks/forecast_agents.ipynb](../notebooks/forecast_agents.ipynb)) through our cost sim for one week or a subset of SKUs to see if agent forecasts could have beaten us (cost-based). |
| **Lessons learned** | Finalize [LESSONS_LEARNED_MEETUP.md](LESSONS_LEARNED_MEETUP.md) for the data science meetup. |

---

## AI agents vs our pipeline

- **Evaluation plan**: See “AI agents vs our pipeline” in [WHY_NOT_BEAT_BENCHMARK.md](WHY_NOT_BEAT_BENCHMARK.md). Same problem (retail sales), different objective (MASE/RMSE vs inventory cost). For a fair comparison, plug agent forecasts into our SIP/newsvendor pipeline and compare 8-week cost, or compare our pinball/CRPS to agent MASE/RMSE on the same holdout.
- **Optional in Week 2**: Run agent forecasts through our sim for one week or a subset of SKUs.
