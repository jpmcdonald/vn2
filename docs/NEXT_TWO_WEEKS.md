# Next Two Weeks: Action List

Concrete backtest and improvement steps following the post-competition review. See [backtesting_against_competition.md](backtesting_against_competition.md) and [HOW_WE_BEAT_THE_WINNER.md](HOW_WE_BEAT_THE_WINNER.md) for context.

---

## Week 1

| Task | Action |
|------|--------|
| **Finalize data** | Ensure Week 1–8 Sales CSVs are in `data/raw/` (see [DATA_INVENTORY.md](DATA_INVENTORY.md)). Add state files if available. |
| **Build demand_long** | Run `python scripts/build_demand_long.py --raw-dir data/raw --out data/processed/demand_long.parquet` so training and eval scripts have long-format demand. |
| **Update docs** | Confirm [backtesting_against_competition.md](backtesting_against_competition.md) and [DATA_INVENTORY.md](DATA_INVENTORY.md) match repo (status, file list). |
| **Leaderboard** | Use `parse_cumulative_leaderboard()` ([src/vn2/competition/leaderboard_parser.py](../src/vn2/competition/leaderboard_parser.py)) on `data/raw/leaderboards/cumulative-leaderboard.txt` to extract our cost, benchmark cost, winner cost; document or save to a small table. |
| **Full 8-week backtest** | Run `python scripts/full_L3_simulation.py --max-weeks 8` (uses correct fold per decision week, no leakage); record realized 8-week cost. |
| **Benchmark 8-week cost** | Run `python scripts/run_rolling_benchmark_8week.py --raw-dir data/raw` (rolling seasonal MA + order-up-to, no leakage); record benchmark 8-week cost. |
| **Compare to top 20%** | Run `python scripts/compare_8week_results.py --our-cost <cost> --benchmark-cost <cost>` to compare against winner and top-20% threshold from [cumulative-leaderboard.txt](data/raw/leaderboards/cumulative-leaderboard.txt). |

---

## Week 2

| Task | Action |
|------|--------|
| **Forecast error report** | Run [scripts/analyze_forecast_metrics.py](../scripts/analyze_forecast_metrics.py) and evaluation (pinball, CRPS, cost-based) per model/selector on holdout; fill [FORECAST_ERROR_SUMMARY.md](FORECAST_ERROR_SUMMARY.md). |
| **Benchmark vs us vs winner** | Use `scripts/compare_8week_results.py` for table (our cost, benchmark cost, winner, top-20% threshold); short write-up in [HOW_WE_BEAT_THE_WINNER.md](HOW_WE_BEAT_THE_WINNER.md) or a short report. |
| **Optional: agent forecasts** | One run of agent forecasts (from [notebooks/forecast_agents.ipynb](../notebooks/forecast_agents.ipynb)) through our cost sim for one week or a subset of SKUs to see if agent forecasts could have beaten us (cost-based). |
| **Lessons learned** | Finalize [LESSONS_LEARNED_MEETUP.md](LESSONS_LEARNED_MEETUP.md) for the data science meetup. |

---

## AI agents vs our pipeline

- **Evaluation plan**: See “AI agents vs our pipeline” in [HOW_WE_BEAT_THE_WINNER.md](HOW_WE_BEAT_THE_WINNER.md). Same problem (retail sales), different objective (MASE/RMSE vs inventory cost). For a fair comparison, plug agent forecasts into our SIP/newsvendor pipeline and compare 8-week cost, or compare our pinball/CRPS to agent MASE/RMSE on the same holdout.
- **Optional in Week 2**: Run agent forecasts through our sim for one week or a subset of SKUs.
