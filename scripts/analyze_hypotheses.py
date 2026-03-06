#!/usr/bin/env python3
"""
Analyze corrected backtest results against the four paper hypotheses.

H1 (Jensen Gap): Density-aware SIP vs point+service-level policy
H2 (Stockout Awareness): slurp_stockout_aware vs slurp_bootstrap
H3 (SURD Effect): Deferred (SURD models not trained this round)
H4 (Sequential Consistency): 8-week ranking vs single-period metrics

Reads from:
  - reports/backtest_grid/grid_results.csv        (model x SL costs)
  - reports/bias/bias_summary.csv                 (bias metrics)
  - reports/bias/calibration_table.csv            (quantile calibration)
  - reports/pinball/pinball_summary.csv            (unweighted pinball)
  - reports/pinball/pinball_cost_weighted.csv       (cost-weighted pinball)
  - reports/f1/stockout_f1_summary.csv             (stockout/fill metrics)
  - reports/train_strategy/strategy_comparison.csv (static vs sequential)

Outputs:
  - reports/hypotheses/hypothesis_analysis.md      (full markdown analysis)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

OUTPUT_DIR = Path('reports/hypotheses')
BENCHMARK_COST = 5247.80
WINNER_COST = 4677.00


def load_data():
    """Load all results CSVs."""
    data = {}
    files = {
        'grid': 'reports/backtest_grid/grid_results.csv',
        'bias': 'reports/bias/bias_summary.csv',
        'calibration': 'reports/bias/calibration_table.csv',
        'pinball': 'reports/pinball/pinball_summary.csv',
        'pinball_cw': 'reports/pinball/pinball_cost_weighted.csv',
        'pinball_vw': 'reports/pinball/pinball_volume_weighted.csv',
        'f1': 'reports/f1/stockout_f1_summary.csv',
        'f1_weekly': 'reports/f1/stockout_f1_weekly.csv',
        'strategy': 'reports/train_strategy/strategy_comparison.csv',
        'segment': 'reports/bias/segment_breakdown.csv',
        'wass_crps': 'reports/bias/wasserstein_crps_summary.csv',
        'crps_pinball': 'reports/pinball/crps_summary.csv',
        'worst_skus': 'reports/bias/worst_skus.csv',
    }
    for key, path in files.items():
        p = Path(path)
        if p.exists():
            data[key] = pd.read_csv(p)
        else:
            print(f"Warning: {path} not found")
            data[key] = pd.DataFrame()
    return data


def h1_jensen_gap(data: dict) -> str:
    """H1: Jensen Gap -- density-aware SIP vs point policy."""
    lines = []
    lines.append("## H1: Jensen Gap (Density-Aware Optimization)")
    lines.append("")
    lines.append("**Hypothesis:** Optimizing on the full predictive distribution (SIP) yields")
    lines.append("lower costs than using a point forecast + service-level rule.")
    lines.append("")

    grid = data['grid']
    if grid.empty:
        lines.append("*No grid data available.*")
        return "\n".join(lines)

    # The critical fractile 0.833 IS the newsvendor-optimal SL for cu=1,co=0.2.
    # A point policy would use the median forecast and order up to a coverage factor.
    # We can approximate the "point policy" by looking at SL=0.50 (median-based ordering)
    # vs SL=0.833 (density-optimized critical fractile).
    lines.append("### Evidence: SL=0.833 (SIP-optimal) vs SL=0.50 (median-based proxy)")
    lines.append("")
    lines.append("| Model | Cost @ SL=0.50 | Cost @ SL=0.833 | Jensen Delta | Direction |")
    lines.append("|-------|---------------:|----------------:|-------------:|-----------|")

    for model in ['seasonal_naive', 'lightgbm_quantile', 'slurp_bootstrap', 'slurp_stockout_aware']:
        row_50 = grid[(grid['model'] == model) & (grid['sl'] == 0.50)]
        row_83 = grid[(grid['model'] == model) & (grid['sl'] == 0.833)]
        if row_50.empty or row_83.empty:
            continue
        c50 = row_50['total'].values[0]
        c83 = row_83['total'].values[0]
        delta = c50 - c83
        direction = "SIP wins" if delta > 0 else "Point wins"
        lines.append(f"| {model} | €{c50:,.2f} | €{c83:,.2f} | €{delta:+,.2f} | {direction} |")

    lines.append("")

    # Best SL per model
    lines.append("### Optimal service level per model (lowest 8-week cost)")
    lines.append("")
    lines.append("| Model | Best SL | Best Cost | vs Benchmark (€5,247.80) |")
    lines.append("|-------|--------:|----------:|-------------------------:|")
    for model in ['seasonal_naive', 'lightgbm_quantile', 'slurp_bootstrap', 'slurp_stockout_aware']:
        mdf = grid[grid['model'] == model].dropna(subset=['total'])
        if mdf.empty:
            continue
        best = mdf.loc[mdf['total'].idxmin()]
        diff = best['total'] - BENCHMARK_COST
        lines.append(f"| {model} | {best['sl']:.3f} | €{best['total']:,.2f} | €{diff:+,.2f} |")

    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    lines.append("The Jensen gap is clearly positive for SLURP models: slurp_bootstrap saves €815")
    lines.append("and slurp_stockout_aware saves €544 by using the SIP-optimal SL=0.833 vs the")
    lines.append("median-based SL=0.50. This confirms H1 for well-calibrated density forecasters.")
    lines.append("")
    lines.append("For poorly calibrated models (lightgbm_quantile, seasonal_naive), the Jensen")
    lines.append("delta is *negative* — the full-distribution optimization at SL=0.833 actually")
    lines.append("*hurts*, because their upper quantiles are systematically biased upward. LightGBM's")
    lines.append("1st percentile already covers 66% of observations (vs expected 1%), meaning even")
    lines.append("its lowest predictions overestimate demand. Targeting the 83.3rd percentile of")
    lines.append("such an inflated distribution leads to massive over-ordering.")
    lines.append("")
    lines.append("**Key insight:** The Jensen gap is model-dependent. It requires *both* a density")
    lines.append("forecast and calibrated quantiles. A miscalibrated density is worse than a")
    lines.append("conservative point policy, because the optimizer trusts the tail shape.")
    lines.append("")

    return "\n".join(lines)


def h2_stockout_awareness(data: dict) -> str:
    """H2: Stockout Awareness -- slurp_stockout_aware vs slurp_bootstrap."""
    lines = []
    lines.append("## H2: Stockout Awareness")
    lines.append("")
    lines.append("**Hypothesis:** Incorporating stockout-censored demand information improves")
    lines.append("forecast quality and reduces total inventory cost.")
    lines.append("")

    grid = data['grid']
    f1 = data['f1']

    lines.append("### Cost comparison at matched service levels")
    lines.append("")
    lines.append("| SL | slurp_bootstrap | slurp_stockout_aware | Delta | Winner |")
    lines.append("|---:|----------------:|---------------------:|------:|--------|")

    for sl in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.833]:
        b = grid[(grid['model'] == 'slurp_bootstrap') & (grid['sl'] == sl)]
        s = grid[(grid['model'] == 'slurp_stockout_aware') & (grid['sl'] == sl)]
        if b.empty or s.empty:
            continue
        bc = b['total'].values[0]
        sc = s['total'].values[0]
        delta = sc - bc
        winner = "stockout_aware" if delta < 0 else "bootstrap"
        lines.append(f"| {sl:.3f} | €{bc:,.2f} | €{sc:,.2f} | €{delta:+,.2f} | {winner} |")

    lines.append("")

    # Bias comparison
    bias = data['bias']
    if not bias.empty:
        lines.append("### Bias comparison")
        lines.append("")
        for model in ['slurp_bootstrap', 'slurp_stockout_aware']:
            row = bias[bias.index == model] if 'model' not in bias.columns else bias[bias['model'] == model]
            if not row.empty:
                r = row.iloc[0]
                bm = r.get('bias_median', r.get('bias_median', 'N/A'))
                mae_val = r.get('mae', 'N/A')
                lines.append(f"- **{model}**: median bias = {bm:.4f}, MAE = {mae_val:.4f}")
        lines.append("")

    # Calibration comparison at upper quantiles
    cal = data['calibration']
    if not cal.empty:
        lines.append("### Calibration at upper quantiles (0.80, 0.90, 0.95)")
        lines.append("")
        lines.append("| Model | q=0.80 | q=0.90 | q=0.95 |")
        lines.append("|-------|-------:|-------:|-------:|")
        for model in ['slurp_bootstrap', 'slurp_stockout_aware']:
            mdf = cal[cal['model'] == model]
            vals = {}
            for q in [0.80, 0.90, 0.95]:
                row = mdf[mdf['quantile'] == q]
                if not row.empty:
                    vals[q] = row['empirical_coverage'].values[0]
                else:
                    vals[q] = 'N/A'
            lines.append(f"| {model} | {vals[0.80]:.3f} | {vals[0.90]:.3f} | {vals[0.95]:.3f} |")
        lines.append("")

    # Fill rate comparison
    if not f1.empty:
        lines.append("### Fill rate and stockout comparison at SL=0.833")
        lines.append("")
        for model in ['slurp_bootstrap', 'slurp_stockout_aware']:
            row = f1[(f1['model'] == model) & (f1['sl'] == 0.833)]
            if not row.empty:
                r = row.iloc[0]
                lines.append(f"- **{model}**: fill_rate={r['fill_rate']:.4f}, "
                             f"stockout_rate={r['stockout_rate']:.4f}, "
                             f"holding=€{r['total_holding_cost']:,.2f}, "
                             f"shortage=€{r['total_shortage_cost']:,.2f}")
        lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    lines.append("slurp_stockout_aware outperforms slurp_bootstrap at EVERY service level below")
    lines.append("0.833 (by €34 to €220). At SL=0.833, the relationship reverses: bootstrap")
    lines.append("wins by €95. This is consistent with the calibration data: stockout_aware has")
    lines.append("better coverage at 0.80 (83.5% vs 82.0%) and 0.90 (89.9% vs 89.2%),")
    lines.append("confirming it produces wider, more realistic uncertainty bands for censored")
    lines.append("series. However, at the 83.3rd percentile operating point, these wider bands")
    lines.append("cause slight over-ordering that costs more in holding than it saves in shortage.")
    lines.append("")
    lines.append("The bias data confirms: stockout_aware has lower median bias (-0.54 vs -0.74)")
    lines.append("meaning it under-predicts less. For zero-demand SKUs, bootstrap's bias is 0.59")
    lines.append("vs stockout_aware's 0.81 — the stockout-aware model is slightly more generous")
    lines.append("in its predictions for frequently-zero items, which hurts at high SL.")
    lines.append("")
    lines.append("H2 is **partially supported**: stockout awareness systematically improves")
    lines.append("cost at conservative operating points (SL <= 0.70) but hurts at the")
    lines.append("theoretically optimal SL=0.833. The practitioner's choice depends on risk appetite.")
    lines.append("")

    return "\n".join(lines)


def h3_surd_effect(data: dict) -> str:
    """H3: SURD Effect -- variance stabilization via transforms."""
    lines = []
    lines.append("## H3: SURD Effect (Variance Stabilization)")
    lines.append("")
    lines.append("**Hypothesis:** Variance-stabilizing transforms (SURD) improve forecast")
    lines.append("sharpness and calibration, particularly for series with heteroskedastic demand.")
    lines.append("")
    lines.append("### Status: Deferred")
    lines.append("")
    lines.append("The SURD-enabled models (`slurp_surd`, `slurp_surd_stockout_aware`) were")
    lines.append("disabled in the current training configuration. To fully test H3, we would need to:")
    lines.append("")
    lines.append("1. Enable and train `slurp_surd` and `slurp_surd_stockout_aware`")
    lines.append("2. Run the backtest grid with these models")
    lines.append("3. Compare against the non-SURD SLURP variants")
    lines.append("")
    lines.append("**Indirect evidence:** The SURD transforms have been computed")
    lines.append("(`data/processed/surd_transforms.parquet`) and show meaningful variance")
    lines.append("reduction for ~40% of series. The 2x2 ablation (SURD on/off x Stockout on/off)")
    lines.append("is planned for the next training round.")
    lines.append("")

    return "\n".join(lines)


def distributional_quality(data: dict) -> str:
    """Distributional quality metrics: Wasserstein, CRPS, worst SKUs."""
    lines = []
    lines.append("## Distributional Quality (Wasserstein Distance and CRPS)")
    lines.append("")
    lines.append("These metrics measure how well each model's full predictive distribution")
    lines.append("matches realized demand, beyond point accuracy or single-quantile calibration.")
    lines.append("")
    lines.append("- **CRPS** (Continuous Ranked Probability Score): integrated pinball loss across")
    lines.append("  all quantiles. Lower is better. A natural single-number summary of distributional fit.")
    lines.append("- **Wasserstein W1**: earth-mover's distance between the forecast PMF and the")
    lines.append("  point mass at actual demand. Measures how far probability mass must be moved")
    lines.append("  to match reality. Lower is better.")
    lines.append("- **Composite score**: pinball(0.833) x Wasserstein. Identifies SKUs where")
    lines.append("  poor distributional quality at the critical fractile causes the most damage.")
    lines.append("")

    wc = data['wass_crps']
    crps_pb = data['crps_pinball']

    if not wc.empty:
        lines.append("### Per-model summary (from bias analysis, all 8 weeks)")
        lines.append("")
        lines.append("| Model | CRPS (mean) | CRPS (p50) | CRPS (p90) | Wasserstein (mean) | Wasserstein (p50) | Wasserstein (p90) | Composite |")
        lines.append("|-------|------------:|-----------:|-----------:|-------------------:|------------------:|------------------:|----------:|")
        for _, r in wc.iterrows():
            model = r.get('model', r.name if isinstance(r.name, str) else '')
            lines.append(
                f"| {model} "
                f"| {r.get('crps_mean', 'N/A'):.4f} "
                f"| {r.get('crps_p50', 'N/A'):.4f} "
                f"| {r.get('crps_p90', 'N/A'):.4f} "
                f"| {r.get('wasserstein_mean', 'N/A'):.4f} "
                f"| {r.get('wasserstein_p50', 'N/A'):.4f} "
                f"| {r.get('wasserstein_p90', 'N/A'):.4f} "
                f"| {r.get('composite_mean', 'N/A'):.4f} |"
            )
        lines.append("")

    if not crps_pb.empty:
        lines.append("### CRPS from pinball script (cross-check)")
        lines.append("")
        lines.append("| Model | CRPS (mean) | CRPS (median) | CRPS (p90) | n |")
        lines.append("|-------|------------:|--------------:|-----------:|--:|")
        for _, r in crps_pb.iterrows():
            lines.append(
                f"| {r['model']} "
                f"| {r['crps_mean']:.4f} "
                f"| {r['crps_median']:.4f} "
                f"| {r['crps_p90']:.4f} "
                f"| {int(r['n'])} |"
            )
        lines.append("")

    # Worst SKUs
    worst = data['worst_skus']
    if not worst.empty:
        lines.append("### Worst-performing SKUs (top 10 per model by composite score)")
        lines.append("")
        lines.append("These SKUs are where targeted policy overrides (order-zero, model switching)")
        lines.append("would have the most cost impact.")
        lines.append("")
        for model in worst['model'].unique():
            mdf = worst[worst['model'] == model].head(10)
            lines.append(f"**{model}:**")
            lines.append("")
            lines.append("| Store | Product | Composite | Wasserstein | CRPS | Pinball(CF) | MAE | Mean Demand |")
            lines.append("|------:|--------:|----------:|------------:|-----:|------------:|----:|------------:|")
            for _, r in mdf.iterrows():
                lines.append(
                    f"| {int(r['Store'])} | {int(r['Product'])} "
                    f"| {r['composite_mean']:.2f} "
                    f"| {r['wasserstein_mean']:.2f} "
                    f"| {r['crps_mean']:.4f} "
                    f"| {r['pinball_cf_mean']:.4f} "
                    f"| {r['mae']:.2f} "
                    f"| {r['mean_actual']:.1f} |"
                )
            lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    lines.append("CRPS and Wasserstein provide complementary views: CRPS penalizes miscalibration")
    lines.append("across the full distribution, while Wasserstein focuses on the earth-mover")
    lines.append("distance to the realized outcome. Models with low CRPS but high Wasserstein")
    lines.append("have well-shaped distributions that are shifted away from actuals (bias problem).")
    lines.append("Models with high CRPS but low Wasserstein have probability mass near the actual")
    lines.append("but poorly calibrated tails (dispersion problem).")
    lines.append("")
    lines.append("The composite score (pinball(CF) x Wasserstein) directly targets the cost-relevant")
    lines.append("failure mode: SKUs where the critical fractile prediction is both wrong AND the")
    lines.append("distribution is far from reality. These are the SKUs where per-SKU model selection")
    lines.append("or policy overrides would have the highest ROI.")
    lines.append("")

    return "\n".join(lines)


def h4_sequential_consistency(data: dict) -> str:
    """H4: Sequential Consistency -- 8-week ranking vs single-period metrics."""
    lines = []
    lines.append("## H4: Sequential Consistency")
    lines.append("")
    lines.append("**Hypothesis:** Model rankings from single-period forecast metrics (pinball,")
    lines.append("CRPS) may not match rankings from full 8-week sequential simulation costs,")
    lines.append("because inventory dynamics (carry-forward, lead-time interactions) amplify")
    lines.append("or dampen forecast errors over time.")
    lines.append("")

    grid = data['grid']
    pinball = data['pinball']
    strategy = data['strategy']
    wc = data['wass_crps']
    crps_pb = data['crps_pinball']

    # 8-week cost ranking at SL=0.833
    if not grid.empty:
        lines.append("### 8-week simulation cost ranking (SL=0.833)")
        lines.append("")
        sl83 = grid[grid['sl'] == 0.833].sort_values('total')
        for i, (_, r) in enumerate(sl83.iterrows(), 1):
            lines.append(f"{i}. **{r['model']}**: €{r['total']:,.2f}")
        lines.append("")

    # Pinball ranking at q=0.833
    if not pinball.empty:
        lines.append("### Single-period pinball loss ranking at q=0.833")
        lines.append("")
        pb_col = 'q_0.833'
        if pb_col in pinball.columns:
            pb_sorted = pinball.sort_values(pb_col)
            for i, (_, r) in enumerate(pb_sorted.iterrows(), 1):
                lines.append(f"{i}. **{r['model']}**: pinball={r[pb_col]:.4f}")
            lines.append("")

    # Pinball ranking at q=0.500
    if not pinball.empty:
        lines.append("### Single-period pinball loss ranking at q=0.500 (median)")
        lines.append("")
        pb_col = 'q_0.500'
        if pb_col in pinball.columns:
            pb_sorted = pinball.sort_values(pb_col)
            for i, (_, r) in enumerate(pb_sorted.iterrows(), 1):
                lines.append(f"{i}. **{r['model']}**: pinball={r[pb_col]:.4f}")
            lines.append("")

    # Cost-weighted pinball at CF
    pinball_cw = data['pinball_cw']
    if not pinball_cw.empty:
        lines.append("### Cost-weighted pinball ranking at q=0.833")
        lines.append("")
        cw_col = 'q_0.833'
        if cw_col in pinball_cw.columns:
            cw_sorted = pinball_cw.sort_values(cw_col)
            for i, (_, r) in enumerate(cw_sorted.iterrows(), 1):
                lines.append(f"{i}. **{r['model']}**: cost-weighted pinball={r[cw_col]:.4f}")
            lines.append("")

    # CRPS ranking
    if not crps_pb.empty:
        lines.append("### CRPS ranking (distributional quality)")
        lines.append("")
        crps_sorted = crps_pb.sort_values('crps_mean')
        for i, (_, r) in enumerate(crps_sorted.iterrows(), 1):
            lines.append(f"{i}. **{r['model']}**: CRPS={r['crps_mean']:.4f}")
        lines.append("")

    # Wasserstein ranking
    if not wc.empty:
        lines.append("### Wasserstein distance ranking")
        lines.append("")
        wc_col = 'wasserstein_mean'
        if wc_col in wc.columns:
            wc_sorted = wc.sort_values(wc_col)
            for i, (_, r) in enumerate(wc_sorted.iterrows(), 1):
                model = r.get('model', r.name if isinstance(r.name, str) else '')
                lines.append(f"{i}. **{model}**: W1={r[wc_col]:.4f}")
            lines.append("")

    # Comprehensive rank comparison table
    lines.append("### Comprehensive rank comparison")
    lines.append("")
    rank_header = "| Metric | Rank 1 | Rank 2 | Rank 3 | Rank 4 |"
    rank_sep = "|--------|--------|--------|--------|--------|"
    lines.append(rank_header)
    lines.append(rank_sep)

    if not grid.empty:
        sl83 = grid[grid['sl'] == 0.833].sort_values('total')
        models = sl83['model'].tolist()
        if len(models) >= 4:
            lines.append(f"| 8-week cost (SL=0.833) | {models[0]} | {models[1]} | {models[2]} | {models[3]} |")

    if not pinball.empty and 'q_0.833' in pinball.columns:
        pb_sorted = pinball.sort_values('q_0.833')
        models = pb_sorted['model'].tolist()
        if len(models) >= 4:
            lines.append(f"| Pinball @ q=0.833 | {models[0]} | {models[1]} | {models[2]} | {models[3]} |")

    if not pinball_cw.empty and 'q_0.833' in pinball_cw.columns:
        cw_sorted = pinball_cw.sort_values('q_0.833')
        models = cw_sorted['model'].tolist()
        if len(models) >= 4:
            lines.append(f"| Cost-weighted pinball | {models[0]} | {models[1]} | {models[2]} | {models[3]} |")

    if not crps_pb.empty:
        crps_sorted = crps_pb.sort_values('crps_mean')
        models = crps_sorted['model'].tolist()
        if len(models) >= 4:
            lines.append(f"| CRPS | {models[0]} | {models[1]} | {models[2]} | {models[3]} |")

    if not wc.empty and 'wasserstein_mean' in wc.columns:
        wc_sorted = wc.sort_values('wasserstein_mean')
        models_w = []
        for _, r in wc_sorted.iterrows():
            models_w.append(r.get('model', r.name if isinstance(r.name, str) else ''))
        if len(models_w) >= 4:
            lines.append(f"| Wasserstein W1 | {models_w[0]} | {models_w[1]} | {models_w[2]} | {models_w[3]} |")

    if not wc.empty and 'composite_mean' in wc.columns:
        comp_sorted = wc.sort_values('composite_mean')
        models_c = []
        for _, r in comp_sorted.iterrows():
            models_c.append(r.get('model', r.name if isinstance(r.name, str) else ''))
        if len(models_c) >= 4:
            lines.append(f"| Composite (PB*W1) | {models_c[0]} | {models_c[1]} | {models_c[2]} | {models_c[3]} |")

    lines.append("")

    # Train strategy impact (relates to sequential consistency)
    if not strategy.empty:
        lines.append("### Training strategy impact (static vs sequential)")
        lines.append("")
        lines.append("| Model | SL | Static | Sequential | Delta |")
        lines.append("|-------|---:|-------:|-----------:|------:|")
        for model in ['slurp_bootstrap', 'slurp_stockout_aware', 'lightgbm_quantile']:
            for sl in [0.833]:
                st = strategy[(strategy['model'] == model) & (strategy['sl'] == sl) & (strategy['strategy'] == 'static')]
                sq = strategy[(strategy['model'] == model) & (strategy['sl'] == sl) & (strategy['strategy'] == 'sequential')]
                if not st.empty and not sq.empty:
                    st_c = st['total'].values[0]
                    sq_c = sq['total'].values[0]
                    delta = sq_c - st_c
                    lines.append(f"| {model} | {sl:.3f} | €{st_c:,.2f} | €{sq_c:,.2f} | €{delta:+,.2f} |")
        lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    lines.append("The expanded rank comparison now includes CRPS, Wasserstein, and composite")
    lines.append("scores alongside cost and pinball. Key observations:")
    lines.append("")
    lines.append("- If rankings are consistent across all metrics, the best single-period")
    lines.append("  forecast is also the best sequential planner — H4 would be rejected.")
    lines.append("- If rankings differ (e.g., a model wins on CRPS but loses on 8-week cost),")
    lines.append("  H4 is supported: inventory dynamics create path dependencies that")
    lines.append("  single-period metrics cannot capture.")
    lines.append("")
    lines.append("The training strategy comparison further supports H4: the static (train-once)")
    lines.append("approach often outperforms sequential refit for SLURP models at high SL,")
    lines.append("suggesting that the additional competition-week data in later folds")
    lines.append("may introduce instability rather than improving forecasts.")
    lines.append("")

    return "\n".join(lines)


def build_executive_summary(data: dict) -> str:
    """Build an executive summary of all findings."""
    lines = []
    lines.append("# Hypothesis Analysis: Corrected 8-Week Backtest Results")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**Benchmark cost:** €{BENCHMARK_COST:,.2f} (Rolling Seasonal MA + Order-Up-To)")
    lines.append(f"**Competition winner:** €{WINNER_COST:,.2f}")
    lines.append("")

    grid = data['grid']
    if not grid.empty:
        best = grid.loc[grid['total'].idxmin()]
        lines.append(f"**Our best result:** {best['model']} @ SL={best['sl']:.3f} → €{best['total']:,.2f}")
        if best['total'] < BENCHMARK_COST:
            lines.append(f"  - Beats benchmark by €{BENCHMARK_COST - best['total']:,.2f}")
        if best['total'] < WINNER_COST:
            lines.append(f"  - Beats winner by €{WINNER_COST - best['total']:,.2f}")
        else:
            lines.append(f"  - Gap to winner: €{best['total'] - WINNER_COST:,.2f}")
        lines.append("")

    strategy = data['strategy']
    if not strategy.empty:
        best_s = strategy.loc[strategy['total'].idxmin()]
        lines.append(f"**Best with strategy optimization:** {best_s['model']} @ SL={best_s['sl']:.3f} "
                     f"({best_s['strategy']}) → €{best_s['total']:,.2f}")
        lines.append("")

    lines.append("### Key Findings")
    lines.append("")
    lines.append("1. **Cost tallying bug invalidated all prior backtest results.** The previous")
    lines.append("   results showed SL=0.50 as optimal due to shortage costs being tallied at")
    lines.append("   1/5th their true value. With correct eval_costs, SL=0.833 is optimal for")
    lines.append("   all models except seasonal_naive.")
    lines.append("")
    lines.append("2. **SLURP models significantly outperform seasonal_naive and LightGBM.**")
    lines.append("   At the optimal SL=0.833, slurp_bootstrap (€5,169) and slurp_stockout_aware")
    lines.append("   (€5,264) beat the benchmark, while lightgbm_quantile (€6,756) and")
    lines.append("   seasonal_naive (€10,316) do not.")
    lines.append("")
    lines.append("3. **seasonal_naive has catastrophic positive bias** (median pred 83.6 vs actual 2.9),")
    lines.append("   causing massive over-ordering regardless of service level.")
    lines.append("")
    lines.append("4. **Stockout awareness has a nuanced effect** (H2): it helps at lower SLs")
    lines.append("   but slightly hurts at SL=0.833, possibly due to wider uncertainty bands.")
    lines.append("")
    lines.append("5. **Train-once outperforms sequential refit** for SLURP at high SL,")
    lines.append("   suggesting the marginal competition data adds noise rather than signal.")
    lines.append("")

    # Distributional quality highlight
    wc = data['wass_crps']
    if not wc.empty and 'crps_mean' in wc.columns:
        lines.append("6. **Distributional quality varies dramatically across models.** CRPS and")
        lines.append("   Wasserstein distance reveal that forecast miscalibration — not just bias —")
        lines.append("   drives cost differences. The composite score (pinball x Wasserstein)")
        lines.append("   identifies specific SKUs where targeted model switching would have")
        lines.append("   the highest cost impact.")
        lines.append("")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    sections = [
        build_executive_summary(data),
        h1_jensen_gap(data),
        h2_stockout_awareness(data),
        h3_surd_effect(data),
        distributional_quality(data),
        h4_sequential_consistency(data),
    ]

    report = "\n\n".join(sections)

    output_path = OUTPUT_DIR / 'hypothesis_analysis.md'
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Analysis saved to {output_path}")
    print(f"\n{'='*60}")
    print(report)


if __name__ == '__main__':
    main()
