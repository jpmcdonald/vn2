#!/usr/bin/env python3
"""
Compare our 8-week backtest cost and benchmark cost to known competition results.

Known competition values (from docs/backtesting_against_competition.md):
  Winner (Bartosz Szablowski): €4,677
  Our actual competition result: €7,787.40 (rank 110 of ~185)
  Top-20% threshold (estimated): rank <= 37 of ~185

The cumulative-leaderboard.txt in this repo only contains Order #1 data (all 380.60),
so we use documented values for the 8-week comparison.

Usage:
  python scripts/compare_8week_results.py --our-cost 5500 --benchmark-cost 5200
  python scripts/compare_8week_results.py --our-cost 5500 --benchmark-cost 5200 --winner-cost 4677
"""

from __future__ import annotations

import argparse
import math

# Known competition 8-week values from docs
KNOWN_WINNER_COST = 4677.0
KNOWN_OUR_ACTUAL_COST = 7787.40
KNOWN_OUR_ACTUAL_RANK = 110
KNOWN_N_COMPETITORS = 185


def main():
    p = argparse.ArgumentParser(description="Compare 8-week costs to competition results.")
    p.add_argument("--our-cost", type=float, required=True, help="Our pipeline 8-week backtest cost")
    p.add_argument("--benchmark-cost", type=float, required=True, help="Rolling benchmark 8-week cost")
    p.add_argument("--winner-cost", type=float, default=KNOWN_WINNER_COST,
                    help=f"Winner 8-week cost (default: {KNOWN_WINNER_COST})")
    p.add_argument("--n-competitors", type=int, default=KNOWN_N_COMPETITORS,
                    help=f"Number of competitors (default: {KNOWN_N_COMPETITORS})")
    args = p.parse_args()

    n = args.n_competitors
    rank_20pct = math.ceil(0.2 * n)

    print("=" * 70)
    print("8-Week Backtest Comparison")
    print("=" * 70)
    print()
    print("  BACKTEST RESULTS")
    print(f"    Our pipeline (seasonal_naive + newsvendor): €{args.our_cost:,.2f}")
    print(f"    Rolling benchmark (seasonal MA + OUP):      €{args.benchmark_cost:,.2f}")
    print()
    print("  COMPETITION REFERENCE (documented)")
    print(f"    Winner (Bartosz Szablowski):                €{args.winner_cost:,.2f}")
    print(f"    Our actual competition result:              €{KNOWN_OUR_ACTUAL_COST:,.2f}  (rank {KNOWN_OUR_ACTUAL_RANK}/{n})")
    print(f"    Top-20% threshold:                         rank ≤ {rank_20pct} of {n}")
    print()

    gap_to_winner = args.our_cost - args.winner_cost
    gap_to_bench = args.our_cost - args.benchmark_cost
    improvement_vs_actual = KNOWN_OUR_ACTUAL_COST - args.our_cost

    print("  ANALYSIS")
    print(f"    Our backtest vs winner:     {'+' if gap_to_winner > 0 else ''}€{gap_to_winner:,.2f}  ({'worse' if gap_to_winner > 0 else 'better'})")
    print(f"    Our backtest vs benchmark:  {'+' if gap_to_bench > 0 else ''}€{gap_to_bench:,.2f}  ({'worse' if gap_to_bench > 0 else 'better'})")
    print(f"    Our backtest vs our actual: {'+' if improvement_vs_actual < 0 else ''}€{abs(improvement_vs_actual):,.2f}  ({'worse' if improvement_vs_actual < 0 else 'better'})")
    print()

    if args.our_cost <= args.benchmark_cost:
        print("    >> Our pipeline BEATS the benchmark")
    else:
        print(f"    >> Benchmark wins by €{gap_to_bench:,.2f}")

    if args.our_cost <= args.winner_cost * 1.2:
        print(f"    >> Within 20% of winner cost (ratio: {args.our_cost / args.winner_cost:.2f}x)")
    else:
        print(f"    >> {args.our_cost / args.winner_cost:.2f}x the winner cost")

    print("=" * 70)


if __name__ == "__main__":
    main()
