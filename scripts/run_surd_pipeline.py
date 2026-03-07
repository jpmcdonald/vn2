#!/usr/bin/env python3
"""
Runbook: Steps 2-5 of the SURD pipeline (backtest, compare, test H3, regenerate).

Prerequisite: Step 1 (training) must be done; checkpoints must exist in the given dir.

Usage:
  python scripts/run_surd_pipeline.py
  python scripts/run_surd_pipeline.py --checkpoints-dir models/checkpoints_h3
  python scripts/run_surd_pipeline.py --skip-backtest --skip-compare  # regenerate only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], step_name: str) -> bool:
    print(f"\n{'='*60}\n[Step] {step_name}\n{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {step_name}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest, compare, H3 test, and regenerate (steps 2-5 of SURD pipeline)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Checkpoint directory (must match training). Default: models/checkpoints",
    )
    parser.add_argument("--skip-backtest", action="store_true", help="Skip Step 2 (backtest grid)")
    parser.add_argument("--skip-compare", action="store_true", help="Skip Step 3 (compare to baseline)")
    parser.add_argument("--skip-h3", action="store_true", help="Skip Step 4 (H3 hypothesis test)")
    parser.add_argument("--skip-regenerate", action="store_true", help="Skip Step 5 (bias, f1, pinball, analyze_hypotheses)")
    args = parser.parse_args()

    cp = str(args.checkpoints_dir)
    py = sys.executable

    if not args.skip_backtest:
        if not run(
            [py, "scripts/backtest_all_models.py", "--checkpoints-dir", cp],
            "Backtest (model x service-level grid)",
        ):
            return 1

    if not args.skip_compare:
        if not run([py, "scripts/compare_backtest_to_baseline.py"], "Compare to baseline"):
            return 1

    if not args.skip_h3:
        if not run(
            [
                py,
                "scripts/test_h3_surd_effect.py",
                "--checkpoints-dir", cp,
                "--output-dir", "reports/hypothesis_tests",
            ],
            "H3 SURD effect test",
        ):
            return 1

    if not args.skip_regenerate:
        if not run(
            [py, "scripts/compute_bias_analysis.py", "--checkpoints-dir", cp],
            "Bias / calibration / Wasserstein / CRPS",
        ):
            return 1
        if not run([py, "scripts/compute_stockout_f1.py"], "Stockout F1 from backtest runs"):
            return 1
        if not run(
            [py, "scripts/compute_pinball_all_quantiles.py", "--checkpoints-dir", cp],
            "Pinball at all quantiles",
        ):
            return 1
        if not run([py, "scripts/analyze_hypotheses.py"], "Hypothesis analysis report"):
            return 1

    print("\n[OK] Pipeline steps 2-5 completed. Update docs/paper/revised_paper.md §5.3 and docs/WORK_LOG_2026_03_05.md with H3 verdict and summary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
