#!/usr/bin/env python3
"""
Integration Flow Pipeline

Orchestrates all stages of integration flow generation:
1. Collect integration points
2. Classify integration points
3. Build integration graph
4. Enumerate flows
5. Generate windows

Can run full pipeline or individual stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_full_pipeline(ledger_paths: list[Path], output_dir: Path) -> None:
    """
    Run all stages of the integration flow pipeline.

    Args:
        ledger_paths: Paths to unit ledger files
        output_dir: Directory for output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Integration Flow Pipeline")
    print("=" * 70)

    # Stage 1: Collect
    print("\nStage 1: Collecting integration points...")
    # TODO: Call stage 1

    # Stage 2: Classify
    print("\nStage 2: Classifying integration points...")
    # TODO: Call stage 2

    # Stage 3: Graph
    print("\nStage 3: Building integration graph...")
    # TODO: Call stage 3

    # Stage 4: Flows
    print("\nStage 4: Enumerating flows...")
    # TODO: Call stage 4

    # Stage 5: Windows
    print("\nStage 5: Generating windows...")
    # TODO: Call stage 5

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        'ledgers',
        nargs='*',
        type=Path,
        help='Unit ledger files to process'
    )
    ap.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./integration-output'),
        help='Output directory for all artifacts'
    )
    ap.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run only specific stage (default: run all)'
    )

    args = ap.parse_args(argv)

    if not args.ledgers:
        print("Error: No ledger files specified", file=sys.stderr)
        return 1

    if args.stage:
        print(f"Running stage {args.stage} only...")
        # TODO: Implement single-stage execution
    else:
        run_full_pipeline(args.ledgers, args.output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
