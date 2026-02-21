#!/usr/bin/env python3
"""
Integration Test Pipeline Driver

Runs the complete integration test generation pipeline:
  Stage 1: Build feasibility-weighted call graph from ledgers + inventories
  Stage 2: Collect integration points from the call graph
  Stage 3: Categorize and reduce execution paths
  Stage 4: Generate test specifications with fixtures

DEFAULT BEHAVIOR (no args):
  - Runs all stages in sequence
  - Uses default paths from config
  - Outputs to dist/integration-output/

EXAMPLES:
  # Run full pipeline
  ./run_integration_analysis.py

  # Run from stage 2 onward (call graph already built)
  ./run_integration_analysis.py --start-from 2

  # Run only stage 1
  ./run_integration_analysis.py --only 1

  # Run with verbose output
  ./run_integration_analysis.py -v

  # Clean outputs and run fresh
  ./run_integration_analysis.py --clean

  # Dry run - print commands without executing
  ./run_integration_analysis.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pybastion_integration import config

STAGES = {
    1: {
        'name': 'Build Call Graph',
        'script': 'stage1_build_call_graph.py',
        'input': None,
        'output': config.get_stage_output(1),
    },
    2: {
        'name': 'Collect Integration Points',
        'script': 'stage2_collect_integration_points.py',
        'input': config.get_stage_output(1),
        'output': config.get_stage_output(2),
    },
    3: {
        'name': 'Categorize Execution Paths',
        'script': 'stage3_categorize_paths.py',
        'input': config.get_stage_output(2),
        'output': config.get_stage_output(3),
    },
    4: {
        'name': 'Generate Test Specifications',
        'script': 'stage4_generate_test_specs.py',
        'input': config.get_stage_output(3),
        'output': config.get_stage_output(4),
    },
}

ALL_STAGES = sorted(STAGES.keys())


def run_stage(stage_num: int, verbose: bool = False, dry_run: bool = False) -> int:
    """
    Run a single pipeline stage.

    Args:
        stage_num: Stage number (1-4)
        verbose: Enable verbose output
        dry_run: Print command without executing

    Returns:
        Exit code from stage script
    """
    stage = STAGES[stage_num]
    script_path = Path(__file__).parent / stage['script']

    if not script_path.exists():
        print(f"ERROR: Stage {stage_num} script not found: {script_path}", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(script_path)]
    if verbose:
        cmd.append('-v')

    print(f"\n{'=' * 70}")
    print(f"Stage {stage_num}: {stage['name']}")
    print(f"{'=' * 70}")

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return 0

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Stage {stage_num} failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    output_path = stage['output']
    if output_path and not output_path.exists():
        print(f"\nWARNING: Expected output not found: {output_path}", file=sys.stderr)

    return 0


def clean_outputs(verbose: bool = False) -> None:
    """Remove all pipeline output files."""
    output_dir = config.get_integration_output_dir()

    if not output_dir.exists():
        if verbose:
            print(f"Output directory doesn't exist: {output_dir}")
        return

    print(f"\nCleaning outputs in: {output_dir}")

    for stage_num, stage in STAGES.items():
        output_path = stage['output']
        if output_path and output_path.exists():
            if verbose:
                print(f"  Removing: {output_path}")
            output_path.unlink()

    print("✓ Outputs cleaned")


def validate_prerequisites(verbose: bool = False) -> bool:
    """
    Validate that pipeline inputs exist before running.

    Returns:
        True if all prerequisites exist
    """
    errors = []

    ledgers_root = config.get_ledgers_root()
    if not ledgers_root.exists():
        errors.append(f"Ledgers root not found: {ledgers_root}")
    elif verbose:
        print(f"✓ Ledgers root exists: {ledgers_root}")

    inventories_root = config.get_inventories_root()
    if not inventories_root.exists():
        errors.append(f"Inventories root not found: {inventories_root}")
    elif verbose:
        print(f"✓ Inventories root exists: {inventories_root}")

    if errors:
        print("\nERROR: Missing prerequisites:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return False

    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        '--start-from',
        type=int,
        choices=ALL_STAGES,
        help='Start from specific stage (runs that stage and all following)'
    )
    ap.add_argument(
        '--stop-at',
        type=int,
        choices=ALL_STAGES,
        help='Stop at specific stage (inclusive)'
    )
    ap.add_argument(
        '--only',
        type=int,
        choices=ALL_STAGES,
        help='Run only a specific stage'
    )
    ap.add_argument(
        '--clean',
        action='store_true',
        help='Clean output files before running'
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    ap.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip prerequisite validation'
    )
    ap.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = ap.parse_args(argv)

    if not args.skip_validation:
        if args.verbose:
            print("Validating prerequisites...")
        if not validate_prerequisites(verbose=args.verbose):
            return 1

    if args.clean:
        clean_outputs(verbose=args.verbose)

    # Determine which stages to run
    if args.only:
        stages_to_run = [args.only]
    else:
        start = args.start_from or ALL_STAGES[0]
        stop = args.stop_at or ALL_STAGES[-1]
        stages_to_run = list(range(start, stop + 1))

    if args.verbose:
        print(f"\nRunning stages: {stages_to_run}")

    for stage_num in stages_to_run:
        stage = STAGES[stage_num]
        if stage['input'] and not stage['input'].exists():
            print(f"\nERROR: Stage {stage_num} input not found: {stage['input']}", file=sys.stderr)
            print("You may need to run earlier stages first.", file=sys.stderr)
            return 1

        exit_code = run_stage(stage_num, verbose=args.verbose, dry_run=args.dry_run)
        if exit_code != 0:
            return exit_code

    print(f"\n{'=' * 70}")
    print("✓ Pipeline completed successfully")
    print(f"{'=' * 70}")

    if not args.dry_run:
        print(f"\nFinal output: {STAGES[max(stages_to_run)]['output']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
