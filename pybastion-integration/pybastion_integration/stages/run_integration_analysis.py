#!/usr/bin/env python3
"""
Integration Test Pipeline Driver

Runs the complete integration test generation pipeline:
1. Stage 1: Collect integration points from ledgers
2. Stage 2: Categorize and reduce execution paths
3. Stage 3: Generate test specifications with fixtures

DEFAULT BEHAVIOR (no args):
  - Runs all stages in sequence
  - Uses default paths from config
  - Outputs to ./integration-output/

EXAMPLES:
  # Run full pipeline
  ./run_integration_pipeline.py

  # Run only stage 2 and 3
  ./run_integration_pipeline.py --start-from 2

  # Run with verbose output
  ./run_integration_pipeline.py -v

  # Clean outputs and run fresh
  ./run_integration_pipeline.py --clean

  # Run only interunit integrations
  ./run_integration_pipeline.py --interunit-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

STAGES = {
    1: {
        'name': 'Collect Integration Points',
        'script': 'stage1_collect_integration_points.py',
        'input': None,
        'output': config.get_stage_output(1)
    },
    2: {
        'name': 'Categorize Execution Paths',
        'script': 'stage2_categorize_paths.py',
        'input': config.get_stage_output(1),
        'output': config.get_stage_output(2)
    },
    3: {
        'name': 'Generate Test Specifications',
        'script': 'stage3_generate_test_specs.py',
        'input': config.get_stage_output(2),
        'output': config.get_stage_output(3)
    }
}


def run_stage(
        stage_num: int,
        verbose: bool = False,
        interunit_only: bool = False,
        dry_run: bool = False
) -> int:
    """
    Run a single pipeline stage.

    Args:
        stage_num: Stage number (1, 2, or 3)
        verbose: Enable verbose output
        interunit_only: Only process interunit integrations
        dry_run: Print command without executing

    Returns:
        Exit code from stage script
    """
    stage = STAGES[stage_num]
    script_path = Path(__file__).parent / stage['script']

    if not script_path.exists():
        print(f"ERROR: Stage {stage_num} script not found: {script_path}", file=sys.stderr)
        return 1

    # Build command
    cmd = [sys.executable, str(script_path)]

    if verbose:
        cmd.append('-v')

    if interunit_only and stage_num in (2, 3):
        cmd.append('--interunit-only')

    # Print stage header
    print(f"\n{'=' * 70}")
    print(f"Stage {stage_num}: {stage['name']}")
    print(f"{'=' * 70}")

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return 0

    # Run the stage
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Stage {stage_num} failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    # Verify output was created
    output_path = stage['output']
    if output_path and not output_path.exists():
        print(f"\nWARNING: Expected output not found: {output_path}", file=sys.stderr)

    return 0


def clean_outputs(verbose: bool = False) -> None:
    """
    Clean all pipeline output files.

    Args:
        verbose: Print what's being removed
    """
    output_dir = config.get_stage_output(1).parent

    if not output_dir.exists():
        if verbose:
            print(f"Output directory doesn't exist: {output_dir}")
        return

    print(f"\nCleaning outputs in: {output_dir}")

    for stage_num in STAGES:
        output_path = STAGES[stage_num]['output']
        if output_path and output_path.exists():
            if verbose:
                print(f"  Removing: {output_path}")
            output_path.unlink()

    print("✓ Outputs cleaned")


def validate_prerequisites(verbose: bool = False) -> bool:
    """
    Validate that prerequisites exist before running pipeline.

    Args:
        verbose: Print detailed validation info

    Returns:
        True if all prerequisites exist
    """
    errors = []

    # Check ledgers root
    ledgers_root = config.get_ledgers_root()
    if not ledgers_root.exists():
        errors.append(f"Ledgers root not found: {ledgers_root}")
    elif verbose:
        print(f"✓ Ledgers root exists: {ledgers_root}")

    # Check EIS root (needed for stage 2)
    eis_root = config.get_target_root() / 'dist' / 'eis'
    if not eis_root.exists():
        errors.append(f"EIS root not found: {eis_root}")
    elif verbose:
        print(f"✓ EIS root exists: {eis_root}")

    # Check callable inventory (needed for stage 1)
    inventory_path = config.get_target_root() / 'dist' / 'inspect' / 'callable-inventory.txt'
    if not inventory_path.exists():
        errors.append(f"Callable inventory not found: {inventory_path}")
    elif verbose:
        print(f"✓ Callable inventory exists: {inventory_path}")

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
        choices=[1, 2, 3],
        help='Start from specific stage (runs that stage and all following)'
    )
    ap.add_argument(
        '--stop-at',
        type=int,
        choices=[1, 2, 3],
        help='Stop at specific stage (runs up to and including that stage)'
    )
    ap.add_argument(
        '--only',
        type=int,
        choices=[1, 2, 3],
        help='Run only a specific stage'
    )
    ap.add_argument(
        '--clean',
        action='store_true',
        help='Clean output files before running'
    )
    ap.add_argument(
        '--interunit-only',
        action='store_true',
        help='Only process interunit integrations (stages 2-3)'
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be run without executing'
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

    # Validate prerequisites
    if not args.skip_validation:
        if args.verbose:
            print("Validating prerequisites...")
        if not validate_prerequisites(verbose=args.verbose):
            return 1

    # Clean if requested
    if args.clean:
        clean_outputs(verbose=args.verbose)

    # Determine which stages to run
    if args.only:
        stages_to_run = [args.only]
    else:
        start = args.start_from or 1
        stop = args.stop_at or 3
        stages_to_run = list(range(start, stop + 1))

    if args.verbose:
        print(f"\nRunning stages: {stages_to_run}")

    # Run stages
    for stage_num in stages_to_run:
        # Check if input exists (except for stage 1)
        stage = STAGES[stage_num]
        if stage['input'] and not stage['input'].exists():
            print(f"\nERROR: Stage {stage_num} input not found: {stage['input']}", file=sys.stderr)
            print(f"You may need to run earlier stages first.", file=sys.stderr)
            return 1

        exit_code = run_stage(
            stage_num,
            verbose=args.verbose,
            interunit_only=args.interunit_only,
            dry_run=args.dry_run
        )

        if exit_code != 0:
            return exit_code

    # Success summary
    print(f"\n{'=' * 70}")
    print("✓ Pipeline completed successfully")
    print(f"{'=' * 70}")

    if not args.dry_run:
        final_output = STAGES[max(stages_to_run)]['output']
        print(f"\nFinal output: {final_output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())