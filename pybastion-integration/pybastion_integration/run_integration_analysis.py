#!/usr/bin/env python3
"""
Integration Test Pipeline Driver

Runs the complete integration test generation pipeline:
  Stage 1: Build feasibility-weighted call graph from ledgers + inventories
  Stage 2: Collect integration points from the call graph
  Stage 3: Categorize and reduce execution paths
  Stage 4: Generate test specifications with fixtures

REQUIRED:
  --target-root   Root directory of the project to analyze

EXAMPLES:
  # Run full pipeline
  ./run_integration_analysis.py --target-root /path/to/project

  # Run from stage 2 onward (call graph already built)
  ./run_integration_analysis.py --target-root /path/to/project --start-from 2

  # Run only stage 1
  ./run_integration_analysis.py --target-root /path/to/project --only 1

  # Run with verbose output
  ./run_integration_analysis.py --target-root /path/to/project -v

  # Clean outputs and run fresh
  ./run_integration_analysis.py --target-root /path/to/project --clean

  # Dry run - print commands without executing
  ./run_integration_analysis.py --target-root /path/to/project --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES_DIR = Path(__file__).parent / 'stages'
ALL_STAGE_NUMS = [1, 2, 3]


def derive_paths(target_root: Path) -> dict:
    """Derive all pipeline paths from the target root."""
    output_dir = target_root / 'dist' / 'integration-output'
    return {
        'ledgers_root': target_root / 'dist' / 'ledgers',
        'inventories_root': target_root / 'dist' / 'inventory',
        'spec_split_output_dir': output_dir / 'split-specs',
        'output_dir': output_dir,
        'stage1_output': output_dir / 'stage1-call-graph.yaml',
        'stage2_output': output_dir / 'stage2-integration-test-specs.yaml',
        'stage3_output': output_dir / 'split-specs' / '*.yaml',
    }


def build_stage_cmd(stage_num: int, target_root: Path, paths: dict, verbose: bool) -> list[str]:
    """Build the command for a given stage."""
    script = STAGES_DIR / {
        1: 'stage1_build_call_graph.py',
        2: 'stage2_generate_test_specs.py',
        3: 'stage3_split_specs.py'
    }[stage_num]

    cmd = [sys.executable, str(script)]

    if stage_num == 1:
        cmd += [
            '--target-root', str(target_root),
            '--ledgers-root', str(paths['ledgers_root']),
            '--inventories-root', str(paths['inventories_root']),
            '--output', str(paths['stage1_output'])
        ]
    elif stage_num == 2:
        cmd += [
            '--target-root', str(target_root),
            '--input', str(paths['stage1_output']),
            '--output', str(paths['stage2_output']),
        ]
    elif stage_num == 3:
        cmd += [
            '--target-root', str(target_root),
            '--input', str(paths['stage2_output']),
            '--output-dir', str(paths['spec_split_output_dir']),
        ]

    if verbose:
        cmd.append('-v')
        print(f"Running command: {' '.join(cmd)}")

    return cmd


def run_stage(stage_num: int, target_root: Path, paths: dict,
              verbose: bool = False, dry_run: bool = False) -> int:
    stage_names = {
        1: 'Build Call Graph',
        2: 'Generate Test Specifications',
        3: 'Split Integration Test Specs'
    }

    print(f"\n{'=' * 70}")
    print(f"Stage {stage_num}: {stage_names[stage_num]}")
    print(f"{'=' * 70}")

    cmd = build_stage_cmd(stage_num, target_root, paths, verbose)

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return 0

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Stage {stage_num} failed with exit code {result.returncode}",
              file=sys.stderr)
        return result.returncode

    return 0


def clean_outputs(paths: dict, verbose: bool = False) -> None:
    """Remove all pipeline output files."""
    output_dir = paths['output_dir']

    if not output_dir.exists():
        if verbose:
            print(f"Output directory doesn't exist: {output_dir}")
        return

    print(f"\nCleaning outputs in: {output_dir}")

    for key in ('stage1_output', 'stage2_output', 'stage3_output'):
        p = paths[key]
        if '*' in str(p):
            for f in p.parent.glob(p.name):
                if verbose:
                    print(f"  Removing: {f}")
                f.unlink()
        elif p.exists():
            if verbose:
                print(f"  Removing: {p}")
            p.unlink()

    print("✓ Outputs cleaned")


def validate_prerequisites(paths: dict, verbose: bool = False) -> bool:
    errors = []

    if not paths['ledgers_root'].exists():
        errors.append(f"Ledgers root not found: {paths['ledgers_root']}")
    elif verbose:
        print(f"✓ Ledgers root exists: {paths['ledgers_root']}")

    if not paths['inventories_root'].exists():
        errors.append(f"Inventories root not found: {paths['inventories_root']}")
    elif verbose:
        print(f"✓ Inventories root exists: {paths['inventories_root']}")

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
        '--target-root',
        type=Path,
        required=True,
        help='Root directory of the project to analyze'
    )
    ap.add_argument(
        '--start-from',
        type=int,
        choices=ALL_STAGE_NUMS,
        help='Start from specific stage (runs that stage and all following)'
    )
    ap.add_argument(
        '--stop-at',
        type=int,
        choices=ALL_STAGE_NUMS,
        help='Stop at specific stage (inclusive)'
    )
    ap.add_argument(
        '--only',
        type=int,
        choices=ALL_STAGE_NUMS,
        help='Run only a specific stage'
    )
    ap.add_argument(
        '--no-clean',
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

    target_root = args.target_root.resolve()
    paths = derive_paths(target_root)

    if args.verbose:
        print(f"Target root: {target_root}")

    if not args.skip_validation:
        if not validate_prerequisites(paths, verbose=args.verbose):
            return 1

    if not args.no_clean:
        clean_outputs(paths, verbose=args.verbose)

    # Determine which stages to run
    if args.only:
        stages_to_run = [args.only]
    else:
        start = args.start_from or ALL_STAGE_NUMS[0]
        stop = args.stop_at or ALL_STAGE_NUMS[-1]
        stages_to_run = list(range(start, stop + 1))

    if args.verbose:
        print(f"\nRunning stages: {stages_to_run}")

    # Ensure output directory exists
    paths['output_dir'].mkdir(parents=True, exist_ok=True)

    for stage_num in stages_to_run:
        # Check that required input exists before running
        stage_input = {
            2: paths['stage1_output'],
            3: paths['stage2_output'],
        }.get(stage_num)

        if stage_input and not stage_input.exists():
            print(f"\nERROR: Stage {stage_num} input not found: {stage_input}", file=sys.stderr)
            print("You may need to run earlier stages first.", file=sys.stderr)
            return 1

        exit_code = run_stage(stage_num, target_root, paths,
                              verbose=args.verbose, dry_run=args.dry_run)
        if exit_code != 0:
            return exit_code

    print(f"\n{'=' * 70}")
    print("✓ Pipeline completed successfully")
    print(f"{'=' * 70}")

    if not args.dry_run:
        final_output = {
            1: paths['stage1_output'],
            2: paths['stage2_output'],
            3: paths['stage3_output'],
        }[max(stages_to_run)]
        print(f"\nFinal output: {final_output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
