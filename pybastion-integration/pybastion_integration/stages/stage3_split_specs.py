#!/usr/bin/env python3
"""
Stage 3: Split Test Specifications by Unit Pair

Input:  Stage 2 output (stage2-integration-test-specs.yaml)
Output: One YAML file per source-unit → target-unit pair, written to a
        subdirectory (spec_split_output_dir from config).

Each output file contains all specs that describe tests between the same
two units, and maps cleanly to a single pytest test module.

Output filenames: {source_unit}__{target_unit}.yaml
  e.g. api__resolution.yaml
       builtin_strategies__repository.yaml

DEFAULT BEHAVIOR (no args):
  - Reads from config.get_stage_output(2)
  - Outputs to config.get_spec_split_output_dir()
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from pybastion_integration import config


# =============================================================================
# Helpers
# =============================================================================

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def safe_filename_segment(name: str) -> str:
    """Convert a unit name to a safe filename segment."""
    # Strip leading paths/packages — use only the final module name
    segment = name.split('.')[-1].split('::')[-1]
    # Replace any remaining non-alphanumeric chars with underscore
    segment = re.sub(r'[^A-Za-z0-9_]', '_', segment)
    return segment.lower()


def pair_key(spec: dict[str, Any]) -> tuple[str, str]:
    """Return (source_unit, target_unit) key for a spec."""
    source_unit = spec.get('source', {}).get('unit', 'unknown')
    target_unit = spec.get('target', {}).get('unit', 'unknown')
    return source_unit, target_unit


def pair_filename(source_unit: str, target_unit: str) -> str:
    """Return output filename for a unit pair."""
    src = safe_filename_segment(source_unit)
    tgt = safe_filename_segment(target_unit)
    return f"{src}__->__{tgt}.yaml"


# =============================================================================
# Splitting
# =============================================================================

def split_specs(
        specs: list[dict[str, Any]],
        verbose: bool = False,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    Group specs by (source_unit, target_unit) pair.

    Returns:
        Dict mapping (source_unit, target_unit) -> list of specs
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for spec in specs:
        key = pair_key(spec)
        groups[key].append(spec)

    if verbose:
        print(f"  {len(groups)} unique unit pairs across {len(specs)} specs")

    return dict(groups)


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Stage 2 input file (default: from config)',
    )
    ap.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for split files (default: from config)',
    )
    ap.add_argument(
        '--target-root',
        type=Path,
        help='Target project root (sets config defaults)',
    )
    ap.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output',
    )

    args = ap.parse_args(argv)

    if args.target_root:
        config.set_target_root(args.target_root)
        if args.verbose:
            print(f"Target root: {args.target_root}")

    input_path = args.input or config.get_stage_output(2)
    output_dir = args.output_dir or config.get_spec_split_output_dir()

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Load stage 2 output
    if args.verbose:
        print(f"Loading: {input_path}")

    stage2_data = load_yaml(input_path)
    specs = stage2_data.get('test_specs', [])

    if not specs:
        print("ERROR: No test specs found in input", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded {len(specs)} specs")

    # Split by unit pair
    if args.verbose:
        print("\nSplitting by unit pair...")

    groups = split_specs(specs, verbose=args.verbose)

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for (source_unit, target_unit), group_specs in sorted(groups.items()):
        filename = pair_filename(source_unit, target_unit)
        output_path = output_dir / filename

        output_data = {
            'stage': 'split-test-specs',
            'source_unit': source_unit,
            'target_unit': target_unit,
            'spec_count': len(group_specs),
            'test_specs': group_specs,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                output_data,
                f,
                default_flow_style=False,
                sort_keys=config.get_yaml_sort_keys(),
                width=config.get_yaml_width(),
                indent=config.get_yaml_indent(),
            )

        if args.verbose:
            print(f"  {filename}: {len(group_specs)} spec(s)")

        written += 1

    print(f"\n✓ Split {len(specs)} specs into {written} file(s) → {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
