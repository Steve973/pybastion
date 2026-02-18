#!/usr/bin/env python3
"""
Stage 2: Classify Integration Points

Input: Integration points from Stage 1
Output: Classified integration points (entry/intermediate/terminal)

This stage classifies integration points based on their role in flows:
- Entry points: First seams (callables with no incoming interunit calls in scope)
- Terminal nodes: Boundaries OR callables with no outgoing interunit calls
- Intermediate: Everything else (has both incoming and outgoing)

DEFAULT BEHAVIOR (no args):
  - Reads from ./integration-output/stage1-integration-points.yaml
  - Outputs to ./integration-output/stage2-classified-points.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from integration import config
from ..shared.yaml_utils import yaml_load, yaml_dump
from ..shared.data_structures import IntegrationPoint, IntegrationPointClassification, load_integration_points


def classify_integration_points(points: list[IntegrationPoint]) -> IntegrationPointClassification:
    """
    Classify integration points into entry/intermediate/terminal.

    Classification logic:
    - Entry point: Source callable has no incoming interunit calls
    - Terminal node: Boundary integration OR no outgoing interunit calls from target
    - Intermediate: Everything else

    Args:
        points: List of IntegrationPoint objects from Stage 1

    Returns:
        IntegrationPointClassification object
    """
    if not points:
        return IntegrationPointClassification()

    # Build indexes for analysis
    # Map: callable_id -> list of integration point IDs where this callable is the SOURCE
    callables_with_outgoing: dict[str, list[str]] = {}

    # Map: target_raw -> list of integration point IDs that call this target
    targets_with_incoming: dict[str, list[str]] = {}

    # Set of all boundary integration IDs
    boundary_ids: set[str] = set()

    for point in points:
        # Track which callables have outgoing calls
        if point.source_callable_id:
            if point.source_callable_id not in callables_with_outgoing:
                callables_with_outgoing[point.source_callable_id] = []
            callables_with_outgoing[point.source_callable_id].append(point.id)

        # Track which targets have incoming calls
        if point.target_raw:
            if point.target_raw not in targets_with_incoming:
                targets_with_incoming[point.target_raw] = []
            targets_with_incoming[point.target_raw].append(point.id)

        # Track boundaries
        if point.boundary:
            boundary_ids.add(point.id)

    # Now classify each point
    entry_points: list[str] = []
    intermediate_seams: list[str] = []
    terminal_nodes: list[str] = []

    for point in points:
        # Boundary integrations are ALWAYS terminal nodes
        if point.id in boundary_ids:
            terminal_nodes.append(point.id)
            continue

        # Check if source callable is an entry point (no incoming interunit calls)
        source_is_entry = point.source_callable_id not in targets_with_incoming

        # Check if target is a terminal (no outgoing calls)
        # We look for the target in our callables_with_outgoing map
        # If target is not there, it either:
        #   1. Is not in our scope (external/stdlib) -> treat as terminal
        #   2. Has no outgoing calls -> terminal
        target_is_terminal = point.target_raw not in callables_with_outgoing

        # Classification decision tree:
        # - If source is entry AND target is terminal -> this is BOTH, but we'll call it entry
        # - If source is entry -> entry point
        # - If target is terminal -> terminal node
        # - Otherwise -> intermediate

        if source_is_entry:
            entry_points.append(point.id)
        elif target_is_terminal:
            terminal_nodes.append(point.id)
        else:
            intermediate_seams.append(point.id)

    return IntegrationPointClassification(
        entry_points=entry_points,
        intermediate_seams=intermediate_seams,
        terminal_nodes=terminal_nodes
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        '--input',
        type=Path,
        default=config.get_stage_input(2),
        help=f'Integration points file from Stage 1 (default: {config.get_stage_input(2)})'
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=config.get_stage_output(2),
        help=f'Output file (default: {config.get_stage_output(2)})'
    )
    ap.add_argument(
        '--target-root',
        type=Path,
        help='Target project root (default: current directory)'
    )
    ap.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = ap.parse_args(argv)

    # Set target root if provided
    if args.target_root:
        config.set_target_root(args.target_root)
        if args.verbose:
            print(f"Target root: {args.target_root}")

    # Validate input file exists
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        print("Run Stage 1 first to generate integration points.", file=sys.stderr)
        return 1

    # Load integration points
    if args.verbose:
        print(f"Loading integration points from: {args.input}")

    points_data = yaml_load(args.input)
    points = load_integration_points(points_data)

    if args.verbose:
        print(f"Loaded {len(points)} integration points")

    # Classify
    if args.verbose:
        print("\nClassifying integration points...")

    classification = classify_integration_points(points)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output - just call to_dict() and dump
    args.output.write_text(yaml_dump(classification.to_dict()), encoding='utf-8')

    # Summary
    print(f"\n✓ Classification complete → {args.output}")
    print(f"  Entry points: {len(classification.entry_points)}")
    print(f"  Intermediate: {len(classification.intermediate_seams)}")
    print(f"  Terminal nodes: {len(classification.terminal_nodes)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
