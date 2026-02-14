#!/usr/bin/env python3
"""
Stage 5: Generate Sliding Windows

Input: Flows from Stage 4
Output: Sliding windows for integration testing

This stage takes complete flows and generates overlapping test windows.
Each window is a contiguous subsequence of integration points that forms
a testable scope.

Window generation:
- Window size: min_window_length to max_window_length (from config)
- Slides by 1 integration point
- Creates overlapping windows for complete coverage

DEFAULT BEHAVIOR (no args):
  - Reads from ./integration-output/stage4-flows.yaml
  - Outputs to ./integration-output/stage5-test-windows.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from integration import config
from ..shared.yaml_utils import yaml_load, yaml_dump
from ..shared.data_structures import (
    TestWindow, WindowEntryPoint, WindowExitPoint,
    GraphNode, load_flows, Flow
)


def generate_windows(flows_data: dict[str, Any], verbose: bool = False) -> list[TestWindow]:
    """
    Generate sliding windows from flows.

    For each flow, creates overlapping windows of configurable size.
    Windows slide by 1 integration point for complete coverage.

    Args:
        flows_data: Flows from Stage 5
        verbose: Print progress information

    Returns:
        List of TestWindow objects
    """
    flows: list[Flow] = load_flows(flows_data)

    if not flows:
        return []

    min_length = config.get_min_window_length()
    max_length = config.get_max_window_length()

    # If max_length is None, use a reasonable default or flow length
    if max_length is None:
        max_length = 999999  # Effectively unlimited

    if verbose:
        max_display = "unlimited" if max_length == 999999 else str(max_length)
        print(f"  Generating windows from {len(flows)} flows...")
        print(f"  Window size: {min_length} - {max_display} integration points")

    windows: list[TestWindow] = []
    window_counter = 0

    for flow in flows:
        flow_length = len(flow.sequence)

        if flow_length < min_length:
            # Flow too short for minimum window
            if verbose:
                print(f"    Skipping {flow.flow_id}: length {flow_length} < min {min_length}")
            continue

        # Determine window size for this flow
        # Use max_length if flow is long enough, otherwise use flow length
        window_size = min(max_length, flow_length)

        # Generate sliding windows
        for start_idx in range(flow_length - window_size + 1):
            end_idx = start_idx + window_size
            window_sequence: list[GraphNode] = flow.sequence[start_idx:end_idx]

            window_counter += 1
            window_id = f"WINDOW_{window_counter:05d}"

            # Extract integration IDs for the window
            integration_ids = [node.id for node in window_sequence]

            # Build entry and exit points
            entry_node = window_sequence[0]
            exit_node = window_sequence[-1]

            entry_point = WindowEntryPoint(
                integration_id=entry_node.id,
                unit=entry_node.source_unit,
                callable=entry_node.source_callable_name,
                target=entry_node.target
            )

            exit_point = WindowExitPoint(
                integration_id=exit_node.id,
                unit=exit_node.source_unit,
                callable=exit_node.source_callable_name,
                target=exit_node.target,
                is_boundary=exit_node.boundary is not None
            )

            # Build window description
            window_desc = ' → '.join([node.target for node in window_sequence])

            windows.append(
                TestWindow(
                    window_id=window_id,
                    source_flow_id=flow.flow_id,
                    start_position=start_idx,
                    length=len(window_sequence),
                    integration_ids=integration_ids,
                    entry_point=entry_point,
                    exit_point=exit_point,
                    description=window_desc,
                    sequence=window_sequence
                )
            )

    if verbose:
        print(f"  Generated {len(windows)} windows")

    return windows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        '--input',
        type=Path,
        default=config.get_stage_output(5),
        help=f'Flows from Stage 5 (default: {config.get_stage_output(5)})'
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=config.get_stage_output(6),
        help=f'Output file (default: {config.get_stage_output(6)})'
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
        print(f"ERROR: Flows file not found: {args.input}", file=sys.stderr)
        print("Run Stage 4 first.", file=sys.stderr)
        return 1

    # Load flows
    if args.verbose:
        print(f"Loading flows from: {args.input}")

    flows_data = yaml_load(args.input)

    flow_count = len(flows_data.get('flows', []))
    if args.verbose:
        print(f"Loaded {flow_count} flows")

    # Generate windows
    if args.verbose:
        print("\nGenerating windows...")

    windows = generate_windows(flows_data, verbose=args.verbose)

    # Build output
    output_data: dict[str, Any] = {
        'stage': 'test-window-generation',
        'windows': [w.to_dict() for w in windows]
    }

    if config.include_metadata():
        # Calculate stats
        total_integrations = sum(w.length for w in windows)
        avg_length = total_integrations / len(windows) if windows else 0

        # Count unique flows covered
        unique_flows = len(set(w.source_flow_id for w in windows))

        output_data['metadata'] = {
            'window_count': len(windows),
            'source_flow_count': flow_count,
            'flows_covered': unique_flows,
            'average_window_length': round(avg_length, 2),
            'total_integration_points': total_integrations,
            'min_window_length': config.get_min_window_length(),
            'max_window_length': config.get_max_window_length()
        }

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    args.output.write_text(yaml_dump(output_data), encoding='utf-8')

    # Summary
    print(f"\n✓ Window generation complete → {args.output}")
    print(f"  Total windows: {len(windows)}")
    print(f"  From {flow_count} flows")

    if windows:
        lengths = [w.length for w in windows]
        print(f"  Window sizes: {min(lengths)} - {max(lengths)} integration points")

    return 0


if __name__ == '__main__':
    sys.exit(main())