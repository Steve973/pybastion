#!/usr/bin/env python3
"""
Stage 4: Pattern Analysis

Input: Integration graph from Stage 3
Output: Analysis of flow patterns, common subsequences, and cycles

This stage identifies patterns causing flow explosion:
- Traverses flows up to configured max depth
- Extracts subsequences from long flows (8+ hops)
- Counts frequency of each unique subsequence
- Detects and records all cycles
- Reports most common patterns in descending order

DEFAULT BEHAVIOR (no args):
  - Reads from ./integration-output/stage3-integration-graph.yaml
  - Outputs to ./integration-output/stage4-pattern-analysis.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from collections import Counter

from integration import config

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..shared.yaml_utils import yaml_load, yaml_dump
from ..shared.data_structures import (
    CallableReference,
    CyclePattern,
    GraphNode,
    load_graph_nodes,
    PatternAnalysisResult,
    PatternAnalysisSummary,
    SubsequencePattern
)


def analyze_patterns(
        graph_data: dict[str, Any],
        verbose: bool = False
) -> PatternAnalysisResult:
    """
    Analyze flow patterns to identify explosion causes.

    Traverses flows from entry points, extracting:
    - Common subsequences in long flows (8+ hops)
    - All detected cycles
    - Callable information for pattern interpretation

    Args:
        graph_data: Integration graph from Stage 3
        verbose: Print progress information

    Returns:
        Analysis results dict
    """
    # Load nodes as GraphNode objects
    nodes_list = load_graph_nodes(graph_data)
    edges = graph_data.get('edges', [])
    classification = graph_data.get('classification', {})

    entry_points = set(classification.get('entryPoints', []))
    terminal_nodes = set(classification.get('terminalNodes', []))

    if not nodes_list or not entry_points:
        return PatternAnalysisResult(
            subsequences=[],
            cycles=[],
            flow_length_distribution={},
            summary=PatternAnalysisSummary(
                total_flows_analyzed=0,
                long_flows=0,
                long_flow_threshold=0,
                unique_subsequences=0,
                cycles_detected=0,
                average_flow_length=0.0,
                max_flow_length=0,
                min_flow_length=0
            )
        )

    # Build adjacency list
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        from_id = edge.get('from')
        to_id = edge.get('to')
        if from_id and to_id:
            if from_id not in adjacency:
                adjacency[from_id] = []
            adjacency[from_id].append(to_id)

    # Build node lookup with GraphNode objects
    nodes_by_id: dict[str, GraphNode] = {node.id: node for node in nodes_list}

    # Pattern storage
    subsequence_counts: Counter = Counter()
    cycle_patterns: list[CyclePattern] = []
    flow_lengths: list[int] = []

    max_depth = config.get_pattern_analysis_max_depth()
    long_flow_threshold = config.get_long_flow_threshold()

    if verbose:
        print(f"  Pattern analysis configuration:")
        print(f"    Max depth: {max_depth}")
        print(f"    Long flow threshold: {long_flow_threshold}+ hops")
        print(f"    Subsequence window sizes: 2-7")
        print(f"  Starting analysis from {len(entry_points)} entry points...")

    flows_analyzed = 0
    long_flows_count = 0
    total_paths_explored = 0

    # DFS from each entry point
    for entry_idx, entry_id in enumerate(entry_points):
        if entry_id not in nodes_by_id:
            continue

        progress_pct = ((entry_idx + 1) / len(entry_points)) * 100

        if verbose:
            print(
                f"\r  [{progress_pct:5.1f}%] Entry {entry_idx + 1}/{len(entry_points)}: {entry_id[:25]:25} | Flows: {flows_analyzed:5} | Paths: {total_paths_explored:7}",
                end='', flush=True)

        # Stack: (current_id, path_so_far, visited_indices)
        # visited_indices tracks positions where each node was seen
        stack = [(entry_id, [entry_id], {entry_id: [0]})]

        while stack:
            total_paths_explored += 1
            current_id, path, visited_indices = stack.pop()

            # Update progress every 1000 paths
            if verbose and total_paths_explored % 1000 == 0:
                print(
                    f"\r  [{progress_pct:5.1f}%] Entry {entry_idx + 1}/{len(entry_points)}: {entry_id[:25]:25} | Flows: {flows_analyzed:5} | Paths: {total_paths_explored:7}",
                    end='', flush=True)

            # Check if we've reached a terminal node
            if current_id in terminal_nodes:
                # Complete flow found
                flows_analyzed += 1
                flow_length = len(path)
                flow_lengths.append(flow_length)

                # Extract subsequences if this is a long flow
                if flow_length >= long_flow_threshold:
                    long_flows_count += 1
                    extract_subsequences(path, subsequence_counts)

                continue

            # Check if current node should be excluded from flows (decorated operation)
            current_node = nodes_by_id.get(current_id)
            if current_node and current_node.exclude_from_flows:
                # Treat as terminal - don't traverse into decorated operations
                flows_analyzed += 1
                flow_length = len(path)
                flow_lengths.append(flow_length)

                if flow_length >= long_flow_threshold:
                    long_flows_count += 1
                    extract_subsequences(path, subsequence_counts)

                continue  # Don't explore further from excluded nodes

            # Check depth limit
            if len(path) >= max_depth:
                # Consider this a complete flow for analysis
                flows_analyzed += 1
                flow_length = len(path)
                flow_lengths.append(flow_length)

                if flow_length >= long_flow_threshold:
                    long_flows_count += 1
                    extract_subsequences(path, subsequence_counts)

                continue

            # Explore neighbors
            neighbors = adjacency.get(current_id, [])
            cycle_found = False  # Track if we found a cycle

            for neighbor_id in neighbors:
                # Check if we've seen this node before (cycle detection)
                if neighbor_id in visited_indices:
                    # Cycle detected - we're about to revisit neighbor_id
                    previous_positions = visited_indices[neighbor_id]

                    # For each previous occurrence of this node, extract the cycle
                    for prev_idx in previous_positions:
                        # Extract from first occurrence to current, then add neighbor to close the loop
                        cycle_path = path[prev_idx:] + [neighbor_id]

                        # Verify this is actually a repeating cycle
                        if is_repeating_cycle(
                                cycle_path,
                                path,
                                neighbor_id,
                                adjacency
                        ):
                            # Record the cycle
                            cycle_info = build_cycle_info(
                                cycle_path,
                                nodes_by_id
                            )

                            # Check if we've already recorded this cycle
                            if not cycle_already_recorded(
                                    cycle_info,
                                    cycle_patterns
                            ):
                                cycle_patterns.append(cycle_info)

                            # Mark that we found a cycle
                            cycle_found = True
                            break

                    # If cycle found, stop exploring this path entirely
                    if cycle_found:
                        break

                else:
                    # No cycle, continue exploration
                    new_visited = {
                        k: v.copy() for k, v in visited_indices.items()
                    }
                    if neighbor_id not in new_visited:
                        new_visited[neighbor_id] = []
                    new_visited[neighbor_id].append(len(path))

                    stack.append((
                        neighbor_id,
                        path + [neighbor_id],
                        new_visited
                    ))

    # Move to new line after progress display
    if verbose:
        print()  # Newline after progress

    if verbose:
        print(f"  Analysis complete:")
        print(f"    Total paths explored: {total_paths_explored:,}")
        print(f"    Flows analyzed: {flows_analyzed}")
        print(f"    Long flows (>={long_flow_threshold}): {long_flows_count}")
        print(f"    Unique subsequences found: {len(subsequence_counts)}")
        print(f"    Cycles detected: {len(cycle_patterns)}")

    # Build results
    return build_analysis_results(
        subsequence_counts,
        cycle_patterns,
        flow_lengths,
        nodes_by_id,
        flows_analyzed,
        long_flows_count,
        long_flow_threshold
    )


def extract_subsequences(
        path: list[str],
        subsequence_counts: Counter
) -> None:
    """
    Extract all subsequences of length 2-7 from the path.

    Args:
        path: List of integration IDs
        subsequence_counts: Counter to update with found subsequences
    """
    path_length = len(path)

    # Extract windows of size 2-7
    for window_size in range(2, 8):  # 2, 3, 4, 5, 6, 7
        if window_size > path_length:
            break

        for i in range(path_length - window_size + 1):
            subsequence = tuple(path[i:i + window_size])
            subsequence_counts[subsequence] += 1


def is_repeating_cycle(
        cycle_path: list[str],
        full_path: list[str],
        returning_to: str,
        adjacency: dict[str, list[str]]
) -> bool:
    """
    Verify that a cycle actually repeats.

    A true cycle has the form: A→B→C→A (first and last nodes are the same)

    Args:
        cycle_path: The suspected cycle sequence with repeated node (e.g., [A, B, C, A])
        full_path: Full path traversed so far
        returning_to: Node we're returning to (should match cycle_path[0] and cycle_path[-1])
        adjacency: Graph adjacency list

    Returns:
        True if this forms a valid cycle
    """
    if len(cycle_path) < 3:  # Need at least A→B→A
        return False

    # Verify first and last nodes are the same (forms a cycle)
    first_node = cycle_path[0]
    last_node = cycle_path[-1]

    if first_node != last_node:
        return False

    # Verify all edges in the cycle exist
    for i in range(len(cycle_path) - 1):
        current = cycle_path[i]
        next_node = cycle_path[i + 1]

        neighbors = adjacency.get(current, [])
        if next_node not in neighbors:
            # Invalid cycle - edge doesn't exist
            return False

    # This is a valid cycle
    return True


def cycle_already_recorded(
        cycle_info: CyclePattern,
        recorded_cycles: list[CyclePattern]
) -> bool:
    """
    Check if this cycle pattern was already recorded.

    Args:
        cycle_info: New cycle info
        recorded_cycles: List of already recorded cycles

    Returns:
        True if cycle already recorded
    """
    new_pattern = tuple(cycle_info.pattern)

    for recorded in recorded_cycles:
        recorded_pattern = tuple(recorded.pattern)
        if new_pattern == recorded_pattern:
            # Same cycle, increment occurrence count
            recorded.occurrences += 1
            return True

    return False


def build_cycle_info(
        cycle_path: list[str],
        nodes_by_id: dict[str, GraphNode]
) -> CyclePattern:
    """
    Build detailed cycle information.

    Args:
        cycle_path: List of integration IDs forming the cycle
        nodes_by_id: Node lookup dict with GraphNode objects

    Returns:
        CyclePattern object
    """
    callables: list[CallableReference] = []

    for node_id in cycle_path:
        node = nodes_by_id.get(node_id)
        if not node:
            continue

        target_resolved = node.target_resolved
        unit_name = target_resolved.unit_name or node.source_unit or 'unknown'
        callable_name = target_resolved.callable_name or node.target or 'unknown'

        callables.append(CallableReference(
            integration_id=node_id,
            unit_name=unit_name,
            callable_name=callable_name,
            fully_qualified=f"{unit_name}::{callable_name}"
        ))

    return CyclePattern(
        pattern=cycle_path,
        length=len(cycle_path),
        occurrences=1,
        callables=callables
    )


def build_analysis_results(
        subsequence_counts: Counter,
        cycle_patterns: list[CyclePattern],
        flow_lengths: list[int],
        nodes_by_id: dict[str, GraphNode],
        flows_analyzed: int,
        long_flows_count: int,
        long_flow_threshold: int
) -> PatternAnalysisResult:
    """
    Build final analysis results structure.

    Args:
        subsequence_counts: Counter of subsequence frequencies
        cycle_patterns: List of detected CyclePattern objects
        flow_lengths: List of all flow lengths
        nodes_by_id: Node lookup dict with GraphNode objects
        flows_analyzed: Total flows analyzed
        long_flows_count: Number of long flows
        long_flow_threshold: Minimum length for "long" flow

    Returns:
        PatternAnalysisResult object
    """
    # Build length distribution
    length_distribution = Counter(flow_lengths)

    # Convert subsequences to SubsequencePattern objects
    subsequences: list[SubsequencePattern] = []
    for subseq, count in subsequence_counts.most_common():
        # Map integration IDs to CallableReference objects
        callables: list[CallableReference] = []
        for node_id in subseq:
            node = nodes_by_id.get(node_id)
            if not node:
                continue

            target_resolved = node.target_resolved
            unit_name = target_resolved.unit_name or node.source_unit or 'unknown'
            callable_name = target_resolved.callable_name or node.target or 'unknown'

            callables.append(CallableReference(
                integration_id=node_id,
                unit_name=unit_name,
                callable_name=callable_name,
                fully_qualified=f"{unit_name}::{callable_name}"
            ))

        subsequences.append(SubsequencePattern(
            pattern=list(subseq),
            length=len(subseq),
            occurrences=count,
            callables=callables
        ))

    # Sort cycles by occurrence count (descending)
    sorted_cycles = sorted(cycle_patterns, key=lambda c: c.occurrences, reverse=True)

    summary = PatternAnalysisSummary(
        total_flows_analyzed=flows_analyzed,
        long_flows=long_flows_count,
        long_flow_threshold=long_flow_threshold,
        unique_subsequences=len(subsequences),
        cycles_detected=len(sorted_cycles),
        average_flow_length=sum(flow_lengths) / len(flow_lengths) if flow_lengths else 0,
        max_flow_length=max(flow_lengths) if flow_lengths else 0,
        min_flow_length=min(flow_lengths) if flow_lengths else 0
    )

    return PatternAnalysisResult(
        subsequences=subsequences,
        cycles=sorted_cycles,
        flow_length_distribution=dict(length_distribution),
        summary=summary
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        '--input',
        type=Path,
        default=config.get_stage_output(3),
        help=f'Integration graph from Stage 3 (default: {config.get_stage_output(3)})'
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=config.get_integration_output_dir() / 'stage4-pattern-analysis.yaml',
        help='Output file (default: integration-output/stage4-pattern-analysis.yaml)'
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
        print(f"ERROR: Integration graph not found: {args.input}", file=sys.stderr)
        print("Run Stage 3 first.", file=sys.stderr)
        return 1

    # Load graph
    if args.verbose:
        print(f"Loading integration graph from: {args.input}")

    graph_data = yaml_load(args.input)

    node_count = len(graph_data.get('nodes', []))
    edge_count = len(graph_data.get('edges', []))

    if args.verbose:
        print(f"Loaded graph: {node_count} nodes, {edge_count} edges")

    # Analyze patterns
    if args.verbose:
        print("\nAnalyzing flow patterns...")

    analysis_results = analyze_patterns(graph_data, verbose=args.verbose)

    # Build output
    output_data = {
        'stage': 'pattern-analysis',
        'analysis': analysis_results
    }

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    args.output.write_text(yaml_dump(output_data), encoding='utf-8')

    # Summary
    summary = analysis_results.summary
    print(f"\n✓ Pattern analysis complete → {args.output}")
    print(f"  Flows analyzed: {summary.total_flows_analyzed}")
    print(f"  Long flows (≥{summary.long_flow_threshold}): {summary.long_flows}")
    print(f"  Unique subsequences: {summary.unique_subsequences}")
    print(f"  Cycles detected: {summary.cycles_detected}")

    if summary.average_flow_length > 0:
        print(f"  Avg flow length: {summary.average_flow_length:.1f} hops")
        print(f"  Flow length range: {summary.min_flow_length}-{summary.max_flow_length}")

    # Show top patterns if any
    if analysis_results.subsequences:
        print(f"\n  Top 5 most common subsequences:")
        for i, subseq in enumerate(analysis_results.subsequences[:5], 1):
            chain = ' → '.join([c.callable_name for c in subseq.callables])
            print(f"    {i}. [{subseq.length} hops, {subseq.occurrences}x]: {chain}")

    # Show cycles if any
    if analysis_results.cycles:
        print(f"\n  Detected cycles:")
        for i, cycle in enumerate(analysis_results.cycles[:5], 1):
            chain = ' → '.join([c.callable_name for c in cycle.callables])
            print(f"    {i}. [{cycle.length} hops, {cycle.occurrences}x]: {chain}")

    if not analysis_results.subsequences and not analysis_results.cycles:
        print("\n  NOTE: No problematic patterns detected.")
        print("  All flows are within acceptable length.")

    return 0


if __name__ == '__main__':
    sys.exit(main())