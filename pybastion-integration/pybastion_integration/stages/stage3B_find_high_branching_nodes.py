#!/usr/bin/env python3
"""
Analyze Integration Graph for Decorator Candidates

Identifies high-branching TARGET callables that should be decorated with
@MechanicalOperation or @UtilityOperation to reduce flow explosion.

The key insight: we want to decorate callables that are CALLED FROM many
places, not the integration nodes themselves.

Usage:
    python analyze_decorator_candidates.py [graph_file]

    Default: ./integration-output/stage3-integration-graph.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..shared.data_structures import load_graph_nodes, TargetAnalysis, TargetAccumulator
from ..shared.yaml_utils import yaml_load

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml",
          file=sys.stderr)
    sys.exit(1)

# =============================================================================
# Decorator Pattern Recognition
# =============================================================================

# MechanicalOperation patterns - deterministic data transformations
MECHANICAL_SERIALIZATION_PATTERNS = [
    'to_mapping', 'from_mapping', 'to_dict', 'from_dict',
    'serialize', 'deserialize', 'to_json', 'from_json',
    'to_toml', 'from_toml', 'to_yaml', 'from_yaml'
]

MECHANICAL_FORMATTING_PATTERNS = [
    'normalize', '_normalize', 'format', 'sanitize', '_safe_'
]

# UtilityOperation patterns - cross-cutting infrastructure
UTILITY_VALIDATION_PATTERNS = [
    'validate', 'check_schema', 'verify_format', 'assert_valid'
]

UTILITY_LOGGING_PATTERNS = [
    'log', 'logger', 'audit', 'record'
]

UTILITY_CACHING_PATTERNS = [
    'cache', 'cached', 'memoize'
]

UTILITY_HASHING_PATTERNS = [
    'hash', '_hash', 'checksum', 'digest', '_short_hash'
]


def suggest_decorator_type(
        target_name: str,
        callable_name: str | None = None
) -> tuple[str, str]:
    """
    Suggest decorator type based on naming patterns.

    Returns:
        Tuple of (decorator_name, operation_type)
    """
    name = target_name.lower()
    if callable_name:
        name = (name + " " + callable_name.lower())

    # MechanicalOperation patterns
    if any(kw in name for kw in MECHANICAL_SERIALIZATION_PATTERNS):
        return 'MechanicalOperation', 'serialization'

    if any(kw in name for kw in MECHANICAL_FORMATTING_PATTERNS):
        return 'MechanicalOperation', 'formatting'

    # UtilityOperation patterns
    if any(kw in name for kw in UTILITY_VALIDATION_PATTERNS):
        return 'UtilityOperation', 'validation'

    if any(kw in name for kw in UTILITY_LOGGING_PATTERNS):
        return 'UtilityOperation', 'logging'

    if any(kw in name for kw in UTILITY_CACHING_PATTERNS):
        return 'UtilityOperation', 'caching'

    if any(kw in name for kw in UTILITY_HASHING_PATTERNS):
        return 'UtilityOperation', 'hashing'

    # Default
    return 'MechanicalOperation', 'conversion'


def analyze_graph(graph_file: Path, verbose: bool = False) -> dict[str, Any]:
    """
    Analyze integration graph for decorator candidates.

    Groups by TARGET callable and counts incoming edges.

    Args:
        graph_file: Path to Stage 3 graph YAML
        verbose: Print detailed analysis

    Returns:
        Analysis results dict
    """
    if not graph_file.exists():
        print(f"ERROR: Graph file not found: {graph_file}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Loading graph from: {graph_file}")

    data = yaml_load(graph_file)

    nodes_list = load_graph_nodes(data)
    nodes = {n.id: n for n in nodes_list}
    edges = data.get('edges', [])

    # Group edges by target callable using accumulator
    targets: dict[str, TargetAccumulator] = {}

    # First pass: build target groups
    for node in nodes.values():
        target_resolved = node.target_resolved

        if target_resolved.status == 'resolved':
            # Use resolved callable ID as key
            unit = target_resolved.unit_name or 'unknown'
            callable_id = target_resolved.callable_id or 'unknown'
            key = f"{unit}::{callable_id}"

            # Get or create accumulator
            if key not in targets:
                targets[key] = TargetAccumulator(
                    target_name=target_resolved.name or node.target,
                    unit_name=unit,
                    callable_id=callable_id,
                    callable_name=target_resolved.callable_name or node.target,
                    resolved=True
                )
            acc = targets[key]
        else:
            # Unresolved - group by target name
            key = f"UNRESOLVED::{node.target}"
            if key not in targets:
                targets[key] = TargetAccumulator(
                    target_name=node.target,
                    resolved=False
                )
            acc = targets[key]

        # Check if this node is excluded
        if node.exclude_from_flows:
            acc.excluded = True

        # Add this node to incoming list
        acc.incoming_nodes.append(node.id)

    # Second pass: count incoming edges to each target
    for edge in edges:
        from_node_id = edge['from']
        to_node_id = edge['to']

        # Find which target the TO node belongs to
        if to_node_id not in nodes:
            continue

        to_node = nodes[to_node_id]
        target_resolved = to_node.target_resolved

        if target_resolved.status == 'resolved':
            unit = target_resolved.unit_name or 'unknown'
            callable_id = target_resolved.callable_id or 'unknown'
            key = f"{unit}::{callable_id}"
        else:
            key = f"UNRESOLVED::{to_node.target}"

        if key in targets:
            targets[key].incoming_edges += 1

    # Convert to TargetAnalysis list and sort by incoming edges
    target_list: list[TargetAnalysis] = []
    for key, acc in targets.items():
        target_list.append(
            TargetAnalysis(
                key=key,
                target_name=acc.target_name,
                unit_name=acc.unit_name,
                callable_id=acc.callable_id,
                callable_name=acc.callable_name,
                resolved=acc.resolved,
                excluded=acc.excluded,
                incoming_node_count=len(acc.incoming_nodes),
                incoming_edge_count=acc.incoming_edges,
                incoming_nodes=acc.incoming_nodes
            )
        )

    target_list.sort(key=lambda x: x.incoming_edge_count, reverse=True)

    # Separate excluded vs candidates
    already_excluded = [t for t in target_list if t.excluded]
    candidates = [t for t in target_list if not t.excluded]

    # Add decorator suggestions to candidates
    for candidate in candidates:
        decorator, op_type = suggest_decorator_type(
            candidate.target_name,
            candidate.callable_name
        )
        candidate.suggested_decorator = decorator
        candidate.suggested_type = op_type

    # Calculate stats
    total_nodes = len(nodes)
    total_edges = len(edges)
    total_targets = len(targets)

    excluded_count = len(already_excluded)
    excluded_edges = sum(t.incoming_edge_count for t in already_excluded)

    return {
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'total_targets': total_targets,
        'excluded_count': excluded_count,
        'excluded_edges': excluded_edges,
        'already_excluded': [t.to_dict() for t in already_excluded],
        'candidates': [t.to_dict() for t in candidates]
    }


def print_report(
        results: dict[str, Any],
        top_n: int = 20,
        min_edges: int = 5
) -> None:
    """
    Print formatted analysis report.

    Args:
        results: Analysis results from analyze_graph
        top_n: Number of top candidates to show
        min_edges: Minimum edge count to show (filter noise)
    """
    print()
    print("=" * 80)
    print("DECORATOR CANDIDATE ANALYSIS - TARGET CALLABLES")
    print("=" * 80)
    print()

    # Summary stats
    print("GRAPH SUMMARY")
    print("-" * 80)
    print(f"  Total integration nodes: {results['total_nodes']}")
    print(f"  Total edges: {results['total_edges']}")
    print(f"  Unique target callables: {results['total_targets']}")
    print()

    # Exclusion stats
    print("CURRENTLY EXCLUDED TARGETS")
    print("-" * 80)
    excluded_count = results['excluded_count']
    excluded_edges = results['excluded_edges']
    pct_edges = (excluded_edges / results['total_edges'] * 100) \
        if results['total_edges'] else 0

    print(f"  Excluded targets: {excluded_count}")
    print(f"  Edges to excluded targets: {excluded_edges} "
          f"({pct_edges:.1f}% of total)")
    print()

    if results['already_excluded']:
        print("  Top excluded targets:")
        for target in results['already_excluded'][:10]:
            location = f"{target['unit_name']}::{target['callable_name']}" \
                if target['resolved'] else target['target_name']
            print(f"    ✓ {target['incoming_edge_count']:3d} edges | "
                  f"{location}")
        if len(results['already_excluded']) > 10:
            remaining = len(results['already_excluded']) - 10
            print(f"    ... and {remaining} more")
        print()

    # Candidates
    candidates = results['candidates']

    # Filter by minimum edges
    candidates = [c for c in candidates
                  if c['incoming_edge_count'] >= min_edges]

    if not candidates:
        print("NO CANDIDATES FOUND")
        print("-" * 80)
        print(f"  No targets with >={min_edges} incoming edges found!")
        print("  All high-branching operations are already excluded.")
        print()
        return

    print("DECORATION CANDIDATES")
    print("-" * 80)
    print(f"  Found {len(candidates)} targets with >={min_edges} edges")
    print(f"  Showing top {min(top_n, len(candidates))}:")
    print()

    for i, target in enumerate(candidates[:top_n], 1):
        decorator = target['suggested_decorator']
        op_type = target['suggested_type']

        if target['resolved']:
            location = f"{target['unit_name']}::{target['callable_name']}"
            callable_id = target['callable_id']
        else:
            location = f"UNRESOLVED::{target['target_name']}"
            callable_id = "N/A"

        print(f"{i:2d}. {target['incoming_edge_count']:3d} edges | "
              f"{location}")
        print(f"    Callable ID: {callable_id}")
        print(f"    Called from {target['incoming_node_count']} "
              f"integration points")
        print(f"    Suggested: @{decorator} | type={op_type}")
        print()

    if len(candidates) > top_n:
        remaining = len(candidates) - top_n
        remaining_edges = sum(c['incoming_edge_count']
                              for c in candidates[top_n:])
        print(f"  ... and {remaining} more candidates "
              f"({remaining_edges} total edges)")
        print()

    # Impact estimate
    print("ESTIMATED IMPACT")
    print("-" * 80)
    top_candidate_edges = sum(c['incoming_edge_count']
                              for c in candidates[:top_n])
    total_candidate_edges = sum(c['incoming_edge_count']
                                for c in candidates)

    impact_pct = (top_candidate_edges / results['total_edges'] * 100) \
        if results['total_edges'] else 0

    print(f"  Decorating top {min(top_n, len(candidates))} targets "
          f"would exclude:")
    print(f"    ~{top_candidate_edges} edges "
          f"({impact_pct:.1f}% of total)")
    print(f"  All remaining candidates: ~{total_candidate_edges} edges")
    print()

    print("RECOMMENDATION")
    print("-" * 80)
    if candidates[:5]:
        print("  Add decorators to the top 5-10 target callables.")
        print("  The decorator goes in the LEDGER for that callable's unit.")
        print()
        print("  Example additions to ledger files:")
        print()
        for target in candidates[:5]:
            if not target['resolved']:
                print(f"    # UNRESOLVED: {target['target_name']}")
                print(f"    # Cannot provide ledger location")
                print()
                continue

            decorator = target['suggested_decorator']
            op_type = target['suggested_type']
            unit = target['unit_name']
            callable_name = target['callable_name']
            callable_id = target['callable_id']

            print(f"    # In ledger for {unit}.yaml:")
            print(f"    # Find callable {callable_id} ({callable_name})")
            print(f"    - id: {callable_id}")
            print(f"      name: {callable_name}")
            print(f"      decorators:")
            print(f"        - name: {decorator}")
            print(f"          kwargs:")
            print(f"            type: {op_type}")
            print()

    print("=" * 80)
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        'graph_file',
        nargs='?',
        type=Path,
        default=Path('dist/integration-output/stage3-integration-graph.yaml'),
        help='Integration graph from Stage 3 '
             '(default: integration-output/stage3-integration-graph.yaml)'
    )
    ap.add_argument(
        '-n', '--top',
        type=int,
        default=20,
        help='Number of top candidates to show (default: 20)'
    )
    ap.add_argument(
        '-m', '--min-edges',
        type=int,
        default=5,
        help='Minimum edge count to show (default: 5)'
    )
    ap.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    ap.add_argument(
        '--json',
        type=Path,
        help='Export results to JSON file'
    )

    args = ap.parse_args(argv)

    # Analyze graph
    results = analyze_graph(args.graph_file, verbose=args.verbose)

    # Print report
    print_report(results, top_n=args.top, min_edges=args.min_edges)

    # Export JSON if requested
    if args.json:
        import json
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to: {args.json}")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())