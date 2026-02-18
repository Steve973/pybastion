#!/usr/bin/env python3
"""
Stage 3: Build Integration Graph

Input: Classified integration points from Stage 2 + original points from Stage 1
Output: Integration graph with edges between integration points

This stage constructs edges between integration points using the rule:
  Edge exists from I1 to I2 if I1.target matches I2.source_callable

The graph shows which integration points can lead to other integration points,
forming the basis for flow enumeration in Stage 4.

DEFAULT BEHAVIOR (no args):
  - Reads Stage 1 and Stage 2 outputs
  - Outputs to ./integration-output/stage3-integration-graph.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ..shared.yaml_utils import yaml_load, yaml_dump
from ..shared.ledger_reader import discover_ledgers, load_ledgers, find_ledger_doc
from ..shared.data_structures import (
    IntegrationPoint, GraphNode, IntegrationEdge, IntegrationGraph,
    IntegrationPointClassification, load_integration_points, load_classification, CallableIndexEntry, TargetResolution
)


def check_for_exclusion_decorator(callable_id: str, unit_name: str, ledger_paths: list[Path]) -> tuple[
    bool, str | None]:
    """
    Check if a callable has MechanicalOperation or UtilityOperation decorator.

    Args:
        callable_id: Callable ID to check (e.g., "C000F001")
        unit_name: Unit name to check (e.g., "multiformat")
        ledger_paths: Paths to ledger files

    Returns:
        Tuple of (should_exclude, decorator_type)
        - should_exclude: True if has MechanicalOperation or UtilityOperation
        - decorator_type: "MechanicalOperation" or "UtilityOperation" or None
    """
    # Load ledgers and find the callable
    ledgers = load_ledgers(ledger_paths)

    for ledger_data in ledgers:
        documents = ledger_data['documents']
        ledger_doc = find_ledger_doc(documents)

        if not ledger_doc:
            continue

        unit_entry = ledger_doc.get('unit', {})

        # Check if this is the correct unit
        if unit_entry.get('name') != unit_name:
            continue

        # Walk entry tree to find the callable
        def find_callable(entry: dict) -> dict | None:
            if entry.get('id') == callable_id:
                return entry
            for child in entry.get('children', []):
                result = find_callable(child)
                if result:
                    return result
            return None

        callable_entry = find_callable(unit_entry)

        if callable_entry:
            # Check decorators
            decorators = callable_entry.get('decorators', [])
            for decorator in decorators:
                decorator_name = decorator.get('name', '')
                if decorator_name in ['MechanicalOperation', 'UtilityOperation']:
                    return True, decorator_name

    return False, None


def build_callable_index(ledger_paths: list[Path], verbose: bool = False) -> dict[str, list[CallableIndexEntry]]:
    """
    Build an index of all callables from ledgers for target resolution.

    Returns dict: callable_name -> list of {
        'unit': unit_name,
        'callableId': callable_id,
        'qualifiedName': full_qualified_name
    }

    Args:
        ledger_paths: Paths to ledger files
        verbose: Print progress

    Returns:
        Callable index dict
    """
    index = {}

    ledgers = load_ledgers(ledger_paths)

    if verbose:
        print(f"  Building callable index from {len(ledgers)} ledger(s)...")

    for ledger_data in ledgers:
        documents = ledger_data['documents']
        ledger_doc = find_ledger_doc(documents)

        if not ledger_doc:
            continue

        unit_name = ledger_doc.get('unit', {}).get('name', 'unknown')
        unit_entry = ledger_doc.get('unit', {})

        # Walk the entry tree to find all callables
        def walk_entries(entry: dict, parent_class: str = ''):
            """Recursively walk entry tree to find callables."""
            entry_kind = entry.get('kind')
            entry_name = entry.get('name', '')
            entry_id = entry.get('id')

            # Build qualified name
            if parent_class:
                qualified = f"{parent_class}.{entry_name}"
            else:
                qualified = entry_name

            # If this is a callable or class, add to index
            # Classes are indexed to support constructor call resolution
            if entry_kind in ['callable', 'class'] and entry_id:
                # Index by the simple name
                if entry_name not in index:
                    index[entry_name] = []
                index[entry_name].append(
                    CallableIndexEntry(
                        unit=unit_name,
                        callable_id=entry_id,
                        qualified_name=qualified,
                        fully_qualified=f"{unit_name}::{qualified}"
                    )
                )

                # Also index by qualified name if different
                if qualified != entry_name:
                    if qualified not in index:
                        index[qualified] = []
                    index[qualified].append(
                        CallableIndexEntry(
                            unit=unit_name,
                            callable_id=entry_id,
                            qualified_name=qualified,
                            fully_qualified=f"{unit_name}::{qualified}"
                        )
                    )

            # Track class context for children
            new_parent = qualified if entry_kind == 'class' else parent_class

            # Recurse into children
            for child in entry.get('children', []):
                walk_entries(child, new_parent)

        # Start walking from unit root
        walk_entries(unit_entry)

    if verbose:
        total_entries = sum(len(v) for v in index.values())
        print(f"  Indexed {total_entries} callable entries under {len(index)} names")

    return index


def resolve_target(target: str, callable_index: dict[str, list[CallableIndexEntry]]) -> TargetResolution:
    """
    Resolve a target string to unit/callable information.

    Args:
        target: Target string (e.g., "load_services", "WheelKey.from_mapping")
        callable_index: Index from build_callable_index

    Returns:
        Resolution dict with status, unitId, callableId, name, qualifiedName
    """
    if not target or target not in callable_index:
        return TargetResolution(
            status='unresolved',
            unit_id=None,
            unit_name=None,
            callable_id=None,
            name=None,
            qualified_name=None,
            callable_name=None
        )

    # Get matches
    matches = callable_index[target]

    if len(matches) == 1:
        # Unique match
        match = matches[0]
        return TargetResolution(
            status='resolved',
            unit_id=match.unit,
            unit_name=match.qualified_name,
            callable_id=match.callable_id,
            name=target,
            qualified_name=match.qualified_name,
            callable_name=target.split('.')[-1]
        )
    else:
        # Multiple matches - ambiguous
        return TargetResolution(
            status='ambiguous',
            unit_id=None,
            unit_name=None,
            callable_id=None,
            name=target,
            qualified_name=None,
            callable_name=None,
            matches=[m.fully_qualified for m in matches]
        )


def build_integration_graph(
        points: list[IntegrationPoint],
        classification: IntegrationPointClassification,
        ledger_paths: list[Path],
        verbose: bool = False
) -> IntegrationGraph:
    """
    Build graph with edges between integration points.

    Edge determination rule:
      Edge from I1 to I2 exists if:
        I1.target resolves to the callable that contains I2

    Args:
        points: Integration points from Stage 1
        classification: Classification from Stage 2
        ledger_paths: Paths to ledger files for resolution
        verbose: Print progress information

    Returns:
        IntegrationGraph object
    """
    if not points:
        return IntegrationGraph(
            nodes=[],
            edges=[],
            classification=classification
        )

    # Build callable index for target resolution
    callable_index = build_callable_index(ledger_paths, verbose=verbose)

    # Resolve all targets and create GraphNode objects
    graph_nodes: list[GraphNode] = []
    resolution_stats = {'resolved': 0, 'unresolved': 0, 'ambiguous': 0, 'excluded': 0}

    if verbose:
        print(f"  Resolving {len(points)} integration point targets...")

    for point in points:
        # Resolve the target
        resolution = resolve_target(point.target_raw, callable_index)

        # Check if target callable should be excluded from flows
        should_exclude = False
        fixture_callable_id = None

        if resolution.status == 'resolved':
            target_callable_id = resolution.callable_id
            target_unit_name = resolution.unit_name
            if target_callable_id and target_unit_name:
                should_exclude, decorator_type = check_for_exclusion_decorator(
                    target_callable_id,
                    target_unit_name,
                    ledger_paths
                )
                if should_exclude:
                    fixture_callable_id = target_callable_id
                    if verbose and resolution_stats['excluded'] < 5:  # Show first 5
                        print(
                            f"    Marking {target_unit_name}::{target_callable_id} ({point.target_raw}) for exclusion ({decorator_type})")
                    resolution_stats['excluded'] += 1

        # Track resolution stats
        status = resolution.status or 'unresolved'
        resolution_stats[status] = resolution_stats.get(status, 0) + 1

        # Create GraphNode
        node = GraphNode(
            id=point.id,
            integration_type=point.integration_type,
            source_unit=point.source_unit,
            source_callable_id=point.source_callable_id,
            source_callable_name=point.source_callable_name,
            target=point.target_raw,
            target_resolved=resolution,
            kind=point.kind,
            execution_paths=point.execution_paths,
            condition=point.condition,
            boundary=point.boundary if point.boundary else None,
            signature=point.signature,
            notes=point.notes,
            exclude_from_flows=should_exclude,
            fixture_callable_id=fixture_callable_id
        )
        graph_nodes.append(node)

    if verbose:
        print(f"    Resolved: {resolution_stats.get('resolved', 0)}")
        print(f"    Unresolved: {resolution_stats.get('unresolved', 0)}")
        print(f"    Ambiguous: {resolution_stats.get('ambiguous', 0)}")

    # Build lookup: (source_unit, source_callable_id) -> list of integration IDs
    integrations_by_source_callable: dict[tuple[str, str], list[str]] = {}
    for node in graph_nodes:
        if node.source_unit and node.source_callable_id:
            key = (node.source_unit, node.source_callable_id)
            if key not in integrations_by_source_callable:
                integrations_by_source_callable[key] = []
            integrations_by_source_callable[key].append(node.id)

    # Build edges using resolved targets
    edges: list[IntegrationEdge] = []

    if verbose:
        print(f"  Building edges...")

    for node in graph_nodes:
        resolution = node.target_resolved

        # Skip if not resolved
        if resolution.status != 'resolved':
            continue

        # Get the resolved target unit and callable ID
        target_unit = resolution.unit_name
        target_callable_id = resolution.callable_id

        if not target_unit or not target_callable_id:
            continue

        # Find all integrations that originate from this target (unit, callable)
        key = (target_unit, target_callable_id)
        target_integrations = integrations_by_source_callable.get(key, [])

        for target_integration_id in target_integrations:
            edge = IntegrationEdge(
                from_id=node.id,
                to_id=target_integration_id,
                reason=f'{node.id} calls {node.target}, which contains {target_integration_id}'
            )
            edges.append(edge)

    if verbose:
        print(f"  Built {len(edges)} edges between {len(graph_nodes)} nodes")

    return IntegrationGraph(
        nodes=graph_nodes,
        edges=edges,
        classification=classification
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        '--points',
        type=Path,
        default=config.get_stage_output(1),  # Stage 1 has the full point data
        help=f'Integration points from Stage 1 (default: {config.get_stage_output(1)})'
    )
    ap.add_argument(
        '--classification',
        type=Path,
        default=config.get_stage_output(2),  # Stage 2 has the classification
        help=f'Classification from Stage 2 (default: {config.get_stage_output(2)})'
    )
    ap.add_argument(
        '--ledgers-root',
        type=Path,
        default=config.get_ledgers_root(),
        help=f'Root directory for ledger discovery (default: {config.get_ledgers_root()})'
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=config.get_stage_output(3),
        help=f'Output file (default: {config.get_stage_output(3)})'
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

    # Validate input files exist
    if not args.points.exists():
        print(f"ERROR: Integration points file not found: {args.points}", file=sys.stderr)
        print("Run Stage 1 first.", file=sys.stderr)
        return 1

    if not args.classification.exists():
        print(f"ERROR: Classification file not found: {args.classification}", file=sys.stderr)
        print("Run Stage 2 first.", file=sys.stderr)
        return 1

    # Discover ledgers for target resolution
    if args.verbose:
        print(f"Discovering ledgers in: {args.ledgers_root}")

    ledger_paths = discover_ledgers(
        root=args.ledgers_root,
        structure=config.get_ledger_structure(),
        anchor=config.get_namespace_anchor()
    )

    if not ledger_paths:
        print(f"ERROR: No ledgers found in {args.ledgers_root}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(ledger_paths)} ledger(s)")

    # Load data
    if args.verbose:
        print(f"Loading integration points from: {args.points}")
    points_data = yaml_load(args.points)
    points = load_integration_points(points_data)

    if args.verbose:
        print(f"Loading classification from: {args.classification}")
    classification_data = yaml_load(args.classification)
    classification = load_classification(classification_data)

    if args.verbose:
        print(f"Loaded {len(points)} integration points")

    # Build graph
    if args.verbose:
        print("\nBuilding integration graph...")

    graph = build_integration_graph(
        points,
        classification,
        ledger_paths,
        verbose=args.verbose
    )

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    args.output.write_text(yaml_dump(graph.to_dict()), encoding='utf-8')

    # Summary
    print(f"\n✓ Graph construction complete → {args.output}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")

    if len(graph.edges) == 0:
        print("\n  NOTE: No edges found. This means no integration points lead to others.")
        print("  This could indicate:")
        print("    - All integration points are to external/stdlib code")
        print("    - Integration points don't chain (each is independent)")
        print("    - Target names don't match source callable names (naming mismatch)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
