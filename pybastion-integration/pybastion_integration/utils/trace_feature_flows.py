#!/usr/bin/env python3
"""
Feature Flow Tracer

Loads the CFG and traces execution paths from FeatureStart to FeatureEnd decorators,
outputs structured YAML with all paths and similarity-based groupings.

Input:  CFG pickle file (from build_cfg.py)
Output: feature-flows.yaml with flows and groupings

Usage:
    python trace_feature_flows.py --cfg cfg.pkl --output feature-flows.yaml
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from pybastion_integration import config


# =============================================================================
# CFG Loading
# =============================================================================

def load_cfg(cfg_path: Path) -> nx.DiGraph:
    """Load the CFG from pickle file."""
    with open(cfg_path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Decorator Analysis
# =============================================================================

def find_nodes_with_decorator(cfg: nx.DiGraph, decorator_name: str) -> list[tuple[str, dict]]:
    """
    Find all nodes that have a specific decorator.

    For callable-level decorators like FeatureStart/FeatureEnd, only returns
    the entry EI (first EI) of each callable, not all EIs.

    Only searches call_node category nodes (actual EIs), not external nodes.

    Returns:
        List of (node_id, node_data) tuples
    """
    matching_nodes = []
    seen_callables = set()

    for node_id, node_data in cfg.nodes(data=True):
        # Only check call nodes (EIs), not external nodes
        if node_data.get('category') != 'call_node':
            continue

        callable_id = node_data.get('callable_id')

        decorators = node_data.get('decorators', [])
        for dec in decorators:
            if dec.get('name') == decorator_name:
                # Only include first EI per callable (entry point)
                if callable_id not in seen_callables:
                    seen_callables.add(callable_id)
                    matching_nodes.append((node_id, node_data))
                break

    return matching_nodes


def get_decorator_kwargs(node_data: dict, decorator_name: str) -> dict[str, Any]:
    """Extract kwargs from a specific decorator on a node."""
    decorators = node_data.get('decorators', [])
    for dec in decorators:
        if dec.get('name') == decorator_name:
            return dec.get('kwargs', {})
    return {}


# =============================================================================
# Path Finding
# =============================================================================

def find_feature_flows(cfg: nx.DiGraph,
                       max_paths_per_flow: int = 20,
                       path_strategy: str = 'shortest') -> list[dict[str, Any]]:
    """
    Find all feature flows from FeatureStart to FeatureEnd.

    Args:
        cfg: Control flow graph
        max_paths_per_flow: Maximum paths to return per feature
        path_strategy: 'shortest' | 'diverse' | 'all'
            - shortest: Return N shortest paths
            - diverse: Return diverse set of paths (different lengths/branches)
            - all: Return all paths up to max (original behavior)

    Returns:
        List of flow dicts with start, end, paths, and metadata
    """
    # Find start and end nodes
    start_nodes = find_nodes_with_decorator(cfg, 'FeatureStart')
    end_nodes = find_nodes_with_decorator(cfg, 'FeatureEnd')

    if not start_nodes:
        print("No FeatureStart decorators found in CFG")
        return []

    if not end_nodes:
        print("No FeatureEnd decorators found in CFG")
        return []

    print(f"Found {len(start_nodes)} FeatureStart node(s)")
    print(f"Found {len(end_nodes)} FeatureEnd node(s)")

    # Find paths between all start/end pairs
    flows = []

    for start_id, start_data in start_nodes:
        start_kwargs = get_decorator_kwargs(start_data, 'FeatureStart')
        feature_name = start_kwargs.get('name', 'unnamed')

        for end_id, end_data in end_nodes:
            end_kwargs = get_decorator_kwargs(end_data, 'FeatureEnd')
            end_feature_name = end_kwargs.get('name', 'unnamed')

            print(f"\nChecking pair: {feature_name} ({start_id}) -> {end_feature_name} ({end_id})")

            # Only match if feature names align
            if feature_name != end_feature_name and feature_name != 'unnamed' and end_feature_name != 'unnamed':
                print(f"  Skipping: feature names don't match")
                continue

            # Find paths based on strategy
            try:
                if path_strategy == 'shortest':
                    print(f"  Finding shortest path...")
                    # Find shortest path first to establish baseline
                    try:
                        shortest = nx.shortest_path(cfg, start_id, end_id)
                        print(f"  Shortest path length: {len(shortest)}")
                        shortest_len = len(shortest)
                    except nx.NetworkXNoPath:
                        # No direct shortest path (probably due to exception branches)
                        # Use all_simple_paths with reasonable cutoff
                        print(f"  No shortest path (exception branches), trying all_simple_paths...")
                        shortest_len = 50  # Reasonable default cutoff

                    # Find paths within 50% of shortest length
                    max_len = int(shortest_len * 1.5)

                    collected_paths = []
                    for path in nx.all_simple_paths(cfg, start_id, end_id, cutoff=max_len):
                        collected_paths.append(path)
                        if len(collected_paths) >= max_paths_per_flow * 2:  # Collect extra to sort
                            break

                    # Sort by length and take shortest N
                    collected_paths.sort(key=len)
                    selected_paths = collected_paths[:max_paths_per_flow]

                elif path_strategy == 'diverse':
                    # Collect paths of varying lengths
                    shortest = nx.shortest_path(cfg, start_id, end_id)
                    shortest_len = len(shortest)
                    max_len = int(shortest_len * 2.0)  # Allow longer paths for diversity

                    collected_paths = []
                    for path in nx.all_simple_paths(cfg, start_id, end_id, cutoff=max_len):
                        collected_paths.append(path)
                        if len(collected_paths) >= max_paths_per_flow * 3:
                            break

                    # Select diverse paths by length buckets
                    collected_paths.sort(key=len)
                    selected_paths = []
                    step = max(1, len(collected_paths) // max_paths_per_flow)
                    for i in range(0, len(collected_paths), step):
                        selected_paths.append(collected_paths[i])
                        if len(selected_paths) >= max_paths_per_flow:
                            break

                else:  # 'all'
                    # Original behavior - take first N paths found
                    selected_paths = []
                    for path in nx.all_simple_paths(cfg, start_id, end_id, cutoff=100):
                        selected_paths.append(path)
                        if len(selected_paths) >= max_paths_per_flow:
                            break

            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                print(f"  Path finding failed: {type(e).__name__}: {e}")
                continue

            if selected_paths:
                print(f"\n  Analyzing {feature_name} paths:")
                flows.append({
                    'feature_name': feature_name,
                    'start_node': start_id,
                    'end_node': end_id,
                    'start_unit': start_data.get('unit'),
                    'start_callable': start_data.get('callable_name'),
                    'end_unit': end_data.get('unit'),
                    'end_callable': end_data.get('callable_name'),
                    'paths': selected_paths,
                    'path_count': len(selected_paths),
                })

    return flows


# =============================================================================
# Path Similarity Analysis
# =============================================================================

def calculate_path_similarity(path1: list[str], path2: list[str]) -> float:
    """
    Calculate similarity between two paths as percentage of shared EIs.

    Returns:
        Similarity score 0.0 to 1.0
    """
    set1 = set(path1)
    set2 = set(path2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def find_divergence_points(cfg: nx.DiGraph, paths: list[list[str]]) -> list[dict[str, Any]]:
    """
    Find points where paths diverge (different successor EIs).

    Returns:
        List of divergence point dicts with EI ID, line, and description
    """
    if not paths:
        return []

    divergences = []
    max_len = max(len(p) for p in paths)

    for i in range(max_len):
        # Get EI at position i for all paths that have it
        eis_at_pos = [p[i] for p in paths if i < len(p)]
        unique_eis = set(eis_at_pos)

        # If more than one unique EI at this position, it's a divergence
        if len(unique_eis) > 1:
            # Get details for first divergent EI
            first_ei = list(unique_eis)[0]
            node_data = cfg.nodes[first_ei]
            constraint = node_data.get('constraint') or {}
            line = constraint.get('line')
            condition = node_data.get('condition', '')
            outcome = node_data.get('outcome', '')

            description = f"{condition} → {outcome}" if condition and outcome else (
                    condition or outcome or "divergence")

            divergences.append({
                'position': i,
                'ei_ids': list(unique_eis),
                'line': line,
                'description': description,
            })

    return divergences


def group_similar_paths(cfg: nx.DiGraph, paths: list[list[str]],
                        similarity_threshold: float = 0.8) -> list[dict[str, Any]]:
    """
    Group paths by similarity.

    Returns:
        List of group dicts with path indices, common EIs, and divergence points
    """
    if not paths:
        return []

    # Start with each path in its own group
    groups: list[list[int]] = [[i] for i in range(len(paths))]

    # Merge similar groups
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                # Compare representative paths from each group
                path_i = paths[groups[i][0]]
                path_j = paths[groups[j][0]]

                if calculate_path_similarity(path_i, path_j) >= similarity_threshold:
                    # Merge groups
                    groups[i].extend(groups[j])
                    groups.pop(j)
                    merged = True
                    break
            if merged:
                break

    # Build group dicts
    result_groups = []
    for group_idx, path_indices in enumerate(groups, start=1):
        # Get all paths in this group
        group_paths = [paths[idx] for idx in path_indices]

        # Find common EIs (intersection of all paths)
        common_eis = set(group_paths[0])
        for path in group_paths[1:]:
            common_eis &= set(path)

        # Find divergence points within this group
        divergences = find_divergence_points(cfg, group_paths)

        result_groups.append({
            'group_id': group_idx,
            'path_count': len(path_indices),
            'path_indices': path_indices,
            'common_ei_count': len(common_eis),
            'divergence_points': divergences,
        })

    return result_groups


# =============================================================================
# YAML Output
# =============================================================================

def extract_path_metadata(cfg: nx.DiGraph, path: list[str]) -> dict[str, Any]:
    """Extract call chain, integration points, and branch points from a path."""
    call_chain = []
    integration_points = []
    branch_points = []

    last_callable_id = None

    for ei_id in path:
        node_data = cfg.nodes[ei_id]
        node_category = node_data.get('category')

        # Handle external nodes differently
        if node_category == 'external_node':
            external_type = node_data.get('type', 'unknown')
            operation_target = node_data.get('operation_target', 'unknown')
            call_chain.append(f"{ei_id}: EXTERNAL_{external_type.upper()}({operation_target})")
            continue

        # Regular call nodes (EIs)
        callable_id = node_data.get('callable_id')
        unit = node_data.get('unit', 'unknown')
        callable_name = node_data.get('callable_name', 'unknown')
        outcome = node_data.get('outcome', '')

        # Show callable change
        if callable_id != last_callable_id:
            last_callable_id = callable_id

        # Show this EI with full path
        call_chain.append(f"{ei_id}: {unit}::{callable_name}::{outcome}")

        # Check for call edges (for integration_points)
        for _, target, edge_data in cfg.out_edges(ei_id, data=True):
            if edge_data.get('edge_type') == 'call':
                integration_type = edge_data.get('integration_type', 'unknown')

                # Get target based on integration type
                if integration_type == 'interunit':
                    target_name = edge_data.get('target_fqn', 'unknown')
                elif integration_type == 'local':
                    target_name = edge_data.get('operation_target', 'unknown')
                else:
                    # External call
                    target_name = edge_data.get('target', edge_data.get('operation_target', 'unknown'))

                integration_points.append(f"{unit}::{callable_name} -> {target_name} ({integration_type})")

        # Branch points
        constraint = node_data.get('constraint')
        if constraint and constraint.get('constraint_type') in ('condition', 'iteration'):
            line = constraint.get('line')
            condition = node_data.get('condition', '')
            outcome = node_data.get('outcome', '')
            if condition or outcome:
                desc = f"{condition} ({outcome})" if condition and outcome else (condition or outcome)
                branch_points.append(f"L{line}: {desc}" if line else desc)

    return {
        'call_chain': call_chain,
        'integration_points': list(dict.fromkeys(integration_points)),
        'branch_points': list(dict.fromkeys(branch_points)),
    }


def build_output_dict(cfg: nx.DiGraph, flows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the complete output dictionary for YAML serialization."""
    output = {
        'flows': [],
        'groupings': [],
    }

    for flow in flows:
        feature_name = flow['feature_name']
        paths = flow['paths']

        # Build flow entry
        flow_entry = {
            'feature_name': feature_name,
            'start_node': flow['start_node'],
            'end_node': flow['end_node'],
            'start_location': f"{flow['start_unit']}::{flow['start_callable']}",
            'end_location': f"{flow['end_unit']}::{flow['end_callable']}",
            'total_paths': flow['path_count'],
            'paths': [],
        }

        # Add all paths with metadata
        for i, path in enumerate(paths, start=1):
            path_id = f"{feature_name}_path_{i:03d}"
            path_metadata = extract_path_metadata(cfg, path)
            flow_entry['paths'].append({
                'path_id': path_id,
                'length': len(path),
                'call_chain': path_metadata['call_chain'],
                'integration_points': path_metadata['integration_points'],
                'branch_points': path_metadata['branch_points'],
            })

        output['flows'].append(flow_entry)

        # Build grouping analysis
        groups = group_similar_paths(cfg, paths)

        grouping_entry = {
            'feature_name': feature_name,
            'total_paths': len(paths),
            'groups': [],
        }

        for group in groups:
            group_dict = {
                'group_id': group['group_id'],
                'path_count': group['path_count'],
                'path_ids': [f"{feature_name}_path_{idx + 1:03d}" for idx in group['path_indices']],
                'common_ei_count': group['common_ei_count'],
                'divergence_count': len(group['divergence_points']),
                'divergence_points': group['divergence_points'],
            }
            grouping_entry['groups'].append(group_dict)

        output['groupings'].append(grouping_entry)

    return output


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--cfg',
        type=Path,
        default=None,
        help='Path to CFG pickle file (default: from config)'
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output YAML file (default: feature-flows.yaml in integration output dir)'
    )
    ap.add_argument(
        '--max-paths',
        type=int,
        default=20,
        help='Maximum paths per feature flow (default: 20)'
    )
    ap.add_argument(
        '--path-strategy',
        choices=['shortest', 'diverse', 'all'],
        default='shortest',
        help='Path selection strategy: shortest (default), diverse, or all'
    )
    ap.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.8,
        help='Similarity threshold for grouping (default: 0.8)'
    )
    ap.add_argument(
        '--target-root',
        type=Path,
        help='Target project root (sets config defaults)'
    )
    ap.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = ap.parse_args(argv)

    if args.target_root:
        config.set_target_root(args.target_root)

    # Determine CFG path
    if args.cfg:
        cfg_path = args.cfg
    else:
        cfg_path = config.get_integration_output_dir() / 'stage1-ei-cfg.pkl'

    if not cfg_path.exists():
        print(f"ERROR: CFG file not found: {cfg_path}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = config.get_integration_output_dir() / 'feature-flows.yaml'

    if args.verbose:
        print(f"Loading CFG from {cfg_path}...")

    cfg = load_cfg(cfg_path)

    if args.verbose:
        print(f"CFG loaded: {cfg.number_of_nodes()} nodes, {cfg.number_of_edges()} edges")

    ####################################################################
    # DEBUGGERY
    ####################################################################
    current = 'U37D3513825_F007_E0009'
    visited = set()
    path = [current]

    for i in range(50):
        if current == 'U37D3513825_F007_E0035':
            print(f"FOUND E0035! At iteration {i}, path length: {len(path)}")
            break

        visited.add(current)
        edges = list(cfg.out_edges(current, data=True))

        if not edges:
            print(f"DEAD END at {current}")
            break

        next_node = None
        for _, target, data in edges:
            if target not in visited:
                next_node = target
                print(f"{current} -> {next_node} ({data.get('edge_type')})")
                break

        if not next_node:
            print(f"CYCLE/NO UNVISITED at {current}")
            break

        path.append(next_node)
        current = next_node
    ####################################################################
    # DEBUGGERY
    ####################################################################

    # Find feature flows
    if args.verbose:
        print("\nSearching for feature flows...")

    flows = find_feature_flows(cfg,
                               max_paths_per_flow=args.max_paths,
                               path_strategy=args.path_strategy)

    if not flows:
        print("\nNo complete feature flows found from FeatureStart to FeatureEnd")
        return 0

    if args.verbose:
        print(f"Found {len(flows)} feature flow(s)")
        for flow in flows:
            print(f"  {flow['feature_name']}: {flow['path_count']} path(s)")

    # Build output structure
    if args.verbose:
        print("\nAnalyzing path similarities and building output...")

    output_dict = build_output_dict(cfg, flows)

    # Write YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            output_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            width=float('inf'),
            indent=config.get_yaml_indent(),
        )

    print(f"\n✓ Feature flow analysis complete → {output_path}")
    print(f"  Features: {len(flows)}")
    print(f"  Total paths: {sum(f['path_count'] for f in flows)}")
    print(f"  Total groups: {sum(len(g['groups']) for g in output_dict['groupings'])}")

    return 0


if __name__ == '__main__':
    sys.exit(main())