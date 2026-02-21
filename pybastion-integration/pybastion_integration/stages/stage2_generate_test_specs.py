#!/usr/bin/env python3
"""
Stage 2: Generate Integration Test Specifications

Input:  Stage 1 call graph (stage1-call-graph.yaml)
        Unit ledger files (for EI branch details used in path categorization)
Output: Integration test specifications for AI test generation

For each integration seam edge in the call graph:
  - Reads execution paths from the edge (carried forward from ledger by stage 1)
  - Categorizes paths into: happy_path, error_handling, edge_cases, alternative_flows
  - Reduces to representative paths to avoid redundant test scenarios
  - Identifies fixture requirements (mechanical nodes + intermediate seam nodes)
  - Produces one spec per seam edge with representative test scenarios

Path categorization uses EI branch details from ledgers:
  - Branch outcomes, constraint types, and conditions drive classification
  - Target callable outcome analysis determines error_handling categorization
  - Representative selection: 1 happy path, 1 per error subcategory, all edge cases, up to 3 alternatives

DEFAULT BEHAVIOR (no args):
  - Reads call graph from config.get_stage_output(1)
  - Reads ledgers from config.get_ledgers_root()
  - Outputs to config.get_stage_output(2)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

from pybastion_integration import config


# =============================================================================
# Load and index call graph
# =============================================================================

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def index_graph(graph: dict[str, Any]) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """
    Returns:
        nodes_by_id:      node_id -> node dict
        edges_by_source:  from_node_id -> list of edge dicts
    """
    nodes_by_id: dict[str, dict] = {n['id']: n for n in graph.get('nodes', [])}
    edges_by_source: dict[str, list[dict]] = {}
    for edge in graph.get('edges', []):
        edges_by_source.setdefault(edge['from'], []).append(edge)
    return nodes_by_id, edges_by_source


# =============================================================================
# Ledger loading — EI details and target outcome analysis
# =============================================================================

def discover_ledgers(root: Path) -> list[Path]:
    return sorted(root.glob('**/*.ledger.yaml'))


def load_ledger_doc(path: Path) -> dict | None:
    with open(path, 'r', encoding='utf-8') as f:
        for doc in yaml.safe_load_all(f):
            if doc and doc.get('docKind') == 'ledger':
                return doc
    return None


def build_ei_details_index(ledger_paths: list[Path]) -> dict[str, dict]:
    """
    Build index of EI ID -> branch detail dict from all ledger branch specs.

    Branch detail includes: outcome, condition, constraint_type, is_terminal, terminates_via
    """
    ei_index: dict[str, dict] = {}

    for path in ledger_paths:
        doc = load_ledger_doc(path)
        if not doc:
            continue
        unit = doc.get('unit')
        if not unit:
            continue

        queue = [unit]
        while queue:
            entry = queue.pop(0)
            callable_data = entry.get('callable', {})
            for branch in (callable_data.get('branches') or []):
                if isinstance(branch, dict) and branch.get('id'):
                    ei_index[branch['id']] = branch
            queue.extend(c for c in (entry.get('children') or []) if isinstance(c, dict))

    return ei_index


def build_callable_index(ledger_paths: list[Path]) -> dict[str, dict]:
    """
    Build index of callable_id -> callable entry dict from all ledgers.
    Used for target outcome analysis.
    """
    callable_index: dict[str, dict] = {}

    for path in ledger_paths:
        doc = load_ledger_doc(path)
        if not doc:
            continue
        unit = doc.get('unit')
        if not unit:
            continue

        queue = [unit]
        while queue:
            entry = queue.pop(0)
            entry_id = entry.get('id')
            if entry_id:
                callable_index[entry_id] = entry
            queue.extend(c for c in (entry.get('children') or []) if isinstance(c, dict))

    return callable_index


# =============================================================================
# Path categorization (ported from old stage 3)
# =============================================================================

def analyze_target_outcomes(target_callable: dict | None) -> dict[str, Any]:
    if not target_callable:
        return {
            'has_exceptions': False,
            'exception_branches': [],
            'success_branches': [],
            'total_branches': 0,
        }

    branches = (target_callable.get('callable') or {}).get('branches') or []
    exception_branches = []
    success_branches = []

    for branch in branches:
        outcome = branch.get('outcome', '').lower()
        terminates_via = branch.get('terminates_via', '')
        is_terminal = branch.get('is_terminal', False)

        is_exception = (
            'exception propagates' in outcome or
            'raises' in outcome or
            terminates_via in ('raise', 'exception')
        )

        if is_exception:
            exception_branches.append(branch)
        elif 'returns' in outcome or not is_terminal:
            success_branches.append(branch)

    return {
        'has_exceptions': len(exception_branches) > 0,
        'exception_branches': exception_branches,
        'success_branches': success_branches,
        'total_branches': len(branches),
    }


def categorize_path(
        path_eis: list[str],
        ei_details: dict[str, dict],
        target_outcomes: dict[str, Any],
) -> dict[str, Any]:
    has_validation_failure = False
    has_empty_iteration = False
    has_boundary_condition = False
    has_alternative_branch = False

    for ei_id in path_eis:
        ei = ei_details.get(ei_id, {})
        outcome = ei.get('outcome', '').lower()
        condition = (ei.get('condition') or '').lower()
        constraint_type = (ei.get('constraint') or {}).get('constraint_type')

        if 'validation' in outcome or 'invalid' in outcome:
            has_validation_failure = True

        if constraint_type == 'iteration' and '0 iterations' in outcome:
            has_empty_iteration = True
            has_boundary_condition = True

        if 'is none' in condition or 'is not none' in condition:
            has_boundary_condition = True

        if constraint_type == 'condition':
            has_alternative_branch = True

    target_can_raise = target_outcomes.get('has_exceptions', False)

    if target_can_raise:
        category = 'error_handling'
        subcategory = (
            'triggers_target_validation_error' if has_validation_failure
            else 'triggers_target_exception'
        )
    elif has_empty_iteration or has_boundary_condition:
        category = 'edge_cases'
        subcategory = 'empty_collection' if has_empty_iteration else 'boundary_condition'
    elif has_alternative_branch:
        category = 'alternative_flows'
        subcategory = 'conditional_branch'
    else:
        category = 'happy_path'
        subcategory = 'success'

    return {
        'category': category,
        'subcategory': subcategory,
        'has_validation_failure': has_validation_failure,
        'has_boundary_condition': has_boundary_condition,
        'target_can_raise': target_can_raise,
        'path_length': len(path_eis),
    }


def find_representative_paths(paths: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for path in paths:
        key = (path['category'], path['subcategory'])
        groups.setdefault(key, []).append(path)

    representatives = []

    for (category, subcategory), group_paths in groups.items():
        if category == 'happy_path':
            shortest = min(group_paths, key=lambda p: p['path_length'])
            representatives.append({
                **shortest,
                'represents_count': len(group_paths),
                'representative_of': f'{category}/{subcategory}',
            })

        elif category == 'error_handling':
            representatives.append({
                **group_paths[0],
                'represents_count': len(group_paths),
                'representative_of': f'{category}/{subcategory}',
            })

        elif category == 'edge_cases':
            for path in group_paths:
                representatives.append({
                    **path,
                    'represents_count': 1,
                    'representative_of': f'{category}/{subcategory}',
                })

        else:  # alternative_flows
            for path in group_paths[:3]:
                representatives.append({
                    **path,
                    'represents_count': len(group_paths) // 3 if len(group_paths) > 3 else 1,
                    'representative_of': f'{category}/{subcategory}',
                })

    return representatives


# =============================================================================
# Fixture identification
# =============================================================================

def identify_fixtures(
        source_node_id: str,
        seam_edge: dict,
        nodes_by_id: dict[str, dict],
        edges_by_source: dict[str, list[dict]],
) -> list[dict[str, Any]]:
    """
    Find fixture requirements for a given seam edge.

    Fixtures are needed for other edges from the same source node that:
      1. Point to a mechanical node, or
      2. Are themselves integration seams (intermediate interunit/boundary calls)
    """
    fixtures = []
    seen_targets: set[str] = {seam_edge['to']}

    for edge in edges_by_source.get(source_node_id, []):
        target_id = edge['to']
        if target_id in seen_targets:
            continue

        target_node = nodes_by_id.get(target_id, {})

        if target_node.get('is_mechanical'):
            fixtures.append({
                'type': 'mechanical_operation',
                'mock_target': target_node.get('qualified_name') or target_node.get('name'),
                'mock_target_fqn': target_node.get('fully_qualified'),
                'callable_id': target_node.get('callable_id'),
                'integration_id': edge.get('integration_id'),
                'signature': edge.get('signature'),
                'reason': 'Mechanical operation — mock to avoid test pollution',
            })
            seen_targets.add(target_id)

        elif edge.get('is_integration_seam') and edge['id'] != seam_edge['id']:
            fixtures.append({
                'type': 'interunit_call',
                'mock_target': target_node.get('qualified_name') or edge.get('to_callable'),
                'mock_target_fqn': target_node.get('fully_qualified') or edge.get('to_callable'),
                'callable_id': target_node.get('callable_id'),
                'integration_id': edge.get('integration_id'),
                'signature': edge.get('signature'),
                'reason': 'Intermediate integration seam — mock to isolate the seam under test',
            })
            seen_targets.add(target_id)

    return fixtures


# =============================================================================
# Spec generation
# =============================================================================

def make_spec_id(integration_id: str, counter: int) -> str:
    h = hashlib.sha256(integration_id.encode()).hexdigest()[:8].upper()
    return f"ITEST_{counter:04d}_{h}"


def build_spec(
        edge: dict,
        nodes_by_id: dict[str, dict],
        edges_by_source: dict[str, list[dict]],
        ei_details: dict[str, dict],
        callable_index: dict[str, dict],
        counter: int,
) -> dict[str, Any]:
    source_node = nodes_by_id.get(edge['from'], {})
    target_node = nodes_by_id.get(edge['to'], {})

    integration_id = edge.get('integration_id', '')
    spec_id = make_spec_id(integration_id, counter)

    # Analyze target outcomes for path categorization
    target_callable_id = target_node.get('callable_id')
    target_callable = callable_index.get(target_callable_id) if target_callable_id else None
    target_outcomes = analyze_target_outcomes(target_callable)

    # Categorize and reduce execution paths
    raw_paths = edge.get('execution_paths') or []
    categorized = []
    for i, path_eis in enumerate(raw_paths):
        if not isinstance(path_eis, list):
            continue
        cat_info = categorize_path(path_eis, ei_details, target_outcomes)
        categorized.append({
            'path_id': f'PATH_{i + 1:03d}',
            'eis': path_eis,
            **cat_info,
        })

    representative_paths = find_representative_paths(categorized) if categorized else []

    path_summary = {
        'happy_path': sum(1 for p in categorized if p['category'] == 'happy_path'),
        'error_handling': sum(1 for p in categorized if p['category'] == 'error_handling'),
        'edge_cases': sum(1 for p in categorized if p['category'] == 'edge_cases'),
        'alternative_flows': sum(1 for p in categorized if p['category'] == 'alternative_flows'),
    }

    fixtures = identify_fixtures(edge['from'], edge, nodes_by_id, edges_by_source)

    # Mark fixture EIs in representative paths
    fixture_integration_ids = {f['integration_id'] for f in fixtures if f.get('integration_id')}

    def mark_fixtures(path_eis: list[str]) -> list[str]:
        marked = []
        for i, ei_id in enumerate(path_eis):
            # Last EI in path is the integration point under test — never a fixture
            if i == len(path_eis) - 1:
                marked.append(ei_id)
                continue
            # Integration ID for this EI is "I" + ei_id
            is_fixture = f"I{ei_id}" in fixture_integration_ids
            marked.append(f"{ei_id}_FIXTURE" if is_fixture else ei_id)
        return marked

    marked_paths = []
    for path in representative_paths:
        marked_paths.append({
            **path,
            'eis': mark_fixtures(path.get('eis', [])),
            'eis_original': path.get('eis', []),
        })

    return {
        'spec_id': spec_id,
        'description': f"Test integration: {edge.get('from_callable')} → {edge.get('to_callable')}",
        'test_type': 'integration',
        'integration_point': {
            'id': integration_id,
            'edge_id': edge['id'],
            'type': edge.get('integration_type'),
            'kind': edge.get('integration_kind'),
            'is_integration_seam': edge.get('is_integration_seam'),
            'target_raw': edge.get('target_raw'),
        },
        'feasibility': {
            'feasible_paths': edge.get('feasible_path_count', 0),
            'total_paths': edge.get('total_path_count', 0),
            'representative_paths': len(representative_paths),
            'original_paths': len(raw_paths),
        },
        'source': {
            'unit': source_node.get('unit'),
            'unit_id': source_node.get('unit_id'),
            'callable_id': source_node.get('callable_id'),
            'name': source_node.get('name'),
            'qualified_name': source_node.get('qualified_name'),
            'fully_qualified': source_node.get('fully_qualified'),
            'kind': source_node.get('kind'),
            'signature': source_node.get('signature'),
            'representative_paths': marked_paths,
            'path_summary': path_summary,
        },
        'target': {
            'unit': target_node.get('unit'),
            'unit_id': target_node.get('unit_id'),
            'callable_id': target_node.get('callable_id'),
            'name': target_node.get('name'),
            'qualified_name': target_node.get('qualified_name'),
            'fully_qualified': target_node.get('fully_qualified'),
            'kind': target_node.get('kind'),
            'signature': target_node.get('signature'),
            'unknown': target_node.get('unknown', False),
            'call_signature': edge.get('signature'),
            'outcome_analysis': target_outcomes,
        },
        'fixture_requirements': fixtures,
    }


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
        help='Stage 1 call graph (default: from config)',
    )
    ap.add_argument(
        '--ledgers-root',
        type=Path,
        default=None,
        help='Root directory for ledger discovery (default: from config)',
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file (default: from config)',
    )
    ap.add_argument(
        '--target-root',
        type=Path,
        help='Target project root (sets config defaults)',
    )
    ap.add_argument(
        '--seam-types',
        nargs='+',
        default=['interunit', 'boundary'],
        metavar='TYPE',
        help='Integration types to generate specs for (default: interunit boundary)',
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

    input_path = args.input or config.get_stage_output(1)
    ledgers_root = args.ledgers_root or config.get_ledgers_root()
    output_path = args.output or config.get_stage_output(2)

    if not input_path.exists():
        print(f"ERROR: Call graph not found: {input_path}", file=sys.stderr)
        return 1

    if not ledgers_root.exists():
        print(f"ERROR: Ledgers root not found: {ledgers_root}", file=sys.stderr)
        return 1

    # Load call graph
    if args.verbose:
        print(f"Loading call graph: {input_path}")
    graph = load_yaml(input_path)

    stats = graph.get('stats', {})
    if args.verbose:
        print(f"  Nodes: {stats.get('total_nodes')} ({stats.get('unknown_nodes')} unknown)")
        print(f"  Edges: {stats.get('total_edges')}")
        print(f"  Seam edges: {stats.get('integration_seam_edges')}")

    nodes_by_id, edges_by_source = index_graph(graph)

    # Load ledgers for EI details and target outcome analysis
    ledger_paths = discover_ledgers(ledgers_root)
    if not ledger_paths:
        print(f"ERROR: No ledgers found in {ledgers_root}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"\nLoading {len(ledger_paths)} ledger(s) for EI details...")

    ei_details = build_ei_details_index(ledger_paths)
    callable_index = build_callable_index(ledger_paths)

    if args.verbose:
        print(f"  EI branches indexed: {len(ei_details)}")
        print(f"  Callables indexed: {len(callable_index)}")

    # Filter to seam edges of requested types
    seam_types = set(args.seam_types)
    seam_edges = [
        e for e in graph.get('edges', [])
        if e.get('is_integration_seam') and e.get('integration_type') in seam_types
    ]

    if args.verbose:
        print(f"\nGenerating specs for {len(seam_edges)} seam edges "
              f"(types: {', '.join(sorted(seam_types))})")

    # Generate specs
    specs = []
    total_original_paths = 0
    total_representative_paths = 0

    for i, edge in enumerate(seam_edges, start=1):
        spec = build_spec(edge, nodes_by_id, edges_by_source, ei_details, callable_index, i)
        specs.append(spec)
        total_original_paths += spec['feasibility']['original_paths']
        total_representative_paths += spec['feasibility']['representative_paths']

    reduction_pct = (
        100 * (1 - total_representative_paths / total_original_paths)
        if total_original_paths else 0
    )

    # Fixture summary
    total_fixtures = sum(len(s.get('fixture_requirements', [])) for s in specs)
    mechanical_fixtures = sum(
        sum(1 for f in s.get('fixture_requirements', []) if f['type'] == 'mechanical_operation')
        for s in specs
    )
    interunit_fixtures = sum(
        sum(1 for f in s.get('fixture_requirements', []) if f['type'] == 'interunit_call')
        for s in specs
    )

    # Path category breakdown across all specs
    category_counts: dict[str, int] = {
        'happy_path': 0, 'error_handling': 0, 'edge_cases': 0, 'alternative_flows': 0
    }
    for spec in specs:
        for cat, count in spec['source']['path_summary'].items():
            category_counts[cat] = category_counts.get(cat, 0) + count

    output_data = {
        'stage': 'integration-test-specs',
        'metadata': {
            'spec_count': len(specs),
            'seam_types': sorted(seam_types),
            'input_file': str(input_path),
            'total_original_paths': total_original_paths,
            'total_representative_paths': total_representative_paths,
            'reduction_percentage': round(reduction_pct, 1),
            'total_fixture_requirements': total_fixtures,
            'mechanical_fixtures': mechanical_fixtures,
            'interunit_fixtures': interunit_fixtures,
            'category_breakdown': category_counts,
        },
        'test_specs': specs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            output_data,
            f,
            default_flow_style=False,
            sort_keys=config.get_yaml_sort_keys(),
            width=config.get_yaml_width(),
            indent=config.get_yaml_indent(),
        )

    print(f"\n✓ Generated {len(specs)} test specifications → {output_path}")
    print(f"  Original paths:      {total_original_paths}")
    print(f"  Representative paths: {total_representative_paths}")
    print(f"  Reduction:           {reduction_pct:.1f}%")
    print(f"  Fixture requirements: {total_fixtures} "
          f"({mechanical_fixtures} mechanical, {interunit_fixtures} interunit)")
    if args.verbose:
        print(f"\n  Path categories:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")

    return 0


if __name__ == '__main__':
    sys.exit(main())