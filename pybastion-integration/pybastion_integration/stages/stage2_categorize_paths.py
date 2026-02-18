#!/usr/bin/env python3
"""
Stage 2: Categorize Execution Paths

Input: Stage 1 output (integration points with execution paths)
Output: Categorized and reduced paths for test generation

This stage analyzes execution paths to integration points and categorizes them into:
- happy_path: All operations succeed, no exceptions, primary flow
- error_handling: Exception paths, validation failures
- edge_cases: Boundary conditions, empty collections, special states
- alternative_flows: Valid alternatives (different conditional branches)

Reduces 1000+ paths into meaningful test categories (typically 80-90% reduction).

DEFAULT BEHAVIOR (no args):
  - Reads from ./integration-output/stage1-integration-points.yaml
  - Loads EIS files from ./dist/eis
  - Outputs to ./integration-output/stage2-categorized-paths.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

from shared.data_structures import IntegrationPoint
from shared.yaml_utils import yaml_dump, yaml_load


def load_eis_for_callable(callable_id: str, eis_files: dict[str, Any]) -> dict[str, dict]:
    """
    Load EI details for a specific callable.

    Args:
        callable_id: Callable ID (e.g., "U37D3513825_F002")
        eis_files: Dict of loaded EIS YAML files

    Returns:
        Dict mapping EI IDs to their full details
    """
    eis_by_id = {}

    for eis_data in eis_files.values():
        functions = eis_data.get('functions', [])
        for func in functions:
            # Check if this function contains EIs for our callable
            if func.get('branches'):
                for branch in func['branches']:
                    ei_id = branch.get('id', '')
                    # Check if this EI belongs to our callable
                    if ei_id.startswith(callable_id):
                        eis_by_id[ei_id] = branch

    return eis_by_id


def find_target_callable_in_ledgers(
        unit_id: str,
        callable_id: str,
        ledgers: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """
    Find a target callable in the ledgers by unit_id and callable_id.

    Args:
        unit_id: Unit ID (e.g., "U37D3513825")
        callable_id: Callable ID (e.g., "U37D3513825_C001_M001")
        ledgers: List of loaded ledger data

    Returns:
        Callable entry from ledger, or None if not found
    """
    from shared.ledger_reader import find_ledger_doc

    # First, find the ledger for this unit_id
    target_ledger = None
    for ledger_data in ledgers:
        documents = ledger_data['documents']
        ledger_doc = find_ledger_doc(documents)
        if not ledger_doc:
            continue

        unit = ledger_doc.get('unit', {})
        if unit.get('id') == unit_id:
            target_ledger = ledger_doc
            break

    if not target_ledger:
        return None

    # Walk the unit tree to find the callable
    unit = target_ledger.get('unit')
    if not unit:
        return None

    entries_to_process = [unit]

    while entries_to_process:
        entry = entries_to_process.pop(0)

        if entry.get('id') == callable_id:
            return entry

        children = entry.get('children', [])
        if isinstance(children, list):
            entries_to_process.extend(c for c in children if isinstance(c, dict))

    return None


def analyze_target_outcomes(target_callable: dict[str, Any] | None) -> dict[str, Any]:
    """
    Analyze target callable's branches to identify outcome types.

    Args:
        target_callable: Target callable entry from ledger

    Returns:
        Dict with outcome analysis: has_exceptions, exception_branches, etc.
    """
    if not target_callable:
        return {
            'has_exceptions': False,
            'exception_branches': [],
            'success_branches': [],
            'total_branches': 0
        }

    callable_spec = target_callable.get('callable', {})
    branches = callable_spec.get('branches', [])

    exception_branches = []
    success_branches = []

    for branch in branches:
        outcome = branch.get('outcome', '').lower()
        is_terminal = branch.get('is_terminal', False)
        terminates_via = branch.get('terminates_via', '')

        # Check if this branch raises an exception
        is_exception = (
                'exception propagates' in outcome or
                'raises' in outcome or
                terminates_via == 'raise' or
                terminates_via == 'exception'
        )

        if is_exception:
            exception_branches.append(branch)
        elif 'returns' in outcome or not is_terminal:
            success_branches.append(branch)

    return {
        'has_exceptions': len(exception_branches) > 0,
        'exception_branches': exception_branches,
        'success_branches': success_branches,
        'total_branches': len(branches)
    }


def categorize_path(
        path_eis: list[str],
        ei_details: dict[str, dict],
        target_outcomes: dict[str, Any]
) -> dict[str, Any]:
    """
    Categorize a single execution path.

    Args:
        path_eis: List of EI IDs in the path
        ei_details: Dict mapping EI IDs to their details
        target_outcomes: Analysis of target callable's outcomes

    Returns:
        Dict with category and metadata
    """
    has_validation_failure = False
    has_empty_iteration = False
    has_boundary_condition = False
    has_alternative_branch = False

    for ei_id in path_eis:
        ei = ei_details.get(ei_id, {})
        outcome = ei.get('outcome', '').lower()
        condition = ei.get('condition', '').lower()
        constraint_type = ei.get('constraint_type')

        # Check for validation failures
        if 'validation' in outcome or 'invalid' in outcome:
            has_validation_failure = True

        # Check for boundary conditions
        if constraint_type == 'iteration':
            if '0 iterations' in outcome:
                has_empty_iteration = True
                has_boundary_condition = True

        # Check for None checks
        if 'is none' in condition.lower() or 'is not none' in condition.lower():
            has_boundary_condition = True

        # Check for conditional branches (not exception/operation)
        if constraint_type == 'condition' and 'is true' in outcome:
            has_alternative_branch = True

    # Categorization logic
    category = None
    subcategory = None

    # Check if target can raise exceptions (integration error handling)
    target_can_raise = target_outcomes.get('has_exceptions', False)

    if target_can_raise:
        # Target callable can raise exceptions - this is integration error handling
        category = 'error_handling'
        if has_validation_failure:
            subcategory = 'triggers_target_validation_error'
        else:
            subcategory = 'triggers_target_exception'

    elif has_empty_iteration or has_boundary_condition:
        category = 'edge_cases'
        if has_empty_iteration:
            subcategory = 'empty_collection'
        else:
            subcategory = 'boundary_condition'

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
        'path_length': len(path_eis)
    }


def find_representative_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Find representative paths for each category to reduce test count.

    Args:
        paths: List of categorized path dicts

    Returns:
        Reduced list of representative paths
    """
    # Group by category and subcategory
    groups: dict[tuple[str, str], list[dict]] = {}

    for path in paths:
        key = (path['category'], path['subcategory'])
        if key not in groups:
            groups[key] = []
        groups[key].append(path)

    representatives = []

    for (category, subcategory), group_paths in groups.items():
        if category == 'happy_path':
            # Keep just one happy path (shortest)
            shortest = min(group_paths, key=lambda p: p['path_length'])
            representatives.append({
                **shortest,
                'represents_count': len(group_paths),
                'representative_of': f'{category}/{subcategory}'
            })

        elif category == 'error_handling':
            # Keep one per subcategory
            for path in group_paths[:1]:  # Just the first one
                representatives.append({
                    **path,
                    'represents_count': len(group_paths),
                    'representative_of': f'{category}/{subcategory}'
                })

        elif category == 'edge_cases':
            # Keep all edge cases (they're important)
            for path in group_paths:
                representatives.append({
                    **path,
                    'represents_count': 1,
                    'representative_of': f'{category}/{subcategory}'
                })

        else:  # alternative_flows
            # Keep up to 3 alternative flows
            for path in group_paths[:3]:
                representatives.append({
                    **path,
                    'represents_count': len(group_paths) // 3 if len(group_paths) > 3 else 1,
                    'representative_of': f'{category}/{subcategory}'
                })

    return representatives


def categorize_integration_point(
        integration_point: dict[str, Any],
        eis_files: dict[str, Any],
        ledgers: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Categorize all paths for a single integration point.

    Args:
        integration_point: Integration point dict from stage1
        eis_files: Dict of loaded EIS files
        ledgers: List of loaded ledger data

    Returns:
        Dict with categorized and reduced paths
    """
    source_callable_id = integration_point.get('source_callable_id')
    execution_paths = integration_point.get('execution_paths', [])

    # Get target info for outcome analysis
    target_resolved = integration_point.get('target_resolved', {})
    target_unit_id = target_resolved.get('unit_id')
    target_callable_id = target_resolved.get('callable_id')

    # Analyze target outcomes if we have target info
    target_outcomes = {'has_exceptions': False, 'exception_branches': []}
    if target_unit_id and target_callable_id:
        target_callable = find_target_callable_in_ledgers(
            target_unit_id,
            target_callable_id,
            ledgers
        )
        target_outcomes = analyze_target_outcomes(target_callable)

    if not execution_paths:
        return {
            'integration_id': integration_point.get('id'),
            'total_paths': 0,
            'all_paths': [],
            'representative_paths': [],
            'categorized_paths': [],
            'path_summary': {
                'happy_path': 0,
                'error_handling': 0,
                'edge_cases': 0,
                'alternative_flows': 0
            },
            'reduction_ratio': 0,
            'target_analysis': target_outcomes
        }

    # Load EI details for this callable
    ei_details = load_eis_for_callable(source_callable_id, eis_files)

    # Categorize each path
    categorized = []
    for i, path_eis in enumerate(execution_paths):
        category_info = categorize_path(path_eis, ei_details, target_outcomes)
        categorized.append({
            'path_id': f'PATH_{i + 1:03d}',
            'eis': path_eis,
            **category_info
        })

    # Find representatives
    representatives = find_representative_paths(categorized)

    # Count by category
    summary = {
        'happy_path': sum(1 for p in categorized if p['category'] == 'happy_path'),
        'error_handling': sum(1 for p in categorized if p['category'] == 'error_handling'),
        'edge_cases': sum(1 for p in categorized if p['category'] == 'edge_cases'),
        'alternative_flows': sum(1 for p in categorized if p['category'] == 'alternative_flows'),
    }

    return {
        'integration_id': integration_point.get('id'),
        'source_unit': integration_point.get('source_unit'),
        'source_callable': integration_point.get('source_callable_name'),
        'target': integration_point.get('target'),
        'integration_type': integration_point.get('integration_type'),
        'total_paths': len(execution_paths),
        'all_paths': categorized,
        'representative_paths': representatives,
        'path_summary': summary,
        'reduction_ratio': len(representatives) / len(execution_paths) if execution_paths else 0,
        'target_analysis': target_outcomes
    }


def main(argv: list[str] | None = None) -> int:
    """Categorize paths from stage1 output."""

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        '--input',
        type=Path,
        default=config.get_stage_output(1),
        help=f'Stage 1 input file (default: {config.get_stage_output(1)})'
    )
    ap.add_argument(
        '--eis-root',
        type=Path,
        default=config.get_target_root() / 'dist' / 'eis',
        help='Root directory containing EIS YAML files (default: {target_root}/dist/eis)'
    )
    ap.add_argument(
        '--ledgers-root',
        type=Path,
        default=config.get_ledgers_root(),
        help=f'Root directory for ledgers (default: {config.get_ledgers_root()})'
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=config.get_stage_output(2),
        help=f'Output file (default: {config.get_stage_output(2)})'
    )
    ap.add_argument(
        '--interunit-only',
        action='store_true',
        help='Only process interunit integrations'
    )
    ap.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = ap.parse_args(argv)

    # Check input exists
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Load stage1 data
    if args.verbose:
        print(f"Loading Stage 1 output: {args.input}")

    stage1_data = yaml_load(args.input)

    # Parse integration points
    points_data = stage1_data.get('integration_points', [])
    integration_points = [IntegrationPoint.from_dict(p) for p in points_data]

    if args.verbose:
        print(f"Loaded {len(integration_points)} integration points")

    # Load all EIS files
    if args.verbose:
        print(f"Loading EIS files from {args.eis_root}")

    if not args.eis_root.exists():
        print(f"ERROR: EIS root not found: {args.eis_root}", file=sys.stderr)
        return 1

    eis_files = {}
    for eis_file in args.eis_root.rglob('*_eis.yaml'):
        with open(eis_file) as f:
            eis_files[eis_file.name] = yaml_load(eis_file)

    if args.verbose:
        print(f"Loaded {len(eis_files)} EIS files")

    # Load ledgers for target analysis
    if args.verbose:
        print(f"Loading ledgers from {args.ledgers_root}")

    from shared.ledger_reader import discover_ledgers, load_ledgers

    ledger_paths = discover_ledgers(args.ledgers_root)
    if not ledger_paths:
        print(f"ERROR: No ledgers found in {args.ledgers_root}", file=sys.stderr)
        return 1

    ledgers = load_ledgers(ledger_paths)

    if args.verbose:
        print(f"Loaded {len(ledgers)} ledger(s)")

    # Filter if requested
    if args.interunit_only:
        integration_points = [
            p for p in integration_points
            if p.integration_type == 'interunit'
        ]
        if args.verbose:
            print(f"Filtered to {len(integration_points)} interunit integrations")

    # Categorize each integration point
    categorized_integrations = []
    total_original_paths = 0
    total_representative_paths = 0

    for i, point in enumerate(integration_points):
        if args.verbose and i % 10 == 0:
            print(f"\rProcessing {i+1}/{len(integration_points)}...", end='', flush=True)

        result = categorize_integration_point(point.to_dict(), eis_files, ledgers)
        categorized_integrations.append(result)

        total_original_paths += result['total_paths']
        total_representative_paths += len(result['representative_paths'])

    print(f"\rProcessing {len(integration_points)}/{len(integration_points)}...", end='', flush=True)
    print()

    # Generate category breakdown
    category_counts: dict[str, int] = {
        'happy_path': 0,
        'error_handling': 0,
        'edge_cases': 0,
        'alternative_flows': 0
    }

    # Generate summary
    summary: dict[str, Any] = {
        'total_integration_points': len(integration_points),
        'total_original_paths': total_original_paths,
        'total_representative_paths': total_representative_paths,
        'reduction_percentage': 100 * (
                    1 - total_representative_paths / total_original_paths) if total_original_paths else 0,
        'average_paths_per_integration': total_original_paths / len(integration_points) if integration_points else 0,
        'average_representatives_per_integration': total_representative_paths / len(
            integration_points) if integration_points else 0,
    }

    for integration in categorized_integrations:
        for path in integration.get('all_paths', []):
            cat = path.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # Add to summary dict for YAML output
    summary['category_breakdown'] = category_counts

    # Print to console
    print(f"\nCategory breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_original_paths) * 100 if total_original_paths > 0 else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    output_data = {
        'stage': 'path-categorization',
        'summary': summary,
        'categorized_integrations': categorized_integrations
    }

    with open(args.output, 'w') as f:
        f.write(yaml_dump(output_data))

    print(f"\n✓ Categorized {len(integration_points)} integration points")
    print(f"  Original paths: {total_original_paths}")
    print(f"  Representative paths: {total_representative_paths}")
    print(f"  Reduction: {summary['reduction_percentage']:.1f}%")
    print(f"  → {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())