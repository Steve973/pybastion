#!/usr/bin/env python3
"""
Stage 3: Generate Integration Test Specifications

Input: Stage 2 output (categorized paths)
Output: Integration test specifications for AI test generation

This stage takes categorized integration points and enriches them with:
- Target callable information from ledgers
- Fixture requirements (mechanical operations + intermediate interunit calls)
- Complete signatures, parameters, and test context

Each spec contains:
- Source callable info (who's calling)
- Target callable info (what's being called)
- Categorized execution paths with fixtures marked
- Fixture requirements (what to mock/stub)
- Target's possible outcomes (from its EIs)

DEFAULT BEHAVIOR (no args):
  - Reads from ./integration-output/stage2-categorized-paths.yaml
  - Loads ledgers from ./ledgers
  - Outputs to ./integration-output/stage3-test-specs.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

from shared.ledger_reader import discover_ledgers, load_ledgers, find_ledger_doc
from shared.yaml_utils import yaml_dump, yaml_load


def find_target_callable(
        unit_id: str,
        callable_id: str,
        ledgers: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """
    Find a callable in the ledgers by unit_id and callable_id.

    Args:
        unit_id: Unit ID (e.g., "U37D3513825")
        callable_id: Callable ID (e.g., "U37D3513825_C001_M001")
        ledgers: List of loaded ledger data

    Returns:
        Callable entry from ledger, or None if not found
    """
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

    # Now walk the unit tree to find the callable with this callable_id
    unit = target_ledger.get('unit')
    if not unit:
        return None

    # BFS through the unit tree
    entries_to_process = [unit]

    while entries_to_process:
        entry = entries_to_process.pop(0)

        if entry.get('id') == callable_id:
            return entry

        children = entry.get('children', [])
        if isinstance(children, list):
            entries_to_process.extend(c for c in children if isinstance(c, dict))

    return None


def build_operation_callables_map(ledgers: list[dict[str, Any]]) -> dict[str, str]:
    """
    Build map of callable IDs that have MechanicalOperation or UtilityOperation decorators.

    Args:
        ledgers: List of loaded ledger data

    Returns:
        Dict mapping callable_id to decorator info (e.g., "MechanicalOperation:validation")
    """
    operation_callables = {}

    for ledger_data in ledgers:
        documents = ledger_data['documents']
        ledger_doc = find_ledger_doc(documents)
        if not ledger_doc:
            continue

        unit = ledger_doc.get('unit')
        if not unit:
            continue

        # Walk unit tree to find callables with operation decorators
        entries_to_process = [unit]

        while entries_to_process:
            entry = entries_to_process.pop(0)

            if entry.get('kind') in ('function', 'method', 'callable'):
                callable_id = entry.get('id')
                decorators = entry.get('decorators', [])

                for decorator in decorators:
                    if isinstance(decorator, dict):
                        name = decorator.get('name', '')
                        if name in ('MechanicalOperation', 'UtilityOperation'):
                            decorator_type = decorator.get('kwargs', {}).get('type', 'unknown')
                            operation_callables[callable_id] = f"{name}:{decorator_type}"
                            break

            children = entry.get('children', [])
            if isinstance(children, list):
                entries_to_process.extend(c for c in children if isinstance(c, dict))

    return operation_callables


def build_ei_to_signature_map(ledgers: list[dict[str, Any]]) -> dict[str, str]:
    """
    Build map of EI IDs to their signatures (what they call).

    Args:
        ledgers: List of loaded ledger data

    Returns:
        Dict mapping ei_id to outcome/signature
    """
    ei_to_signature = {}

    for ledger_data in ledgers:
        documents = ledger_data['documents']
        ledger_doc = find_ledger_doc(documents)
        if not ledger_doc:
            continue

        unit = ledger_doc.get('unit')
        if not unit:
            continue

        # Walk unit tree to get all callable branches
        entries_to_process = [unit]

        while entries_to_process:
            entry = entries_to_process.pop(0)

            if entry.get('kind') in ('function', 'method', 'callable'):
                callable_spec = entry.get('callable', {})
                branches = callable_spec.get('branches', [])

                # Map each branch EI to its outcome
                for branch in branches:
                    if isinstance(branch, dict):
                        ei_id = branch.get('id')
                        outcome = branch.get('outcome', '')
                        if ei_id and outcome:
                            ei_to_signature[ei_id] = outcome

            children = entry.get('children', [])
            if isinstance(children, list):
                entries_to_process.extend(c for c in children if isinstance(c, dict))

    return ei_to_signature


def build_signature_to_callable_map(ledgers: list[dict[str, Any]]) -> dict[str, str]:
    """
    Build map of function names to callable IDs.

    Args:
        ledgers: List of loaded ledger data

    Returns:
        Dict mapping function name to callable_id
    """
    signature_to_callable = {}

    for ledger_data in ledgers:
        documents = ledger_data['documents']
        ledger_doc = find_ledger_doc(documents)
        if not ledger_doc:
            continue

        unit = ledger_doc.get('unit')
        if not unit:
            continue

        # Walk unit tree to map names to callable IDs
        entries_to_process = [unit]

        while entries_to_process:
            entry = entries_to_process.pop(0)

            if entry.get('kind') in ('function', 'method', 'callable'):
                callable_id = entry.get('id')
                name = entry.get('name')
                if callable_id and name:
                    signature_to_callable[name] = callable_id

            children = entry.get('children', [])
            if isinstance(children, list):
                entries_to_process.extend(c for c in children if isinstance(c, dict))

    return signature_to_callable


def identify_fixture_eis(
        execution_paths: list[dict[str, Any]],
        operation_callables: dict[str, str],
        ei_to_signature: dict[str, str],
        signature_to_callable: dict[str, str],
        all_integration_facts: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    Identify which EIs in execution paths need fixtures.

    An EI needs a fixture if:
    1. It calls a function with MechanicalOperation/UtilityOperation decorator
    2. It's an interunit call in the middle of a path (not the last EI)

    Args:
        execution_paths: List of path dicts with 'eis' field
        operation_callables: Map of callable_id to decorator info
        ei_to_signature: Map of EI to what it calls
        signature_to_callable: Map of function name to callable_id
        all_integration_facts: Map of integration_id to integration fact (for qualified names)

    Returns:
        Dict mapping EI ID to fixture info
    """
    fixture_requirements = {}

    for path_data in execution_paths:
        path_eis = path_data.get('eis', [])

        # Check each EI except the last (which is the integration point)
        for idx, ei_id in enumerate(path_eis[:-1]):
            if ei_id in fixture_requirements:
                continue  # Already identified

            # Get what this EI does
            outcome = ei_to_signature.get(ei_id, '')
            if not outcome:
                continue

            # Extract function name from outcome
            func_name = outcome.split('(')[0].strip() if '(' in outcome else outcome.strip()

            # Remove common prefixes/suffixes
            for prefix in ['executes → ', 'succeeds → ']:
                if func_name.startswith(prefix):
                    func_name = func_name[len(prefix):]

            # Look up callable ID
            target_callable_id = signature_to_callable.get(func_name)

            if not target_callable_id:
                continue

            # Check for mechanical operation decorator
            if target_callable_id in operation_callables:
                fixture_requirements[ei_id] = {
                    'type': 'mechanical_operation',
                    'mock_target': func_name,  # Unqualified for local functions
                    'decorator': operation_callables[target_callable_id],
                    'reason': 'Function is mechanical - mock it directly'
                }
                continue

            # Check if it's an interunit call (different unit IDs)
            ei_unit_id = ei_id.split('_')[0] if '_' in ei_id else ''
            target_unit_id = target_callable_id.split('_')[0] if '_' in target_callable_id else ''

            if ei_unit_id and target_unit_id and ei_unit_id != target_unit_id:
                # Try to get the qualified name from integration facts
                integration_id = f"I{ei_id}"
                integration_fact = all_integration_facts.get(integration_id, {})
                qualified_target = integration_fact.get('target', func_name)

                fixture_requirements[ei_id] = {
                    'type': 'interunit_call',
                    'mock_target': qualified_target,
                    'target_callable_id': target_callable_id,
                    'reason': f'{ei_id} calls {func_name} - mock the callee'
                }

    return fixture_requirements


def create_test_spec(
        categorized_integration: dict[str, Any],
        ledgers: list[dict[str, Any]],
        fixture_maps: dict[str, Any],
        all_integration_facts: dict[str, dict[str, Any]],
        spec_counter: int
) -> dict[str, Any]:
    """
    Create an integration test specification from categorized integration.

    Args:
        categorized_integration: Integration dict from stage2
        ledgers: List of loaded ledger data
        fixture_maps: Dict with operation_callables, ei_to_signature, signature_to_callable
        all_integration_facts: Map of integration_id to integration fact data
        spec_counter: Counter for generating spec IDs

    Returns:
        Test specification dictionary
    """
    spec_id = f"ITEST_{spec_counter:04d}"

    integration_id = categorized_integration.get('integration_id', 'unknown')
    source_unit = categorized_integration.get('source_unit', 'unknown')
    source_callable = categorized_integration.get('source_callable', 'unknown')
    target_raw = categorized_integration.get('target', 'unknown')
    integration_type = categorized_integration.get('integration_type', 'unknown')

    # Get categorized paths
    all_paths = categorized_integration.get('all_paths', [])
    representative_paths = categorized_integration.get('representative_paths', [])
    path_summary = categorized_integration.get('path_summary', {})
    target_analysis = categorized_integration.get('target_analysis', {})

    # Identify fixtures for all paths
    fixture_requirements = identify_fixture_eis(
        all_paths,
        fixture_maps['operation_callables'],
        fixture_maps['ei_to_signature'],
        fixture_maps['signature_to_callable'],
        all_integration_facts
    )

    # Mark fixtures in paths by appending _FIXTURE
    def mark_fixtures_in_path(path_eis: list[str]) -> list[str]:
        """Append _FIXTURE to EI IDs that need fixtures."""
        marked = []
        for ei_id in path_eis:
            if ei_id in fixture_requirements:
                marked.append(f"{ei_id}_FIXTURE")
            else:
                marked.append(ei_id)
        return marked

    # Mark fixtures in representative paths
    marked_representative_paths = []
    for path in representative_paths:
        marked_path = {
            **path,
            'eis': mark_fixtures_in_path(path.get('eis', [])),
            'eis_original': path.get('eis', [])  # Keep original for reference
        }
        marked_representative_paths.append(marked_path)

    # Build source info
    source_info = {
        'unit': source_unit,
        'callable': source_callable,
        'representative_paths': marked_representative_paths,
        'path_summary': path_summary,
        'total_paths': categorized_integration.get('total_paths', 0),
        'representative_count': len(representative_paths)
    }

    # Build target info - try to load from stage1 data if available
    # For now, just use what we have from categorized integration
    target_info: dict[str, Any] = {
        'fqn': target_raw,
    }

    # Try to parse target from integration_id or other fields
    # This is a simplified version - stage1 had better target resolution
    target_info['type'] = integration_type

    # Add target analysis from stage2
    if target_analysis:
        target_info['outcome_analysis'] = target_analysis

    spec = {
        'spec_id': spec_id,
        'description': f"Test integration: {source_unit}.{source_callable} → {target_raw}",
        'test_type': 'integration',
        'integration_point': {
            'id': integration_id,
            'type': integration_type,
        },
        'source': source_info,
        'target': target_info,
        'fixture_requirements': fixture_requirements
    }

    return spec


def build_integration_facts_map(ledgers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Build map of integration IDs to their facts (for getting qualified target names).

    Args:
        ledgers: List of loaded ledger data

    Returns:
        Dict mapping integration_id to integration fact
    """
    integration_facts = {}

    for ledger_data in ledgers:
        documents = ledger_data['documents']
        ledger_doc = find_ledger_doc(documents)
        if not ledger_doc:
            continue

        unit = ledger_doc.get('unit')
        if not unit:
            continue

        # Walk unit tree to find integration facts
        entries_to_process = [unit]

        while entries_to_process:
            entry = entries_to_process.pop(0)

            if entry.get('kind') in ('function', 'method', 'callable'):
                callable_spec = entry.get('callable', {})
                integration = callable_spec.get('integration', {})

                # Extract integration facts from all categories
                for category in ['interunit', 'stdlib', 'extlib', 'boundaries']:
                    facts = integration.get(category, [])
                    for fact in facts:
                        if isinstance(fact, dict):
                            integration_id = fact.get('id')
                            if integration_id:
                                integration_facts[integration_id] = fact

            children = entry.get('children', [])
            if isinstance(children, list):
                entries_to_process.extend(c for c in children if isinstance(c, dict))

    return integration_facts


def generate_test_specs(
        stage2_output: Path,
        ledger_paths: list[Path],
        verbose: bool = False
) -> list[dict[str, Any]]:
    """
    Generate integration test specifications.

    Args:
        stage2_output: Path to Stage 2 output file
        ledger_paths: Paths to ledger files
        verbose: Print progress information

    Returns:
        List of test specification dictionaries
    """
    # Load Stage 2 categorized integrations
    if verbose:
        print(f"Loading Stage 2 output: {stage2_output}")

    stage2_data = yaml_load(stage2_output)

    categorized_integrations = stage2_data.get('categorized_integrations', [])

    if verbose:
        print(f"Loaded {len(categorized_integrations)} categorized integrations")

    # Load ledgers
    if verbose:
        print(f"Loading {len(ledger_paths)} ledger(s)")

    ledgers = load_ledgers(ledger_paths)

    # Build fixture detection maps
    if verbose:
        print("Building fixture detection maps...")

    operation_callables = build_operation_callables_map(ledgers)
    ei_to_signature = build_ei_to_signature_map(ledgers)
    signature_to_callable = build_signature_to_callable_map(ledgers)
    integration_facts = build_integration_facts_map(ledgers)

    fixture_maps = {
        'operation_callables': operation_callables,
        'ei_to_signature': ei_to_signature,
        'signature_to_callable': signature_to_callable
    }

    if verbose:
        print(f"  Operation callables: {len(operation_callables)}")
        print(f"  EI signatures: {len(ei_to_signature)}")
        print(f"  Function mappings: {len(signature_to_callable)}")
        print(f"  Integration facts: {len(integration_facts)}")

    # Filter to only interunit integrations
    interunit_integrations = [
        integration for integration in categorized_integrations
        if integration.get('integration_type') == 'interunit'
    ]

    if verbose:
        print(f"Generating specs for {len(interunit_integrations)} interunit integrations")

    # Generate test specs
    specs = []
    spec_counter = 1

    for integration in interunit_integrations:
        spec = create_test_spec(integration, ledgers, fixture_maps, integration_facts, spec_counter)
        specs.append(spec)
        spec_counter += 1

    return specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        '--input',
        type=Path,
        default=config.get_stage_output(2),
        help=f'Stage 2 input file (default: {config.get_stage_output(2)})'
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
        default=config.get_stage_output(3),
        help=f'Output file (default: {config.get_stage_output(3)})'
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

    # Discover ledgers
    if args.verbose:
        print(f"Discovering ledgers in: {args.ledgers_root}")

    ledger_paths = discover_ledgers(args.ledgers_root)

    if not ledger_paths:
        print(f"ERROR: No ledgers found in {args.ledgers_root}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(ledger_paths)} ledger(s)")

    # Generate test specs
    specs = generate_test_specs(args.input, ledger_paths, verbose=args.verbose)

    if args.verbose:
        print(f"\nGenerated {len(specs)} test specifications")

        # Count fixture requirements
        total_fixtures = sum(len(spec.get('fixture_requirements', {})) for spec in specs)
        print(f"Total fixture requirements: {total_fixtures}")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    output_data = {
        'stage': 'integration-test-specs',
        'test_specs': specs,
        'metadata': {
            'spec_count': len(specs),
            'input_file': str(args.input),
            'total_fixture_requirements': sum(len(spec.get('fixture_requirements', {})) for spec in specs)
        },
    }

    args.output.write_text(yaml_dump(output_data), encoding='utf-8')
    print(f"\n✓ Generated {len(specs)} test specifications → {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())