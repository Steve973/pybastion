#!/usr/bin/env python3
"""
Test script to build fixture EI mapping.

This identifies which execution path EIs call functions with MechanicalOperation
or UtilityOperation decorators, so they can be marked as fixtures.
"""

from pathlib import Path
import yaml
import sys


def find_ledger_doc(documents):
    """Find the ledger document in a multi-doc YAML."""
    for doc in documents:
        if isinstance(doc, dict) and doc.get('docKind') == 'ledger':
            return doc
    return None


def find_derived_ids_doc(documents):
    """Find the derived IDs document in a multi-doc YAML."""
    for doc in documents:
        if isinstance(doc, dict) and doc.get('docKind') == 'derived-ids':
            return doc
    return None


def build_operation_callables_map(ledger_paths):
    """
    Build map of callable IDs that have operation decorators.

    Returns:
        dict: {callable_id: decorator_info}
    """
    operation_callables = {}

    for ledger_path in ledger_paths:
        with open(ledger_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))

        ledger_doc = find_ledger_doc(docs)
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
                            print(f"Found operation callable: {callable_id} -> {name}:{decorator_type}")
                            break

            children = entry.get('children', [])
            if isinstance(children, list):
                entries_to_process.extend(c for c in children if isinstance(c, dict))

    print(f"\nTotal operation callables: {len(operation_callables)}")
    return operation_callables


def build_ei_to_signature_map(ledger_paths):
    """
    Build map of EI IDs to their signatures (what they call).

    Uses the ledger document's branches to get EI outcomes/signatures.

    Returns:
        dict: {ei_id: signature}
    """
    ei_to_signature = {}

    for ledger_path in ledger_paths:
        with open(ledger_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))

        ledger_doc = find_ledger_doc(docs)
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

                # Map each branch EI to its outcome (which contains the call signature)
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


def build_signature_to_callable_map(ledger_paths):
    """
    Build map of function signatures to callable IDs.

    Returns:
        dict: {signature: callable_id}
    """
    signature_to_callable = {}

    for ledger_path in ledger_paths:
        with open(ledger_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))

        ledger_doc = find_ledger_doc(docs)
        if not ledger_doc:
            continue

        unit = ledger_doc.get('unit')
        if not unit:
            continue

        # Walk unit tree to map signatures to callable IDs
        entries_to_process = [unit]

        while entries_to_process:
            entry = entries_to_process.pop(0)

            if entry.get('kind') in ('function', 'method', 'callable'):
                callable_id = entry.get('id')
                name = entry.get('name')
                if callable_id and name:
                    # Map simple name to callable ID
                    signature_to_callable[name] = callable_id

            children = entry.get('children', [])
            if isinstance(children, list):
                entries_to_process.extend(c for c in children if isinstance(c, dict))

    return signature_to_callable


def check_execution_paths(ledger_paths, operation_callables, ei_to_signature, signature_to_callable):
    """
    Check execution paths and identify fixture EIs.

    Returns:
        tuple: (fixture_ei_map, interunit_in_paths)
    """
    fixture_ei_map = {}
    interunit_in_paths = []  # Track actual interunit calls that aren't the last item
    execution_paths_counter = 0

    for ledger_path in ledger_paths:
        with open(ledger_path, 'r') as f:
            docs = list(yaml.safe_load_all(f))

        ledger_doc = find_ledger_doc(docs)
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

                # Check execution paths
                for category in ['interunit', 'stdlib', 'extlib', 'boundaries']:
                    facts = integration.get(category, [])
                    for fact in facts:
                        if isinstance(fact, dict):
                            integration_id = fact.get('id')
                            execution_paths = fact.get('execution_paths', [])

                            # For each execution path
                            for path in execution_paths:
                                execution_paths_counter += 1
                                # For each EI in the path (except the last)
                                for idx, ei_id in enumerate(path[:-1]):
                                    # Get what this EI does
                                    outcome = ei_to_signature.get(ei_id, '')
                                    if not outcome:
                                        continue

                                    # Extract function name from outcome
                                    func_name = outcome.split('(')[0].strip() if '(' in outcome else outcome.strip()

                                    # Look up callable ID
                                    target_callable_id = signature_to_callable.get(func_name)

                                    if target_callable_id:
                                        # Check for operation decorator (fixture)
                                        if target_callable_id in operation_callables:
                                            if ei_id not in fixture_ei_map:
                                                fixture_ei_map[ei_id] = operation_callables[target_callable_id]
                                                print(
                                                    f"Fixture EI: {ei_id} calls {func_name} ({target_callable_id}) -> {operation_callables[target_callable_id]}")

                                        # Check if it's interunit (different unit IDs)
                                        ei_unit_id = (
                                            ei_id.split('_')[0]
                                            if '_' in ei_id
                                            else ''
                                        )
                                        target_unit_id = (
                                            target_callable_id.split('_')[0]
                                            if '_' in target_callable_id
                                            else ''
                                        )

                                        if ei_unit_id and target_unit_id and ei_unit_id != target_unit_id:
                                            # Interunit call mid-path!
                                            interunit_in_paths.append({
                                                'integration_id': integration_id,
                                                'source_unit_id': ei_unit_id,
                                                'target_unit_id': target_unit_id,
                                                'ei_id': ei_id,
                                                'target_callable': func_name,
                                                'position': idx
                                            })

            children = entry.get('children', [])
            if isinstance(children, list):
                entries_to_process.extend(c for c in children if isinstance(c, dict))

    print(f"\nTotal execution paths: {execution_paths_counter}")

    return fixture_ei_map, interunit_in_paths


def main():
    ledgers_root = Path('dist/ledgers')

    if not ledgers_root.exists():
        print(f"ERROR: {ledgers_root} not found")
        return 1

    # Find all ledger files
    ledger_paths = list(ledgers_root.rglob('*.ledger.yaml'))
    print(f"Found {len(ledger_paths)} ledger files\n")

    # Build operation callables map
    print("=== Building operation callables map ===")
    operation_callables = build_operation_callables_map(ledger_paths)

    # Build EI to signature map
    print("\n=== Building EI to signature map ===")
    ei_to_signature = build_ei_to_signature_map(ledger_paths)
    print(f"Total EI signatures: {len(ei_to_signature)}")

    # Build signature to callable map
    print("\n=== Building signature to callable map ===")
    signature_to_callable = build_signature_to_callable_map(ledger_paths)
    print(f"Total signature mappings: {len(signature_to_callable)}")

    # Check execution paths
    print("\n=== Checking execution paths ===")
    fixture_map, interunit_in_paths = check_execution_paths(
        ledger_paths,
        operation_callables,
        ei_to_signature,
        signature_to_callable
    )

    print(f"\n{'=' * 60}")
    print(f"Total fixture EIs: {len(fixture_map)}")
    print(f"{'=' * 60}")

    # Show some examples
    if fixture_map:
        print("\nFirst 10 fixture EIs:")
        for ei_id, decorator_type in list(fixture_map.items())[:10]:
            print(f"  {ei_id} -> {decorator_type}")

    # Report interunit calls in middle of paths
    if interunit_in_paths:
        print(f"\n{'=' * 60}")
        print(f"WARNING: Found {len(interunit_in_paths)} interunit calls in middle of execution paths")
        print(f"{'=' * 60}")
        for item in interunit_in_paths[:20]:  # Show more
            print(f"  {item['source_unit_id']} → {item['target_unit_id']}")
            print(f"    Integration: {item['integration_id']} position {item['position']}")
            print(f"    EI: {item['ei_id']} → {item['target_callable']}")
            print()

    return 0


if __name__ == '__main__':
    sys.exit(main())