#!/usr/bin/env python3
"""
Stage 1: Collect Integration Points

Input: Collection of unit ledger files (auto-discovered or explicit)
Output: Flat list of all integration points with resolved targets

This stage extracts all integration facts (interunit and boundary) from
unit ledgers and creates IntegrationPoint objects for each one. It also
resolves interunit targets using the callable inventory.

Target Resolution:
- interunit: Resolved using callable-inventory.txt (unit_id + callable_id)
- stdlib: Marked as 'stdlib' status
- extlib: Marked as 'extlib' status
- boundary: Marked as 'boundary' status
- unresolved: Only for interunit targets not found in inventory

DEFAULT BEHAVIOR (no args):
  - Discovers ledgers in ./ledgers
  - Loads callable inventory from ./dist/inspect/callable-inventory.txt
  - Outputs to ./integration-output/stage1-integration-points.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Add integration directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

from shared.data_structures import IntegrationPoint, TargetRef, BoundarySummary, IntegrationPointCollection
from shared.ledger_reader import discover_ledgers, load_ledgers, find_ledger_doc, extract_integration_facts
from shared.yaml_utils import yaml_dump


def load_callable_inventory(inventory_path: Path | None = None) -> dict[str, tuple[str, str]]:
    """
    Load callable inventory from text file.

    Format: fully.qualified.name:UNIT_ID_CALLABLE_ID
    Example: project.module.ClassName.method:U37D3513825_C001_M001

    Args:
        inventory_path: Path to callable-inventory.txt
                       (default: {target_root}/dist/inspect/callable-inventory.txt)

    Returns:
        Dict mapping fully qualified names to (unit_id, callable_id) tuples
    """
    if inventory_path is None:
        inventory_path = config.get_target_root() / 'dist' / 'inspect' / 'callable-inventory.txt'

    if not inventory_path.exists():
        # No inventory file found - return empty dict
        print(f"WARNING: Callable inventory not found: {inventory_path}")
        return {}

    inventory = {}

    with inventory_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse: fully.qualified.name:UNIT_ID_CALLABLE_ID
            if ':' not in line:
                continue

            qualified_name, combined_id = line.split(':', 1)

            # Extract unit_id and callable_id
            # Combined format could be:
            #   - U12345_C001 (class)
            #   - U12345_C001_M001 (method)
            #   - U12345_F001 (function)
            parts = combined_id.split('_', 1)
            if len(parts) == 2:
                unit_id = parts[0]  # e.g., "U12345"
                callable_id = combined_id  # e.g., "U12345_C001_M001"
                inventory[qualified_name] = (unit_id, callable_id)

    return inventory


def create_integration_point(fact: dict[str, Any], callable_inventory: dict[str, tuple[str, str]]) -> IntegrationPoint:
    """
    Create an IntegrationPoint object from an extracted integration fact.

    Args:
        fact: Integration fact dictionary from extract_integration_facts
        callable_inventory: Dict mapping qualified names to (unit_id, callable_id)

    Returns:
        IntegrationPoint object
    """
    # Extract basic fields
    integration_id = fact.get('id', 'unknown')
    source_unit = fact.get('source_unit', 'unknown')
    source_callable_id = fact.get('source_callable_id', 'unknown')
    source_callable_name = fact.get('source_callable_name', 'unknown')
    target_raw = fact.get('target', '')
    kind = fact.get('kind', 'call')
    integration_type = fact.get('integration_type', 'unknown')

    # Extract execution paths
    execution_paths = fact.get('execution_paths', [])
    if not isinstance(execution_paths, list):
        execution_paths = []

    # Extract optional fields
    condition = fact.get('condition')
    signature = fact.get('signature')
    notes = fact.get('notes')

    # Resolve the target based on integration type
    target_resolved: TargetRef

    if integration_type == 'interunit':
        # Lookee lookee in the inventory!
        if target_raw in callable_inventory:
            unit_id, callable_id = callable_inventory[target_raw]
            # Extract unit name from target (everything before last dot usually)
            # e.g., "project.module.ClassName.method" -> unit is "project.module"
            parts = target_raw.rsplit('.', 1)
            callable_name = parts[1] if len(parts) == 2 else target_raw

            target_resolved = TargetRef(
                status='interunit',
                raw=target_raw,
                unit_id=unit_id,
                callable_id=callable_id,
                callable_name=callable_name,
                name=target_raw
            )
        else:
            # Interunit but not in inventory - unresolved
            target_resolved = TargetRef(
                status='unresolved',
                raw=target_raw
            )

    elif integration_type == 'stdlib':
        target_resolved = TargetRef(
            status='stdlib',
            raw=target_raw
        )

    elif integration_type == 'extlib':
        target_resolved = TargetRef(
            status='extlib',
            raw=target_raw
        )

    elif integration_type == 'boundary':
        # Boundaries are typically also extlib or stdlib, but boundary takes precedence
        target_resolved = TargetRef(
            status='boundary',
            raw=target_raw
        )

    else:
        # Unknown integration type - mark as unresolved
        target_resolved = TargetRef(
            status='unresolved',
            raw=target_raw
        )

    # Extract boundary information if this is a boundary integration
    boundary = None
    boundary_data = fact.get('boundary')
    if boundary_data and isinstance(boundary_data, dict):
        boundary = BoundarySummary(
            kind=boundary_data.get('kind', 'other'),
            protocol=boundary_data.get('protocol'),
            system=boundary_data.get('system'),
            endpoint=boundary_data.get('endpoint'),
            operation=boundary_data.get('operation'),
            resource=boundary_data.get('resource')
        )

    return IntegrationPoint(
        id=integration_id,
        integration_type=integration_type,
        source_unit=source_unit,
        source_callable_id=source_callable_id,
        source_callable_name=source_callable_name,
        target_raw=target_raw,
        target_resolved=target_resolved,
        kind=kind,
        execution_paths=execution_paths,
        condition=condition,
        boundary=boundary,
        signature=signature,
        notes=notes
    )


def collect_integration_points(
        ledger_paths: list[Path],
        callable_inventory_path: Path | None = None,
        verbose: bool = False
) -> list[IntegrationPoint]:
    """
    Collect all integration points from the provided ledgers.

    Args:
        ledger_paths: Paths to unit ledger YAML files
        callable_inventory_path: Path to callable-inventory.txt (optional)
        verbose: Print progress information

    Returns:
        List of IntegrationPoint objects
    """
    points = []

    # Load callable inventory for target resolution
    if verbose:
        print("Loading callable inventory...")

    callable_inventory = load_callable_inventory(callable_inventory_path)

    if verbose:
        if callable_inventory:
            print(f"  Loaded {len(callable_inventory)} callable entries")
        else:
            print("  No callable inventory found - interunit targets will be unresolved")

    # Load all ledgers
    ledgers = load_ledgers(ledger_paths)

    if verbose:
        print(f"Loaded {len(ledgers)} ledger(s)")

    # Extract integration facts from each ledger
    for ledger_data in ledgers:
        path = ledger_data['path']
        documents = ledger_data['documents']

        # Find the ledger document
        ledger_doc = find_ledger_doc(documents)
        if not ledger_doc:
            if verbose:
                print(f"  WARNING: No ledger document in {path.name}")
            continue

        unit_name = ledger_doc.get('unit', {}).get('name', 'unknown')

        # Extract integration facts
        facts = extract_integration_facts(ledger_doc)

        if verbose:
            print(f"  {path.name}: {len(facts)} integration point(s)")

        # Convert each fact to an IntegrationPoint object
        for fact in facts:
            try:
                point = create_integration_point(fact, callable_inventory)
                points.append(point)
            except Exception as e:
                print(f"  WARNING: Failed to create integration point from {fact.get('id')}: {e}")

    return points


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        'ledgers',
        nargs='*',
        type=Path,
        help='Unit ledger YAML files to process (default: auto-discover from --ledgers-root)'
    )
    ap.add_argument(
        '--ledgers-root',
        type=Path,
        default=config.get_ledgers_root(),
        help=f'Root directory for ledger discovery (default: {config.get_ledgers_root()})'
    )
    ap.add_argument(
        '--target-root',
        type=Path,
        help='Target project root (default: current directory)'
    )
    ap.add_argument(
        '--structure',
        choices=['auto', 'flat', 'package'],
        default=config.get_ledger_structure(),
        help=f'Ledger directory structure (default: {config.get_ledger_structure()})'
    )
    ap.add_argument(
        '--anchor',
        default=config.get_namespace_anchor(),
        help='Namespace anchor to strip for package resolution'
    )
    ap.add_argument(
        '--callable-inventory',
        type=Path,
        help='Path to callable-inventory.txt (default: {target_root}/dist/inspect/callable-inventory.txt)'
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=config.get_stage_output(1),
        help=f'Output file (default: {config.get_stage_output(1)})'
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

    # Determine ledger paths
    if args.ledgers:
        # Explicit ledgers provided
        ledger_paths = args.ledgers
        if args.verbose:
            print(f"Using {len(ledger_paths)} explicitly provided ledger(s)")
    else:
        # Auto-discover ledgers
        if args.verbose:
            print(f"Auto-discovering ledgers in: {args.ledgers_root}")

        ledger_paths = discover_ledgers(
            root=args.ledgers_root,
            structure=args.structure,
            anchor=args.anchor
        )

        if not ledger_paths:
            print(f"ERROR: No ledgers found in {args.ledgers_root}", file=sys.stderr)
            return 1

        if args.verbose:
            print(f"Discovered {len(ledger_paths)} ledger(s)")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Collect integration points
    if args.verbose:
        print("\nCollecting integration points...")

    points = collect_integration_points(
        ledger_paths,
        callable_inventory_path=args.callable_inventory,
        verbose=args.verbose
    )

    if args.verbose:
        print(f"\nCollected {len(points)} total integration points")

        # Show breakdown by type
        interunit_count = sum(1 for p in points if p.integration_type == 'interunit')
        extlib_count = sum(1 for p in points if p.integration_type == 'extlib')
        stdlib_count = sum(1 for p in points if p.integration_type == 'stdlib')
        boundary_count = sum(1 for p in points if p.boundary is not None)
        unknown_count = sum(1 for p in points if p.integration_type == 'unknown')
        print(f"  Interunit: {interunit_count}")
        print(f"  Extlib: {extlib_count}")
        print(f"  Stdlib: {stdlib_count}")
        print(f"  Boundaries: {boundary_count}")
        print(f"  Unknown: {unknown_count}")

        # Show resolution statistics
        resolved_count = sum(1 for p in points if p.target_resolved and p.target_resolved.status == 'interunit')
        extlib_status_count = sum(1 for p in points if p.target_resolved and p.target_resolved.status == 'extlib')
        stdlib_status_count = sum(1 for p in points if p.target_resolved and p.target_resolved.status == 'stdlib')
        boundary_status_count = sum(1 for p in points if p.target_resolved and p.target_resolved.status == 'boundary')
        unresolved_count = sum(1 for p in points if p.target_resolved and p.target_resolved.status == 'unresolved')

        print(f"\nResolution Status:")
        print(f"  Resolved (interunit): {resolved_count}")
        print(f"  Extlib: {extlib_status_count}")
        print(f"  Stdlib: {stdlib_status_count}")
        print(f"  Boundary: {boundary_status_count}")
        print(f"  Unresolved: {unresolved_count}")

    # Build output collection
    collection = IntegrationPointCollection(
        points=points,
        ledger_count=len(ledger_paths) if config.include_metadata() else None,
        ledgers_root=str(args.ledgers_root) if config.include_metadata() and not args.ledgers else None,
        explicit_ledgers=[str(p) for p in args.ledgers] if config.include_metadata() and args.ledgers else None
    )

    # Write output
    args.output.write_text(yaml_dump(collection.to_dict()), encoding='utf-8')
    print(f"\nâœ“ Collected {len(points)} integration points â†’ {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
