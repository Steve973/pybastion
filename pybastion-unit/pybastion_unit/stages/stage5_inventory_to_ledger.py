#!/usr/bin/env python3
"""
Transform inventory files into unit ledger YAML format.

Takes inventory with CallableEntry objects and branches, generates
the three-document ledger format required by the specification.
"""

from __future__ import annotations

import argparse
import ast
import sys
import yaml
from pathlib import Path
from typing import Any

from pybastion_unit.shared.models import CallableEntry, IntegrationCategory


# =============================================================================
# Type Tracking
# =============================================================================

def extract_known_types(filepath: Path) -> dict[str, str]:
    """
    Extract type information from source file.

    Returns dict mapping variable names to their types.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    known_types: dict[str, str] = {}

    class TypeTracker(ast.NodeVisitor):
        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            """Capture annotated assignments: x: int = 5"""
            if isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Name):
                known_types[node.target.id] = node.annotation.id
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Capture parameter type hints"""
            # Regular parameters
            for param in node.args.args:
                if param.annotation and isinstance(param.annotation, ast.Name):
                    known_types[param.arg] = param.annotation.id
            # Keyword-only parameters
            for param in node.args.kwonlyargs:
                if param.annotation and isinstance(param.annotation, ast.Name):
                    known_types[param.arg] = param.annotation.id
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Capture parameter type hints in async functions"""
            # Regular parameters
            for param in node.args.args:
                if param.annotation and isinstance(param.annotation, ast.Name):
                    known_types[param.arg] = param.annotation.id
            # Keyword-only parameters
            for param in node.args.kwonlyargs:
                if param.annotation and isinstance(param.annotation, ast.Name):
                    known_types[param.arg] = param.annotation.id
            self.generic_visit(node)

    tracker = TypeTracker()
    tracker.visit(tree)

    return known_types


# =============================================================================
# Document 1: Derived IDs
# =============================================================================

def generate_derived_ids_doc(
        unit_name: str,
        language: str,
        unit_id: str,
        entries: list[CallableEntry]
) -> dict[str, Any]:
    """Generate Document 1: Derived IDs."""

    def collect_entries(entries_list: list[CallableEntry], collected: list[dict[str, Any]]) -> None:
        """Recursively collect entry assignments."""
        for entry in entries_list:
            collected.append({
                'id': entry.id,
                'kind': entry.kind,
                'name': entry.name,
                'address': f'{unit_name}::{entry.name}@L{entry.line_start}'
            })
            if entry.children:
                collect_entries(entry.children, collected)

    def collect_branches(entries_list: list[CallableEntry], collected: list[dict[str, Any]]) -> None:
        """Recursively collect branch (EI) assignments."""
        for entry in entries_list:
            if entry.branches:
                for branch in entry.branches:
                    collected.append({
                        'id': branch.id,
                        'address': f'{unit_name}::{entry.name}@L{branch.line}',
                        'summary': f'{branch.condition} → {branch.outcome}'
                    })
            if entry.children:
                collect_branches(entry.children, collected)

    entry_assignments: list[dict[str, Any]] = []
    branch_assignments: list[dict[str, Any]] = []

    collect_entries(entries, entry_assignments)
    collect_branches(entries, branch_assignments)

    return {
        'docKind': 'derived-ids',
        'schemaVersion': '1.0.0',
        'unit': {
            'name': unit_name,
            'language': language,
            'unitId': unit_id
        },
        'assigned': {
            'entries': entry_assignments,
            'branches': branch_assignments
        }
    }


# =============================================================================
# Document 2: Ledger
# =============================================================================

def transform_entry_to_ledger(
        entry: CallableEntry,
        project_types: set[str],
        known_types: dict[str, str]
) -> dict[str, Any]:
    """Transform a CallableEntry to ledger Entry format."""

    ledger_entry: dict[str, Any] = {
        'id': entry.id,
        'kind': entry.kind,
        'name': entry.name
    }

    # Add optional fields
    if entry.visibility:
        ledger_entry['visibility'] = entry.visibility

    if entry.signature:
        ledger_entry['signature'] = entry.signature

    if entry.decorators:
        ledger_entry['decorators'] = entry.decorators

    if entry.modifiers:
        ledger_entry['modifiers'] = entry.modifiers

    if entry.base_classes:
        ledger_entry['base_classes'] = entry.base_classes

    # Handle callable-specific fields
    if entry.needs_callable_analysis:
        callable_spec = entry.to_ledger_callable_spec(project_types, known_types)
        ledger_entry['callable'] = callable_spec

    # Recursively transform children
    if entry.children:
        ledger_entry['children'] = [
            transform_entry_to_ledger(child, project_types, known_types)
            for child in entry.children
        ]

    return ledger_entry


def generate_ledger_doc(
        unit_id: str,
        unit_name: str,
        entries: list[CallableEntry],
        project_types: set[str],
        known_types: dict[str, str]
) -> dict[str, Any]:
    """Generate Document 2: Ledger."""

    # Transform all entries
    ledger_entries = [
        transform_entry_to_ledger(entry, project_types, known_types)
        for entry in entries
    ]

    # Create unit entry
    unit_entry: dict[str, Any] = {
        'id': unit_id,
        'kind': 'unit',
        'name': unit_name,
        'children': ledger_entries
    }

    return {
        'docKind': 'ledger',
        'schemaVersion': '1.0.0',
        'unit': unit_entry
    }


# =============================================================================
# Document 3: Review
# =============================================================================

def generate_review_doc(
        unit_name: str,
        language: str,
        entries: list[CallableEntry],
        project_types: set[str],
        known_types: dict[str, str],
        quality_metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate Document 3: Ledger Generation Review."""

    # Count stats
    callables_analyzed = 0
    total_eis = 0
    integration_counts: dict[IntegrationCategory, int] = {
        IntegrationCategory.INTERUNIT: 0,
        IntegrationCategory.EXTLIB: 0,
        IntegrationCategory.STDLIB: 0,
        IntegrationCategory.BOUNDARY: 0,
        IntegrationCategory.UNKNOWN: 0
    }
    unknown_integrations: list[str] = []

    def count_recursive(entries_list: list[CallableEntry]) -> None:
        """Recursively count stats."""
        nonlocal callables_analyzed, total_eis

        for entry in entries_list:
            if entry.needs_callable_analysis:
                callables_analyzed += 1
                total_eis += len(entry.branches)

                # Categorize integrations
                categorized = entry.categorize_integrations(project_types, known_types)
                for category_str, facts in categorized.items():
                    category = IntegrationCategory(category_str)
                    integration_counts[category] += len(facts)

                    # Track unknown integrations for findings
                    if category == IntegrationCategory.UNKNOWN:
                        # Skip if this entry has MechanicalOperation/UtilityOperation decorator
                        has_mechop_or_utilop: bool = False
                        if entry.decorators:
                            for decorator in entry.decorators or []:
                                if decorator.get('name') in ['MechanicalOperation', 'UtilityOperation']:
                                    has_mechop_or_utilop = True
                                    break

                        if not has_mechop_or_utilop:
                            for fact in facts:
                                # Integration ID might not exist if no execution paths
                                if 'id' in fact:
                                    unknown_integrations.append(fact['id'])
                                else:
                                    unknown_integrations.append(f"{fact['target']} (no execution path)")

            if entry.children:
                count_recursive(entry.children)

    count_recursive(entries)

    # Build findings
    findings: list[dict[str, Any]] = []

    if unknown_integrations:
        findings.append({
            'severity': 'warn',
            'category': 'ambiguity',
            'message': f'Could not categorize {len(unknown_integrations)} integration point(s) - marked as unknown',
            'appliesTo': unknown_integrations,
            'recommendedAction': 'Review unknown integrations and update project type inventory or categorization rules'
        })

    review_doc = {
        'docKind': 'ledger-generation-review',
        'schemaVersion': '1.0.0',
        'unit': {
            'name': unit_name,
            'language': language,
            'callablesAnalyzed': str(callables_analyzed),
            'totalExeItems': str(total_eis),
            'interunitIntegrations': str(integration_counts[IntegrationCategory.INTERUNIT]),
            'extlibIntegrations': str(integration_counts[IntegrationCategory.EXTLIB]),
            'stdlibIntegrations': str(integration_counts[IntegrationCategory.STDLIB]),
            'boundaryIntegrations': str(integration_counts[IntegrationCategory.BOUNDARY])
        },
        'findings': findings
    }

    # Merge quality metrics if provided
    if quality_metrics:
        review_doc['qualityMetrics'] = quality_metrics.get('metrics', {})
    review_doc['overallQualityGrade'] = quality_metrics.get('overallGrade', 'unknown')

    # Add flagged callables if present
    if quality_metrics.get('flaggedCallables'):
        review_doc['qualityFlags'] = quality_metrics['flaggedCallables']

    return review_doc


# =============================================================================
# Main Transformation
# =============================================================================

def load_project_types(project_inventory_path: Path | None) -> set[str]:
    """
    Load project type inventory from file.

    Expected format: one fully-qualified name per line or FQN:ID pairs.
    """
    if not project_inventory_path or not project_inventory_path.exists():
        return set()

    project_types: set[str] = set()
    with open(project_inventory_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle FQN:ID format (split on colon)
                if ':' in line:
                    fqn = line.split(':', 1)[0]
                    project_types.add(fqn)
                else:
                    project_types.add(line)

    return project_types


def transform_inventory_to_ledger(
        inventory_path: Path,
        project_inventory_path: Path | None,
        quality_file_path: Path | None,
        output_path: Path
) -> None:
    """Transform inventory YAML to three-document ledger YAML."""

    # Load inventory
    print(f"Loading inventory: {inventory_path}")
    with open(inventory_path, 'r', encoding='utf-8') as f:
        inventory = yaml.safe_load(f)

    # Extract metadata
    unit_name = inventory['unit']
    language = inventory['language']
    unit_id = inventory['unit_id']
    filepath = Path(inventory['filepath'])

    # Extract type information from source file
    print(f"  → Extracting type information from source")
    known_types = extract_known_types(filepath)
    print(f"  → Found {len(known_types)} typed variables")

    # Parse entries to CallableEntry objects
    print(f"  → Parsing {len(inventory['entries'])} entries")
    entries = [CallableEntry.from_dict(e) for e in inventory['entries']]

    # Load project types
    project_types = load_project_types(project_inventory_path)
    print(f"  → Loaded {len(project_types)} project types")

    # Load quality metrics if provided
    quality_metrics = None
    if quality_file_path and quality_file_path.exists():
        print(f"  → Loading quality metrics from {quality_file_path}")
        with open(quality_file_path, 'r', encoding='utf-8') as f:
            quality_metrics = yaml.safe_load(f)
        print(f"  → Quality grade: {quality_metrics.get('overallGrade', 'unknown')}")

    # Generate three documents
    print("  → Generating Document 1 (Derived IDs)")
    doc1 = generate_derived_ids_doc(unit_name, language, unit_id, entries)

    print("  → Generating Document 2 (Ledger)")
    doc2 = generate_ledger_doc(unit_id, unit_name, entries, project_types, known_types)

    print("  → Generating Document 3 (Review)")
    doc3 = generate_review_doc(unit_name, language, entries, project_types, known_types, quality_metrics)

    # Write three-document YAML
    print(f"  → Writing ledger: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Document 1
        yaml.dump(doc1, f, sort_keys=False, allow_unicode=True, width=float('inf'))
        f.write('\n---\n')

        # Document 2
        yaml.dump(doc2, f, sort_keys=False, allow_unicode=True, width=float('inf'))
        f.write('\n---\n')

        # Document 3
        yaml.dump(doc3, f, sort_keys=False, allow_unicode=True, width=float('inf'))

    # Print summary
    summary = inventory.get('summary', {})
    print(f"\n✓ Ledger generated successfully")
    print(f"  Unit: {unit_name} ({language})")
    print(f"  Callables: {doc3['unit']['callablesAnalyzed']}")
    print(f"  Total EIs: {doc3['unit']['totalExeItems']}")
    print(f"  Integrations: {doc3['unit']['interunitIntegrations']} interunit, "
          f"{doc3['unit']['extlibIntegrations']} extlib, "
          f"{doc3['unit']['stdlibIntegrations']} stdlib, "
          f"{doc3['unit']['boundaryIntegrations']} boundary")
    if doc3['findings']:
        print(f"  Findings: {len(doc3['findings'])}")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Transform inventory YAML to unit ledger format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--inventory', type=Path, required=True,
                        help='Path to inventory YAML file')
    parser.add_argument('--project-inventory', type=Path,
                        help='Path to project type inventory file')
    parser.add_argument('--quality-file', type=Path, default=None,
                        help='Path to quality file for additional analysis')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path for output ledger YAML')

    args = parser.parse_args()

    if not args.inventory.exists():
        print(f"Error: Inventory file not found: {args.inventory}")
        return 1

    try:
        transform_inventory_to_ledger(
            args.inventory,
            args.project_inventory,
            args.quality_file,
            args.output
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())