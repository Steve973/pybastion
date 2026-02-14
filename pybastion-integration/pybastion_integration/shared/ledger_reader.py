"""
Utilities for reading and parsing unit ledgers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def discover_ledgers(
    root: Path,
    structure: str = 'auto',
    anchor: str | None = None
) -> list[Path]:
    """
    Discover all ledger files in a directory tree.

    Args:
        root: Root directory to search
        structure: Directory structure ('auto' | 'flat' | 'package') - currently ignored
        anchor: Namespace anchor - currently ignored

    Returns:
        List of paths to ledger files, sorted by path

    Note:
        This is a simplified discovery that just finds *-ledger.yaml files.
        The 'structure' and 'anchor' parameters are accepted for compatibility
        but not currently used.
    """
    if not root.exists():
        return []

    if root.is_file():
        # Single file provided
        return [root] if _is_ledger_file(root) else []

    # Directory - recursively find all ledger files
    ledgers = []
    for path in root.rglob('*'):
        if path.is_file() and _is_ledger_file(path):
            ledgers.append(path.resolve())

    return sorted(ledgers, key=lambda p: str(p).lower())


def _is_ledger_file(path: Path) -> bool:
    """Check if a file is a ledger file based on naming convention."""
    name_lower = path.name.lower()
    return name_lower.endswith('.ledger.yaml') or name_lower.endswith('.ledger.yml')


def load_ledgers(ledger_paths: list[Path]) -> list[dict[str, Any]]:
    """
    Load multiple unit ledger files.

    Args:
        ledger_paths: Paths to unit ledger YAML files

    Returns:
        List of parsed ledger data, each containing:
        - 'path': Path to the ledger file
        - 'documents': List of YAML documents from the file
    """
    ledgers = []
    for path in ledger_paths:
        try:
            with path.open('r', encoding='utf-8') as f:
                docs = list(yaml.safe_load_all(f))
            ledgers.append({
                'path': path,
                'documents': [d for d in docs if d is not None]
            })
        except Exception as e:
            # Log error but continue with other ledgers
            print(f"WARNING: Failed to load {path}: {e}")

    return ledgers


def find_ledger_doc(documents: list[Any]) -> dict[str, Any] | None:
    """
    Find the ledger document (Document 2) in a multi-doc YAML.

    Args:
        documents: List of parsed YAML documents

    Returns:
        The ledger document (docKind: "ledger"), or None if not found
    """
    for doc in documents:
        if isinstance(doc, dict) and doc.get('docKind') == 'ledger':
            return doc
    return None


def find_derived_ids_doc(documents: list[Any]) -> dict[str, Any] | None:
    """
    Find the derived IDs document (Document 1) in a multi-doc YAML.

    Args:
        documents: List of parsed YAML documents

    Returns:
        The derived IDs document (docKind: "derived-ids"), or None if not found
    """
    for doc in documents:
        if isinstance(doc, dict) and doc.get('docKind') == 'derived-ids':
            return doc
    return None


def extract_integration_facts(ledger_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract all integration facts from a ledger document.

    Walks the unit tree to find all callables and extracts their
    integration.interunit and integration.boundaries facts.

    Args:
        ledger_doc: Parsed ledger document (Document 2, docKind: "ledger")

    Returns:
        List of integration fact dictionaries, each containing:
        - All fields from the integration fact
        - 'source_unit': Unit name
        - 'source_callable_id': Callable ID
        - 'source_callable_name': Callable name
        - 'integrationKind': 'interunit' or 'boundary'
    """
    facts = []

    # Get the unit entry (root of the tree)
    unit = ledger_doc.get('unit')
    if not unit or not isinstance(unit, dict):
        return facts

    unit_name = unit.get('name', 'unknown')

    # Walk all entries depth-first
    entries_to_process = [unit]

    while entries_to_process:
        entry = entries_to_process.pop(0)

        # Add children to process queue
        children = entry.get('children', [])
        if isinstance(children, list):
            entries_to_process.extend(c for c in children if isinstance(c, dict))

        # Only process callable entries
        if entry.get('kind') not in ('function', 'method', 'callable'):
            continue

        callable_spec = entry.get('callable')
        if not callable_spec or not isinstance(callable_spec, dict):
            continue

        callable_id = entry.get('id', 'unknown')
        callable_name = entry.get('name', 'unknown')

        integration = callable_spec.get('integration')
        if not integration or not isinstance(integration, dict):
            continue

        # Extract all integration fact categories
        categories = {
            'interunit': 'interunit',
            'stdlib': 'stdlib',
            'extlib': 'extlib',
            'unknown': 'unknown',
            'boundaries': 'boundary'
        }

        for category_key, integration_kind in categories.items():
            facts_list = integration.get(category_key, [])
            if isinstance(facts_list, list):
                for fact in facts_list:
                    if isinstance(fact, dict):
                        facts.append({
                            **fact,
                            'source_unit': unit_name,
                            'source_callable_id': callable_id,
                            'source_callable_name': callable_name,
                            'integration_type': integration_kind
                        })

    return facts