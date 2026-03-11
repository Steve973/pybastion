#!/usr/bin/env python3
"""
Build Execution Instance Level Control Flow Graph

Constructs a complete control flow graph from ledger files where:
  - Nodes are Execution Instances (EIs) - individual statement outcomes
  - Edges represent control flow:
    * Sequential: flow to next statement in same callable
    * Call: transfer to called function's entry EI
    * Return: transfer back from callee's exit EIs to caller's continuation
  - Handles local calls, interunit calls, and external integrations

Algorithm:
  1. Process each callable, adding all EI nodes
  2. Check pending returns queue - if this callable was called, wire up actual return edges
  3. For each EI, create appropriate edges:
     - If terminal: no edges
     - If call: create call edge + stub return edge, queue for later refinement
     - If sequential: edge to next EI
  4. At end, any remaining stub nodes indicate missing/external callables

Input:  *.ledger.yaml files, callable-inventory.txt
Output: NetworkX DiGraph (serialized as pickle)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from pybastion_common.models import NodeCategory, CallNodeType, ExternalNodeType
from pybastion_integration import config
from pybastion_integration.config import StoreInConfig


# =============================================================================
# EI Registry Building
# =============================================================================

def discover_inventory_files(inventories_root: Path) -> list[Path]:
    """Discover all inventory YAML files."""
    # Look for both .inventory.yaml and _inventory.yaml patterns
    inventory_files = list(inventories_root.rglob('*.inventory.yaml'))
    inventory_files.extend(inventories_root.rglob('*_inventory.yaml'))
    return sorted(set(inventory_files))


def build_ei_registry_from_inventories(inventory_paths: list[Path]) -> dict[str, list[str]]:
    """
    Build complete EI registry from all inventory files.

    Args:
        inventory_paths: List of paths to inventory YAML files

    Returns:
        Dict mapping callable_id -> [list of EI IDs in order]
    """
    ei_registry = {}

    for inv_path in inventory_paths:
        with open(inv_path, 'r', encoding='utf-8') as f:
            inventory = yaml.safe_load(f)

        if not inventory or 'entries' not in inventory:
            continue

        # Process all entries recursively
        def process_entries(entries: list[dict[str, Any]]) -> None:
            for entry in entries:
                callable_id = entry.get('id')

                # Extract EI IDs from branches (directly under entry, not under 'callable')
                if 'branches' in entry and entry['branches']:
                    ei_ids = [b['id'] for b in entry['branches']]
                    ei_registry[callable_id] = ei_ids

                # Recurse into children
                if 'children' in entry:
                    process_entries(entry['children'])

        process_entries(inventory.get('entries', []))

    return ei_registry


# =============================================================================
# Utility Functions
# =============================================================================

def load_callable_inventory(filepath: Path | None) -> dict[str, str]:
    """
    Load callable inventory file (FQN:ID pairs).

    Returns:
        Dict mapping fully qualified names to callable IDs
    """
    inventory = {}
    if not filepath or not filepath.exists():
        return inventory

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            fqn, callable_id = line.split(':', 1)
            inventory[fqn] = callable_id

    return inventory


def discover_ledgers(ledgers_root: Path) -> list[Path]:
    """Discover all ledger YAML files under ledgers_root."""
    # Try both naming patterns
    ledgers = list(ledgers_root.rglob('*.ledger.yaml'))
    ledgers.extend(ledgers_root.rglob('*_ledger.yaml'))
    return sorted(set(ledgers))  # Deduplicate


# =============================================================================
# EI Node Creation
# =============================================================================

def add_ei_node(
        cfg: nx.DiGraph,
        ei_id: str,
        branch_data: dict[str, Any],
        callable_id: str,
        unit_name: str,
        callable_name: str,
        decorators: list[dict[str, Any]]
) -> None:
    """
    Add an EI node to the CFG with all relevant attributes.

    Args:
        cfg: NetworkX graph
        ei_id: Execution instance ID
        branch_data: Branch/EI data from ledger
        callable_id: Parent callable ID
        unit_name: Unit this EI belongs to
        callable_name: Callable this EI belongs to
        decorators: Combined callable + statement level decorators
    """
    condition = branch_data.get('condition', '')
    outcome = branch_data.get('outcome', '')
    is_terminal = branch_data.get('is_terminal', False)
    terminates_via = branch_data.get('terminates_via')
    constraint = branch_data.get('constraint')

    cfg.add_node(
        ei_id,
        category=NodeCategory.CALL_NODE.value,
        type=None,  # Will be set when we determine call type
        callable_id=callable_id,
        unit=unit_name,
        callable_name=callable_name,
        condition=condition,
        outcome=outcome,
        is_terminal=is_terminal,
        terminates_via=terminates_via,
        constraint=constraint,
        decorators=decorators
    )


# =============================================================================
# Exit EI Identification
# =============================================================================
def find_exception_exit_eis(branches: list[dict[str, Any]]) -> list[str]:
    """Find EIs that exit via exceptions/raises."""
    exit_eis = []
    for branch in branches:
        is_terminal = branch.get('is_terminal', False)
        terminates_via = branch.get('terminates_via', '')

        if is_terminal and terminates_via in ('exception', 'raise'):
            exit_eis.append(branch['id'])

    return exit_eis


def find_successful_exit_eis(branches: list[dict[str, Any]]) -> list[str]:
    """
    Find EIs that represent successful exits from a callable.

    Exit EIs are:
    - Terminal with terminates_via='return' (explicit/implicit returns)
    - Have 'yield' in outcome (context managers - may not be marked terminal)
    - Non-terminal EIs with no subsequent non-exception EIs (implicit return at end)

    Args:
        branches: List of branch/EI dicts for a callable

    Returns:
        List of EI IDs that are successful exit points
    """
    exit_eis = []

    for i, branch in enumerate(branches):
        is_terminal = branch.get('is_terminal', False)
        terminates_via = branch.get('terminates_via', '')
        outcome = branch.get('outcome', '').lower()

        # Terminal returns are successful exits
        if is_terminal and terminates_via in ('return', 'implicit-return', 'yield'):
            exit_eis.append(branch['id'])
            continue

        # Yields (context managers) - may not be marked terminal
        if 'yield' in outcome:
            exit_eis.append(branch['id'])
            continue

        # Skip terminal exceptions
        if is_terminal:
            continue

        # Check if this non-terminal EI is followed only by exceptions
        # (meaning it's the last successful path - implicit return)
        has_non_exception_successor = False
        for j in range(i + 1, len(branches)):
            successor = branches[j]
            succ_is_terminal = successor.get('is_terminal', False)
            succ_terminates_via = successor.get('terminates_via', '')

            # If there's a non-exception successor, this isn't an exit
            if not succ_is_terminal or succ_terminates_via not in ('exception', 'raise'):
                has_non_exception_successor = True
                break

        if not has_non_exception_successor:
            exit_eis.append(branch['id'])

    return exit_eis


# =============================================================================
# Unresolved Call Categorization
# =============================================================================

def categorize_unresolved_call(operation_target: str) -> ExternalNodeType:
    """
    Categorize an unresolved call (has operation_target but no integration entry).

    Args:
        operation_target: The target being called (e.g., 'sorted', 'c.get', 'ValueError')

    Returns:
        ExternalNodeType indicating what kind of external call this is
    """
    # Known built-ins and standard library constructors
    builtins = {
        'sorted', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
        'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'sum', 'min', 'max', 'abs', 'all', 'any', 'round',
        'ValueError', 'TypeError', 'KeyError', 'RuntimeError', 'AttributeError',
        'IndexError', 'OSError', 'IOError', 'Exception', 'StopIteration',
        'print', 'open', 'input', 'iter', 'next', 'hasattr', 'getattr', 'setattr',
        'isinstance', 'issubclass', 'callable', 'type', 'id', 'hash', 'hex', 'bin', 'oct'
    }

    if operation_target in builtins:
        return ExternalNodeType.STDLIB

    # String literal methods (starts with quote)
    if operation_target.startswith(("'", '"')):
        return ExternalNodeType.STDLIB

    # Method calls or attribute access (contains '.')
    # Could be project type or external type - mark as unknown
    if '.' in operation_target:
        return ExternalNodeType.UNKNOWN

    # Default to unknown
    return ExternalNodeType.UNKNOWN


# =============================================================================
# Integration Resolution
# =============================================================================

def find_integration_for_ei(ei_id: str, integration_data: dict[str, list[dict]]) -> tuple[str, dict] | None:
    """
    Find integration point data for an EI.

    Args:
        ei_id: EI ID (e.g., 'U37D3513825_F007_E0005')
        integration_data: Integration section from ledger

    Returns:
        (category, integration_dict) or None if not found
    """
    integration_id = 'I' + ei_id

    for category in ['interunit', 'stdlib', 'extlib', 'boundary', 'unknown']:
        for int_point in integration_data.get(category, []):
            if int_point.get('id') == integration_id:
                return category, int_point

    return None


# =============================================================================
# Local Callable Resolution
# =============================================================================

def find_local_callable_id(
        target_name: str,
        current_unit_callables: dict[str, str],
        unit_name: str
) -> str | None:
    """
    Find callable ID for a local function call within the same unit.

    Args:
        target_name: Function name being called (e.g., '_normalize_strategy_configs')
        current_unit_callables: Map of callable_name -> callable_id for this unit
        unit_name: Current unit name

    Returns:
        Callable ID if found locally, None otherwise
    """
    # Try direct lookup by name
    if target_name in current_unit_callables:
        return current_unit_callables[target_name]

    return None


def handle_external_call(
        cfg: nx.DiGraph,
        ei_id: str,
        operation_target: str,
        external_type: ExternalNodeType,
        next_ei_id: str | None,
        integration_data: dict[str, Any] | None = None
) -> None:
    """
    Handle external call - create external node with call and return edges.

    External nodes are created and completed immediately (no stub/queue needed).

    Args:
        cfg: NetworkX graph
        ei_id: Call site EI ID
        operation_target: What's being called
        external_type: Type of external call
        next_ei_id: Where to return to after call
        integration_data: Optional integration metadata
    """
    # Create unique external node ID
    external_node_id = f'EXTERNAL_{external_type.value.upper()}_{ei_id}'

    # Create external node
    cfg.add_node(
        external_node_id,
        category=NodeCategory.EXTERNAL_NODE.value,
        type=external_type.value,
        operation_target=operation_target,
        called_from=ei_id,
        returns_to=next_ei_id,
        signature=integration_data.get('signature') if integration_data else None,
        execution_paths=integration_data.get('execution_paths', []) if integration_data else []
    )

    # Create call edge
    cfg.add_edge(
        ei_id,
        external_node_id,
        edge_type='call',
        integration_type=external_type.value,
        operation_target=operation_target
    )
    print(f"DEBUG: In handle_external_call: added call edge {ei_id} -> {external_node_id} ")

    # Create return edge directly (if there's a next EI)
    if next_ei_id:
        cfg.add_edge(
            external_node_id,
            next_ei_id,
            edge_type='return',
            returns_from_external=True
        )
        print(f"DEBUG: In handle_external_call: added return edge {external_node_id} -> {next_ei_id} ")


# =============================================================================
# Edge Creation for Calls
# =============================================================================

def handle_call_ei(
        cfg: nx.DiGraph,
        ei: dict[str, Any],
        ei_index: int,
        all_branches: list[dict[str, Any]],
        integration_data: dict[str, list[dict]],
        callable_inventory: dict[str, str],
        ei_registry: dict[str, list[str]],
        stub_queue: dict[str, list[dict[str, Any]]],
        current_unit_callables: dict[str, str],
        unit_name: str
) -> None:
    """
    Handle a call EI - create call edge and queue return if needed.

    Args:
        cfg: NetworkX graph
        ei: Current EI data
        ei_index: Index in branches list
        all_branches: All branches in this callable
        integration_data: Integration section from ledger
        callable_inventory: FQN -> callable_id mapping
        ei_registry: callable_id -> [EI IDs] mapping
        stub_queue: Map of entry_ei_id -> list of pending returns
        current_unit_callables: Map of local callable names to IDs
        unit_name: Current unit name
    """
    ei_id = ei['id']
    constraint = ei.get('constraint', {})
    operation_target = constraint.get('operation_target') if constraint else None

    if not operation_target:
        # Not actually a call despite having constraint
        return

    # Find next sequential EI for return destination
    next_ei_id = None
    next_ei_index = None
    for j in range(ei_index + 1, len(all_branches)):
        next_branch = all_branches[j]
        # Skip terminal exception branches
        if (
                next_branch.get('is_terminal') and
                next_branch.get('terminates_via') == 'exception'
        ):
            continue
        next_ei_id = next_branch['id']
        next_ei_index = j
        break

    # Check if next_ei_id is part of a conditional group and collect all alternatives
    next_ei_ids: list[str] | None = []

    if next_ei_id:
        next_branch = all_branches[next_ei_index]
        next_constraint = next_branch.get('constraint', {})

        if next_constraint and next_constraint.get('polarity') is not None:
            # Collect all alternatives with same expr
            expr = next_constraint.get('expr')

            # Look backward from next_idx to find earlier alternatives
            for j in range(next_ei_index - 1, -1, -1):
                alt = all_branches[j]
                alt_constraint = alt.get('constraint', {})
                if (alt_constraint and
                        alt_constraint.get('expr') == expr and
                        alt_constraint.get('polarity') is not None):
                    next_ei_ids.insert(0, alt['id'])  # Insert at front to maintain order
                else:
                    break

            # Add current
            next_ei_ids.append(next_ei_id)

            # Look forward from next_idx to find later alternatives
            for j in range(next_ei_index + 1, len(all_branches)):
                alt = all_branches[j]
                alt_constraint = alt.get('constraint', {})
                if (alt_constraint and
                        alt_constraint.get('expr') == expr and
                        alt_constraint.get('polarity') is not None):
                    next_ei_ids.append(alt['id'])
                else:
                    break
        else:
            # Not a conditional - just single target
            next_ei_ids = [next_ei_id]
    else:
        next_ei_ids = []

    print(f"DEBUG: For {ei_id}, collected next_ei_ids: {next_ei_ids}")

    # Check if this is an integration
    integration_info = find_integration_for_ei(ei_id, integration_data)

    if integration_info:
        category, int_point = integration_info
        handle_integration_call(
            cfg, ei_id, category, int_point, next_ei_ids,
            callable_inventory, ei_registry, stub_queue
        )
    else:
        # Local call
        handle_local_call(
            cfg, ei_id, operation_target, next_ei_ids,
            current_unit_callables, unit_name,
            ei_registry, stub_queue
        )


def handle_integration_call(
        cfg: nx.DiGraph,
        ei_id: str,
        category: str,
        int_point: dict[str, Any],
        next_ei_ids: list[str] | None,
        callable_inventory: dict[str, str],
        ei_registry: dict[str, list[str]],
        stub_queue: dict[str, list[dict[str, Any]]]
) -> None:
    """Handle integration call (interunit/stdlib/extlib/boundary)."""
    print(f"DEBUG: handle_integration_call called with next_ei_ids: {next_ei_ids}")
    target_fqn = int_point.get('target', '')

    for next_ei_id in next_ei_ids:
        if category == 'interunit':
            # Interunit call - this is a CALL_NODE that may need stub queueing
            target_callable_id = callable_inventory.get(target_fqn)

            if target_callable_id:
                # Look up entry EI from pre-built registry
                if target_callable_id not in ei_registry:
                    # Callable not in inventory - might be external or missing
                    # Treat as external unknown
                    handle_external_call(
                        cfg, ei_id, target_fqn,
                        ExternalNodeType.UNKNOWN, next_ei_id, int_point
                    )
                    continue

                target_entry = ei_registry[target_callable_id][0]

                # Create entry node if it doesn't exist
                if not cfg.has_node(target_entry):
                    cfg.add_node(target_entry)

                # Create call edge
                cfg.add_edge(
                    ei_id,
                    target_entry,
                    edge_type='call',
                    integration_type='interunit',
                    target_fqn=target_fqn,
                    target_callable=target_callable_id,
                    signature=int_point.get('signature', ''),
                    execution_paths=int_point.get('execution_paths', [])
                )

                # Mark this EI as interunit call
                cfg.nodes[ei_id]['type'] = CallNodeType.INTERUNIT.value

                # Wire up return edge
                if next_ei_id:
                    # Check if the target callable has already been processed
                    entry_node_data = cfg.nodes[target_entry]

                    if 'callable_id' in entry_node_data:
                        print(f"DEBUG: Immediate wiring for {target_callable_id}")
                        # Already processed - wire up return immediately

                        # Reconstruct branches from graph nodes
                        branches = []
                        for node_id in cfg.nodes():
                            node_data = cfg.nodes[node_id]
                            if node_data.get('callable_id') == target_callable_id:
                                branches.append({
                                    'id': node_id,
                                    'is_terminal': node_data.get('is_terminal', False),
                                    'terminates_via': node_data.get('terminates_via', ''),
                                    'outcome': node_data.get('outcome', '')
                                })

                        success_exit_eis = find_successful_exit_eis(branches)
                        exception_exit_eis = find_exception_exit_eis(branches)

                        print(f"DEBUG: In handle_integration_call: Found {len(success_exit_eis)} "
                              f"success-path exits for {target_callable_id}: {success_exit_eis}")
                        print(f"DEBUG: In handle_integration_call: Found {len(exception_exit_eis)} "
                              f"exception-path exits for {target_callable_id}: {exception_exit_eis}")

                        # Create return edges from each exit EI
                        if success_exit_eis:
                            for exit_ei in success_exit_eis:
                                cfg.add_edge(
                                    exit_ei,
                                    next_ei_id,
                                    edge_type='return',
                                    success_outcome=True,
                                    returns_from=target_callable_id,
                                    original_call_site=ei_id
                                )
                                print("DEBUG: In handle_integration_call: Created success-case "
                                      f"return edge {exit_ei} -> {next_ei_id}")

                        if exception_exit_eis:
                            for exit_ei in exception_exit_eis:
                                cfg.add_edge(
                                    exit_ei,
                                    next_ei_id,
                                    edge_type='return',
                                    success_outcome=False,
                                    returns_from=target_callable_id,
                                    original_call_site=ei_id
                                )
                                print("DEBUG: In handle_integration_call: Created exception-case "
                                      f"return edge {exit_ei} -> {next_ei_id}")
                    else:
                        # Not processed yet - queue for later
                        if target_entry not in stub_queue:
                            stub_queue[target_entry] = []
                        stub_queue[target_entry].append({
                            'return_to': next_ei_id,
                            'call_site': ei_id
                        })

        else:
            # External call (stdlib/extlib/boundary) - create EXTERNAL_NODE
            external_type = ExternalNodeType.from_integration_category(category)

            handle_external_call(
                cfg, ei_id, int_point.get('target', ''),
                external_type, next_ei_id, int_point
            )


def handle_local_call(
        cfg: nx.DiGraph,
        ei_id: str,
        operation_target: str,
        next_ei_ids: list[str] | None,
        current_unit_callables: dict[str, str],
        unit_name: str,
        ei_registry: dict[str, list[str]],
        stub_queue: dict[str, list[dict[str, Any]]]
) -> None:
    """Handle local function call within same unit."""
    print(f"DEBUG: handle_local_call called with next_ei_ids: {next_ei_ids}")
    target_callable_id = find_local_callable_id(
        operation_target,
        current_unit_callables,
        unit_name
    )

    for next_ei_id in next_ei_ids:
        if not target_callable_id:
            # Can't find target - it's an unresolved call (method, builtin, etc.)
            # Create external node for it
            external_type = categorize_unresolved_call(operation_target)

            handle_external_call(
                cfg, ei_id, operation_target,
                external_type, next_ei_id
            )
            continue

        # Found local callable - this is a CALL_NODE
        # Look up entry EI from pre-built registry
        if target_callable_id not in ei_registry:
            # Not in inventory - shouldn't happen for local but treat as external unknown
            handle_external_call(
                cfg, ei_id, operation_target,
                ExternalNodeType.UNKNOWN, next_ei_id
            )
            continue

        target_entry = ei_registry[target_callable_id][0]

        # Create entry node if it doesn't exist
        if not cfg.has_node(target_entry):
            cfg.add_node(target_entry)

        # Create call edge
        cfg.add_edge(
            ei_id,
            target_entry,
            edge_type='call',
            integration_type='local',
            target_callable=target_callable_id,
            operation_target=operation_target
        )

        # Mark this EI as local call
        cfg.nodes[ei_id]['type'] = CallNodeType.LOCAL.value

        # Wire up return edge
        if next_ei_id:
            # Check if the target callable has already been processed
            # If the entry node has full attributes (category, callable_id), it's been processed
            entry_node_data = cfg.nodes[target_entry]

            if 'callable_id' in entry_node_data:
                print(f"DEBUG: Immediate wiring for {target_callable_id}")
                # Already processed - wire up return immediately

                # Reconstruct branches from graph nodes
                branches = []
                for node_id in cfg.nodes():
                    node_data = cfg.nodes[node_id]
                    if node_data.get('callable_id') == target_callable_id:
                        branches.append({
                            'id': node_id,
                            'is_terminal': node_data.get('is_terminal', False),
                            'terminates_via': node_data.get('terminates_via', ''),
                            'outcome': node_data.get('outcome', '')
                        })

                # Now use find_successful_exit_eis
                success_exit_eis = find_successful_exit_eis(branches)
                exception_exit_eis = find_exception_exit_eis(branches)

                print(f"DEBUG: In handle_local_call: Found {len(success_exit_eis)} "
                      f"success-path exits for {target_callable_id}: {success_exit_eis}")
                print(f"DEBUG: In handle_local_call: Found {len(exception_exit_eis)} "
                      f"exception-path exits for {target_callable_id}: {exception_exit_eis}")

                # Create return edges from each exit EI
                if success_exit_eis:
                    for exit_ei in success_exit_eis:
                        cfg.add_edge(
                            exit_ei,
                            next_ei_id,
                            edge_type='return',
                            success_outcome=True,
                            returns_from=target_callable_id,
                            original_call_site=ei_id
                        )
                        print("DEBUG: In handle_local_call: Created success-case "
                              f"return edge {exit_ei} -> {next_ei_id}")

                if exception_exit_eis:
                    for exit_ei in exception_exit_eis:
                        cfg.add_edge(
                            exit_ei,
                            next_ei_id,
                            edge_type='return',
                            success_outcome=False,
                            returns_from=target_callable_id,
                            original_call_site=ei_id
                        )
                        print("DEBUG: In handle_local_call: Created exception-case "
                              f"return edge {exit_ei} -> {next_ei_id}")
            else:
                # Not processed yet - queue for later
                if target_entry not in stub_queue:
                    stub_queue[target_entry] = []
                stub_queue[target_entry].append({
                    'return_to': next_ei_id,
                    'call_site': ei_id
                })


# =============================================================================
# Return Edge Refinement
# =============================================================================

def refine_return_edges(
        cfg: nx.DiGraph,
        callable_id: str,
        entry_ei_id: str,
        branches: list[dict[str, Any]],
        stub_queue: dict[str, list[dict[str, Any]]]
) -> None:
    """
    Wire up return edges if this callable's entry was stubbed.

    Args:
        cfg: NetworkX graph
        callable_id: Current callable being processed
        entry_ei_id: Entry EI ID for this callable
        branches: Branches/EIs for this callable
        stub_queue: Map of entry_ei_id -> list of pending returns
    """
    # Check if this entry node was stubbed
    if entry_ei_id not in stub_queue:
        print(f"DEBUG: {callable_id} not in stub_queue")
        return

    print(f"DEBUG: Refining {len(stub_queue[entry_ei_id])} returns for {callable_id}")

    success_exit_eis = find_successful_exit_eis(branches)
    exception_exit_eis = find_exception_exit_eis(branches)

    if not success_exit_eis and not exception_exit_eis:
        print(f"Warning: No exit EIs found for {callable_id}")
        return

    print(
        f"DEBUG: In refine_return_edges: Found {len(success_exit_eis)} success-path exits for {callable_id}: {success_exit_eis}")
    print(
        f"DEBUG: In refine_return_edges: Found {len(exception_exit_eis)} exception-path exits for {callable_id}: {exception_exit_eis}")

    # Wire up return edges for each caller
    for pending in stub_queue[entry_ei_id]:
        return_to = pending['return_to']
        call_site = pending['call_site']

        # Create return edges from each exit EI
        if success_exit_eis:
            for exit_ei in success_exit_eis:
                cfg.add_edge(
                    exit_ei,
                    return_to,
                    edge_type='return',
                    success_outcome=True,
                    returns_from=callable_id,
                    original_call_site=call_site
                )
                print(f"DEBUG: In refine_return_edges: Created success-case return edge {exit_ei} -> {return_to}")

        if exception_exit_eis:
            for exit_ei in exception_exit_eis:
                cfg.add_edge(
                    exit_ei,
                    return_to,
                    edge_type='return',
                    success_outcome=False,
                    returns_from=callable_id,
                    original_call_site=call_site
                )
                print(f"DEBUG: In refine_return_edges: Created exception-case return edge {exit_ei} -> {return_to}")

    # Remove from stub queue - this callable is now complete
    del stub_queue[entry_ei_id]


# =============================================================================
# Callable Processing
# =============================================================================

def process_callable(
        cfg: nx.DiGraph,
        callable_entry: dict[str, Any],
        unit_name: str,
        callable_inventory: dict[str, str],
        ei_registry: dict[str, list[str]],
        stub_queue: dict[str, list[dict[str, Any]]],
        current_unit_callables: dict[str, str]
) -> None:
    """
    Process a single callable - add nodes and create edges.

    Args:
        cfg: NetworkX graph
        callable_entry: Callable data from ledger
        unit_name: Unit name
        callable_inventory: FQN -> callable_id mapping
        ei_registry: callable_id -> [EI IDs] mapping
        stub_queue: Map of entry_ei_id -> list of pending returns
        current_unit_callables: Map of local callable names to IDs
    """
    # Skip stub callables
    if callable_entry.get('is_stub', False):
        return

    callable_id = callable_entry['id']
    callable_name = callable_entry.get('name', '')
    callable_decorators = callable_entry.get('decorators', [])

    # Get branches (EIs) and integration data from callable spec
    callable_spec = callable_entry.get('callable', {})
    branches = callable_spec.get('branches', [])

    # Extract integration data from this callable's spec
    integration_data = callable_spec.get('integration', {})

    if not branches:
        # No EIs for this callable
        return

    # Step 1: Add all EI nodes
    ei_ids = []
    for branch in branches:
        ei_id = branch['id']
        ei_ids.append(ei_id)

        # Combine decorators
        stmt_decorators = branch.get('decorators', [])
        all_decorators = callable_decorators + stmt_decorators

        add_ei_node(
            cfg, ei_id, branch, callable_id,
            unit_name, callable_name, all_decorators
        )

    # NOTE: ei_registry is pre-built from inventories, not populated here

    # Step 2: Refine any pending returns for THIS callable
    entry_ei_id = ei_ids[0]
    refine_return_edges(cfg, callable_id, entry_ei_id, branches, stub_queue)

    # Step 3: Create edges for each EI
    for i, branch in enumerate(branches):
        ei_id = branch['id']

        if branch.get('is_terminal') and branch.get('terminates_via') in ('raise', 'exception'):
            # Terminal exception - no outgoing edges
            continue

        # Check for conditional branching (if/elif evaluation EIs)
        conditional_targets = branch.get('conditional_targets')
        if conditional_targets:
            for ct in conditional_targets:
                if ct.get('is_terminal'):
                    # This branch terminates - no edge needed
                    continue

                target_ei = ct.get('target_ei')
                if target_ei:
                    cfg.add_edge(
                        ei_id,
                        target_ei,
                        edge_type='branch',
                        condition=ct.get('condition')
                    )
            continue  # Done with this EI
        # Check if branch has explicit next_ei
        next_ei_target = branch.get('next_ei')
        if next_ei_target:
            cfg.add_edge(ei_id, next_ei_target, edge_type='sequential')
            continue

        # Only if no next_ei, do the old inference (shouldn't happen anymore)
        constraint = branch.get('constraint', {})
        has_operation_target = (
                constraint and
                isinstance(constraint, dict) and
                constraint.get('operation_target')
        )

        if has_operation_target:
            # It's a call
            handle_call_ei(
                cfg, branch, i, branches,
                integration_data, callable_inventory,
                ei_registry, stub_queue,
                current_unit_callables, unit_name
            )
        else:
            # Not a call - create branch/sequential edges
            constraint = branch.get('constraint', {})

            # Check if this is part of a conditional (has constraint with polarity)
            if constraint and constraint.get('polarity') is not None:
                # Check if branch has explicit next_ei from control flow
                next_ei_target = branch.get('next_ei')
                if next_ei_target:
                    # Use explicit control flow target
                    cfg.add_edge(ei_id, next_ei_target, edge_type='sequential')
                    continue

                # No explicit next_ei - use alternative collection logic
                is_subsequent = False
                if i > 0:
                    prev_constraint = branches[i - 1].get('constraint', {})
                    if (prev_constraint and
                            prev_constraint.get('expr') == constraint.get('expr') and
                            prev_constraint.get('polarity') is not None):
                        is_subsequent = True

                if is_subsequent:
                    print(f"DEBUG: {ei_id} is subsequent, skipping")
                    if branch.get('is_terminal'):
                        continue
                    # If no explicit next_ei, create sequential edge to next EI
                    if not branch.get('next_ei'):
                        if i + 1 < len(branches):
                            next_ei = branches[i + 1]['id']
                            cfg.add_edge(ei_id, next_ei, edge_type='sequential')
                    continue
                else:
                    # First alternative - collect all and create edges to next
                    print(f"DEBUG: {ei_id} is FIRST alternative, collecting...")
                    expr = constraint.get('expr')
                    alternatives = [ei_id]
                    j = i + 1
                    while j < len(branches):
                        alt = branches[j]
                        alt_constraint = alt.get('constraint', {})
                        if (alt_constraint and alt_constraint.get('expr') == expr and
                                alt_constraint.get('polarity') is not None):
                            alternatives.append(alt['id'])
                            j += 1
                        else:
                            break

                    # j now points to the first EI after all alternatives
                    # Take the next EI (don't skip terminals)
                    next_ei = None
                    if j < len(branches):
                        next_ei = branches[j]['id']

                    # Create an edge from each alternative to the next statement
                    if next_ei:
                        for alt_id in alternatives:
                            print(f"DEBUG: First alt edge: {alt_id} -> {next_ei}")
                            cfg.add_edge(alt_id, next_ei, edge_type='sequential')
                    continue  # Done with this EI

            # This is a regular sequential node or a branching point
            # Find what comes next
            next_line_eis = []

            # Check if next EI(s) are conditional branches with same expression
            if i + 1 < len(branches):
                next_branch = branches[i + 1]
                next_constraint = next_branch.get('constraint', {})

                # If next has a conditional constraint, gather all same-expression alternatives
                if next_constraint and next_constraint.get('polarity') is not None:
                    expr = next_constraint.get('expr')
                    # Collect all consecutive EIs with same expression
                    for j in range(i + 1, len(branches)):
                        alt = branches[j]
                        alt_constraint = alt.get('constraint', {})
                        if (alt_constraint and
                                alt_constraint.get('expr') == expr and
                                alt_constraint.get('polarity') is not None):
                            # Include ALL alternatives, even terminal ones
                            next_line_eis.append(alt['id'])
                        else:
                            break
                else:
                    # Regular sequential - just find next non-terminal
                    print(f"DEBUG: {ei_id} hitting else block")
                    for j in range(i + 1, len(branches)):
                        next_branch = branches[j]
                        next_line_eis.append(next_branch['id'])
                        print(f"DEBUG: Added {next_branch['id']} to next_line_eis")
                        break

            # Create edges to all successors
            for next_ei in next_line_eis:
                edge_type = 'branch' if len(next_line_eis) > 1 else 'sequential'
                print(f"DEBUG: Creating edge {ei_id} -> {next_ei} (type={edge_type})")
                cfg.add_edge(ei_id, next_ei, edge_type=edge_type)


# =============================================================================
# Ledger Processing
# =============================================================================

def extract_unit_callables(ledger_doc: dict[str, Any]) -> dict[str, str]:
    """
    Extract callable name -> callable ID mapping for a unit.

    Args:
        ledger_doc: Ledger document (docKind: 'ledger')

    Returns:
        Dict mapping callable names to their IDs
    """
    callables_map = {}

    def traverse_entries(entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            if entry.get('kind') in ['function', 'method', 'class']:
                callables_map[entry['name']] = entry['id']

            # Recurse into children
            if 'children' in entry:
                traverse_entries(entry['children'])

    unit_entry = ledger_doc.get('unit', {})
    if 'children' in unit_entry:
        traverse_entries(unit_entry['children'])

    return callables_map


def process_ledger(
        cfg: nx.DiGraph,
        ledger_path: Path,
        callable_inventory: dict[str, str],
        ei_registry: dict[str, list[str]],
        stub_queue: dict[str, list[dict[str, Any]]],
        verbose: bool = False
) -> None:
    """
    Process a single ledger file.

    Args:
        cfg: NetworkX graph
        ledger_path: Path to ledger YAML file
        callable_inventory: FQN -> callable_id mapping
        ei_registry: callable_id -> [EI IDs] mapping
        stub_queue: Map of entry_ei_id -> list of pending returns
        verbose: Print progress info
    """
    if verbose:
        print(f"Processing {ledger_path.name}...")

    # Load ledger (3-document YAML)
    with open(ledger_path, 'r', encoding='utf-8') as f:
        docs = list(yaml.safe_load_all(f))

    if len(docs) < 2:
        print(f"Warning: {ledger_path} has fewer than 2 documents")
        return

    # Document 2 is the ledger
    ledger_doc = None
    for doc in docs:
        if doc.get('docKind') == 'ledger':
            ledger_doc = doc
            break

    if not ledger_doc:
        print(f"Warning: No ledger document found in {ledger_path}")
        return

    unit_entry = ledger_doc.get('unit', {})
    unit_name = unit_entry.get('name', '')

    # Build map of callable names -> IDs for this unit
    current_unit_callables = extract_unit_callables(ledger_doc)

    # Process all callables
    def process_entries(entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            if entry.get('kind') in ['function', 'method']:
                process_callable(
                    cfg, entry, unit_name,
                    callable_inventory,
                    ei_registry, stub_queue,
                    current_unit_callables
                )

            # Recurse into children
            if 'children' in entry:
                process_entries(entry['children'])

    if 'children' in unit_entry:
        process_entries(unit_entry['children'])


# =============================================================================
# Graph Building
# =============================================================================

def build_cfg(
        ledger_paths: list[Path],
        callable_inventory: dict[str, str],
        ei_registry: dict[str, list[str]],
        verbose: bool = False
) -> tuple[nx.DiGraph, dict[str, list[dict[str, Any]]]]:
    """
    Build complete control flow graph from ledgers.

    Args:
        ledger_paths: List of ledger file paths
        callable_inventory: FQN -> callable_id mapping
        ei_registry: callable_id -> [EI IDs] mapping (pre-built from inventories)
        verbose: Print progress info

    Returns:
        (cfg, stub_queue) tuple
    """
    cfg = nx.DiGraph()
    stub_queue: dict[str, list[dict[str, Any]]] = {}  # entry_ei_id -> list of pending returns

    # Process all ledgers
    for ledger_path in ledger_paths:
        process_ledger(
            cfg, ledger_path, callable_inventory,
            ei_registry, stub_queue, verbose
        )

    if verbose:
        print(f"\nCFG Statistics:")
        print(f"  Nodes: {cfg.number_of_nodes()}")
        print(f"  Edges: {cfg.number_of_edges()}")
        print(f"  Unresolved stubs remaining: {len(stub_queue)}")

        # Count edge types
        edge_types = defaultdict(int)
        for u, v, data in cfg.edges(data=True):
            edge_types[data.get('edge_type', 'unknown')] += 1

        print(f"\nEdge types:")
        for edge_type, count in sorted(edge_types.items()):
            print(f"    {edge_type}: {count}")

        if stub_queue:
            print(f"\n  Warning: {len(stub_queue)} entry nodes remain stubbed (callables not processed)")
            if verbose and len(stub_queue) <= 10:
                print(f"  Stubbed entry nodes:")
                for entry_ei_id in list(stub_queue.keys())[:10]:
                    print(f"    {entry_ei_id} ({len(stub_queue[entry_ei_id])} callers)")

    return cfg, stub_queue


# =============================================================================
# Serialization
# =============================================================================

def serialize_graph(cfg: nx.DiGraph, output_dir: Path, ser_fmt: str) -> None:
    """Serialize graph to file in the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename based on format
    base_name = 'stage1-ei-cfg'
    if ser_fmt == 'pickle':
        output_path = output_dir / f'{base_name}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(cfg, f)
    elif ser_fmt == 'yaml':
        output_path = output_dir / f'{base_name}.yaml'
        graph_data = nx.node_link_data(cfg)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(graph_data, f, default_flow_style=False)
    elif ser_fmt == 'graphml':
        output_path = output_dir / f'{base_name}.graphml'
        nx.write_graphml(cfg, output_path)
    else:
        raise ValueError(f"Unsupported serialization format: {ser_fmt}")

    print(f"Saved CFG to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--target-root',
        type=Path,
        default=Path.cwd(),
        action=StoreInConfig,
        config_obj=config,
        setter_method='set_target_root',
        help='Target project root (default: current directory)'
    )
    parser.add_argument(
        '--format',
        choices=['pickle', 'yaml', 'graphml'],
        default='pickle',
        help='Output format (default: pickle)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args(argv)

    target_root = config.get_target_root()
    if args.verbose:
        print(f"Target root: {target_root}")

    # Validate inputs
    ledgers_root = config.get_ledgers_root()
    if not ledgers_root.exists():
        print(f"ERROR: Ledgers root not found: {ledgers_root}", file=sys.stderr)
        return 1

    callable_inventory_path = config.get_callable_inventory_path()
    if not callable_inventory_path.exists():
        print(f"ERROR: Callable inventory not found: {callable_inventory_path}", file=sys.stderr)
        return 1

    # Discover ledgers
    ledger_paths = discover_ledgers(ledgers_root)
    if not ledger_paths:
        print(f"ERROR: No ledger YAML files found in {ledgers_root}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(ledger_paths)} ledger file(s)")

    # Load callable inventory
    callable_inventory = load_callable_inventory(callable_inventory_path)

    if args.verbose:
        print(f"Loaded {len(callable_inventory)} callable inventory entries")

    # Discover and load inventory files to build ei_registry
    inventories_root = config.get_inventories_root()
    if not inventories_root.exists():
        print(f"ERROR: Inventories root not found: {inventories_root}", file=sys.stderr)
        return 1

    inventory_paths = discover_inventory_files(inventories_root)
    if not inventory_paths:
        print(f"ERROR: No inventory YAML files found in {inventories_root}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(inventory_paths)} inventory file(s)")

    # Build complete EI registry from inventories
    ei_registry = build_ei_registry_from_inventories(inventory_paths)

    if args.verbose:
        print(f"Built EI registry with {len(ei_registry)} callables")

    # Build CFG
    if args.verbose:
        print("\nBuilding control flow graph...")

    cfg, stub_queue = build_cfg(ledger_paths, callable_inventory, ei_registry, verbose=args.verbose)

    # Serialize graph
    serialize_graph(cfg, config.get_integration_output_dir(), args.format)

    # Write stub queue report if there are unresolved stubs
    if stub_queue:
        stub_report_path = config.get_integration_output_dir() / 'stage1-unresolved-stubs.yaml'

        # Format stub queue for reporting
        stub_report = {
            'total_unresolved': len(stub_queue),
            'stubs': []
        }

        for entry_ei_id, callers in sorted(stub_queue.items()):
            stub_report['stubs'].append({
                'entry_ei_id': entry_ei_id,
                'caller_count': len(callers),
                'callers': [
                    {
                        'call_site': caller['call_site'],
                        'return_to': caller['return_to']
                    }
                    for caller in callers
                ]
            })

        with open(stub_report_path, 'w', encoding='utf-8') as f:
            yaml.dump(stub_report, f, default_flow_style=False, sort_keys=False)

        print(f"\nWrote stub queue report to {stub_report_path}")

    print(f"\n✓ Control flow graph complete")
    print(f"  EI nodes: {cfg.number_of_nodes()}")
    print(f"  Edges: {cfg.number_of_edges()}")
    if stub_queue:
        print(f"  Unresolved stubs: {len(stub_queue)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
