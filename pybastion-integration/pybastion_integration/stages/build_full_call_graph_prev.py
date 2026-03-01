#!/usr/bin/env python3
"""
Build Control Flow Graph (EI-level)

Constructs a complete control flow graph from ledger files using NetworkX where:
  - Nodes are Execution Instances (EIs) - individual statements
  - Edges are control flow transitions (intra-function + inter-function calls)
  - Node attributes include line, condition, outcome, constraints, decorators
  - Edge attributes include transition type and conditions

This creates a unified CFG across the entire codebase suitable for feature
flow tracing from FeatureStart to FeatureEnd decorators.

Input:  *.ledger.yaml files
Output: NetworkX DiGraph object (optionally serialized)

DEFAULT BEHAVIOR (no args):
  - Reads ledgers from config.get_ledgers_root()
  - Outputs to config.get_stage_output('cfg')
  - Format: pickle (NetworkX native)
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
from pybastion_integration.shared.smt_path_checker import PathFeasibilityChecker
from pybastion_integration.shared.models import Branch, ExternalTargetType


# =============================================================================
# Ledger Discovery
# =============================================================================

def discover_ledgers(ledgers_root: Path) -> list[Path]:
    """Discover all *.ledger.yaml files under ledgers_root."""
    return sorted(ledgers_root.rglob('*.ledger.yaml'))


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


# =============================================================================
# EI Node Creation
# =============================================================================

def add_ei_node(cfg: nx.DiGraph, ei_id: str, branch_data: dict[str, Any],
                callable_id: str, unit_name: str, callable_name: str,
                decorators: list[dict[str, Any]]) -> None:
    """
    Add an EI node to the CFG.

    Args:
        cfg: NetworkX graph
        ei_id: Execution instance ID
        branch_data: Branch/EI data from ledger
        callable_id: Parent callable ID
        unit_name: Unit this EI belongs to
        callable_name: Callable this EI belongs to
        decorators: List of decorator dicts (combined callable + statement level)
    """
    condition = branch_data.get('condition', '')
    outcome = branch_data.get('outcome', '')
    is_terminal = branch_data.get('is_terminal', False)
    terminates_via = branch_data.get('terminates_via')
    constraint = branch_data.get('constraint')

    cfg.add_node(
        ei_id,
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
# CFG Edge Creation
# =============================================================================

def add_cfg_edges_from_execution_paths(cfg: nx.DiGraph,
                                       execution_paths: list[list[str]]) -> None:
    """
    Add control flow edges from execution paths.

    Execution paths show sequences of EIs that can execute together.
    Each consecutive pair forms a control flow edge.

    Args:
        cfg: NetworkX graph
        execution_paths: List of EI sequences (e.g., [[E1, E2, E3], [E1, E2, E4]])
    """
    for path in execution_paths:
        for i in range(len(path) - 1):
            from_ei = path[i]
            to_ei = path[i + 1]

            # Add edge if not already present
            if not cfg.has_edge(from_ei, to_ei):
                cfg.add_edge(
                    from_ei,
                    to_ei,
                    edge_type='sequential'
                )


def add_integration_call_edges(cfg: nx.DiGraph, category: str, int_point: dict[str, Any],
                               target_callable_id: str | None,
                               ei_registry: dict[str, list[str]]) -> None:
    """
    Add inter-function call edges from an integration point.

    When EI calls another function, create edge to that function's entry EI
    or to an external target node.

    Args:
        cfg: NetworkX graph
        category: Integration category (e.g., 'interunit', 'extlib', 'stdlib')
        int_point: Full integration point data from ledger (includes id, target, execution_paths, etc.)
        target_callable_id: Callable being called (None for external calls)
        ei_registry: Map of callable_id -> list of EI IDs
    """
    # Extract source EI from integration point
    from_ei = int_point['id']
    if from_ei.startswith('I'):
        from_ei = from_ei[1:]  # Strip the 'I' prefix to get actual EI ID

    # Determine target node
    if target_callable_id and target_callable_id in ei_registry:
        # Interunit call to known callable - use entry EI
        target_eis = ei_registry[target_callable_id]
        if not target_eis:
            return
        to_node = target_eis[0]  # Entry EI
    else:
        # External call - use category-specific external node
        ext_type = ExternalTargetType.from_category(category)
        to_node = ext_type.value

    # Add edge with full integration metadata
    if not cfg.has_edge(from_ei, to_node):
        cfg.add_edge(
            from_ei,
            to_node,
            edge_type='call',
            is_integration_seam=True,
            integration_type=category,
            integration_id=int_point.get('id'),
            integration_kind=int_point.get('kind', 'call'),
            target_raw=int_point.get('target'),
            signature=int_point.get('signature'),
            execution_paths=int_point.get('execution_paths', []),
            target_callable=target_callable_id
        )


# =============================================================================
# Ledger Processing
# =============================================================================

def process_ledger(call_graph: nx.DiGraph, ledger_path: Path,
                   callable_inventory: dict[str, str],
                   ei_registry: dict[str, list[str]],
                   verbose: bool = False) -> None:
    """
    Processes the given ledger file to build and update a call graph and registry for
    entity identifiers (EI), and integrates information from the callable inventory.

    Arguments:
        call_graph (nx.DiGraph): The directed graph that represents relationships
            between callable entities.
        ledger_path (Path): The path to the YAML ledger file containing the ledger document
            and derived-ids document.
        callable_inventory (dict[str, str]): A dictionary mapping callable entity names
            to their respective metadata or definitions.
        ei_registry (dict[str, list[str]]): A registry that maps EI identifiers to a list
            of associated data, which is updated during processing.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Raises:
        None

    Side Effects:
        Updates the call_graph with additional nodes and edges based on the content of
        the ledger file.
        Modifies the ei_registry and callable_inventory based on extracted ledger data.
        Prints warning and informational messages if verbose is set to True.

    Details:
        The function reads the YAML ledger file and identifies two types of documents:
        - 'ledger' documents containing unit and callable information.
        - 'derived-ids' documents holding supplemental information linking entity identifiers
          (EIs) to specific line numbers in the source code.

        Builds a mapping of EI identifiers to line numbers from the derived-ids document if
        available. Extracts unit-level information and processes each callable entity found
        within the unit (including functions, methods, and classes).

        For each callable entity, `_process_callable` is invoked to handle detailed
        integration into the call graph and registry. The function also supports verbose
        mode for displaying progress and any potential warnings.
    """

    # Load all documents from the ledger file
    with open(ledger_path, 'r', encoding='utf-8') as f:
        documents = list(yaml.safe_load_all(f))

    # Find both documents
    derived_doc = None
    ledger_doc = None
    for doc in documents:
        if not doc:
            continue
        kind = doc.get('docKind')
        if kind == 'derived-ids':
            derived_doc = doc
        elif kind == 'ledger':
            ledger_doc = doc

    if not ledger_doc:
        if verbose:
            print(f"  WARNING: No ledger document in {ledger_path.name}")
        return

    # Build EI ID -> line number map from derived-ids
    ei_line_map: dict[str, int] = {}
    if derived_doc:
        assigned = derived_doc.get('assigned', {})
        for branch_info in assigned.get('branches', []):
            ei_id = branch_info.get('id')
            address = branch_info.get('address', '')
            # Extract line from address like "api::resolve@L230"
            if ei_id and '@L' in address:
                try:
                    line_str = address.split('@L')[1]
                    line_num = int(line_str)
                    ei_line_map[ei_id] = line_num
                except (IndexError, ValueError):
                    pass

    if verbose:
        print(f"\n  Built line map with {len(ei_line_map)} entries")
        # Show first few
        for ei_id, line in list(ei_line_map.items())[:5]:
            print(f"    {ei_id} -> L{line}")

    # Extract unit info
    unit = ledger_doc.get('unit', {})
    unit_name = unit.get('name', 'unknown')
    unit_id = unit.get('id', 'unknown')

    if verbose:
        print(f"\rProcessing {unit_name} ({unit_id})...", end='', flush=True)

    # Get all callables from unit.children
    children = unit.get('children', [])

    # Process each callable (function, method, class)
    for entry in children:
        kind = entry.get('kind')

        if kind == 'function':
            _process_callable(call_graph, entry, unit_name, ei_registry,
                              callable_inventory, ei_line_map, verbose)

        elif kind == 'class':
            # Process methods inside the class
            for method in entry.get('children', []):
                if method.get('kind') == 'method':
                    _process_callable(call_graph, method, unit_name, ei_registry,
                                      callable_inventory, ei_line_map, verbose)


def _process_callable(cfg: nx.DiGraph, callable_entry: dict[str, Any],
                      unit_name: str, ei_registry: dict[str, list[str]],
                      callable_inventory: dict[str, str],
                      ei_line_map: dict[str, int],
                      verbose: bool = False) -> None:
    """Process a single callable (function or method)."""
    callable_id = callable_entry['id']
    callable_name = callable_entry['name']

    # Get callable-level decorators
    callable_decorators = callable_entry.get('decorators', [])

    callable_data = callable_entry.get('callable', {})
    branches = callable_data.get('branches', [])

    # Track EI IDs for this callable
    ei_ids = []

    # Create nodes for all EIs in this callable
    for i, branch in enumerate(branches):
        ei_id = branch.get('id')
        if not ei_id:
            continue

        # First EI gets callable-level decorators, all get statement-level decorators
        decorators = branch.get('decorators', [])
        if i == 0:
            decorators = callable_decorators + decorators

        add_ei_node(cfg, ei_id, branch, callable_id, unit_name, callable_name, decorators)
        ei_ids.append(ei_id)

    # Register this callable's EIs
    ei_registry[callable_id] = ei_ids

    # Add intra-function CFG edges (now with line map)
    _add_intra_callable_edges(cfg, branches, unit_name, ei_line_map, verbose)

    # Add inter-function call edges at integration points
    integration_data = callable_data.get('integration', {})
    _add_integration_edges(cfg, integration_data, callable_inventory, ei_registry)


def _add_intra_callable_edges(cfg: nx.DiGraph, branches: list[dict[str, Any]],
                              unit_name: str, ei_line_map: dict[str, int],
                              verbose: bool = False) -> None:
    """Add control flow edges within a callable using pairwise SMT feasibility checking."""

    # Convert branch dicts to Branch objects
    branch_objects = []
    for b in branches:
        try:
            ei_id = b.get('id')
            # Get line from map, fallback to constraint, fallback to 0
            line = ei_line_map.get(ei_id)
            if line is None and b.get('constraint'):
                line = b.get('constraint', {}).get('line', 0)
            if line is None:
                line = 0

            branch_obj_dict = {
                'id': ei_id,
                'line': line,
                'condition': b['condition'],
                'outcome': b['outcome'],
                'constraint': b.get('constraint'),
                'is_terminal': b.get('is_terminal', False),
                'terminates_via': b.get('terminates_via'),
                'decorators': b.get('decorators', []),
                'metadata': b.get('metadata', {})
            }
            branch_obj = Branch.from_dict(branch_obj_dict)
            branch_objects.append(branch_obj)
        except:
            continue

    if not branch_objects:
        return

    # Sort by line number
    sorted_branches = sorted(branch_objects, key=lambda b: b.line)

    # Initialize SMT checker
    checker = PathFeasibilityChecker(timeout_ms=100)

    # Check each EI's potential successors
    total_checks = 0
    for i, current in enumerate(sorted_branches):
        # Skip terminal branches
        if current.is_terminal:
            continue

        # Get excludes from constraint
        excludes = set()
        if current.constraint and hasattr(current.constraint, 'excludes'):
            excludes = set(current.constraint.excludes or [])

        # Find structural successors (next few EIs that aren't excluded)
        successors = []
        for j in range(i + 1, min(i + 6, len(sorted_branches))):
            next_branch = sorted_branches[j]
            if next_branch.id not in excludes:
                successors.append(next_branch)

        # Check feasibility of each successor
        for successor in successors:
            total_checks += 1
            if verbose and total_checks % 100 == 0:
                print(f"\r  {unit_name}... checked {total_checks} edges",
                      end='', flush=True)

            # Check if path [current, successor] is feasible
            result = checker.check_path([current, successor])

            if result.is_feasible:
                cfg.add_edge(
                    current.id,
                    successor.id,
                    edge_type='sequential',
                    feasible=True
                )

    if verbose and total_checks > 0:
        print(f"\r  {unit_name}... added edges from {total_checks} checks" + " " * 20)


def _enumerate_candidate_paths(branches: list[Branch]) -> list[list[str]]:
    """
    Enumerate structurally valid candidate paths through a callable.

    Uses constraint excludes and line ordering to generate plausible paths.
    """
    # Sort by line
    sorted_branches = sorted(branches, key=lambda b: b.line)

    # Build adjacency based on excludes
    adjacency: dict[str, list[str]] = {}

    for i, current in enumerate(sorted_branches):
        if current.is_terminal:
            adjacency[current.id] = []
            continue

        # Get excludes
        excludes = set()
        if current.constraint and hasattr(current.constraint, 'excludes'):
            excludes = set(current.constraint.excludes or [])

        # Find successors (non-excluded EIs at same or later lines)
        successors = []
        for j in range(i + 1, len(sorted_branches)):
            next_branch = sorted_branches[j]
            if next_branch.id not in excludes:
                successors.append(next_branch.id)

        adjacency[current.id] = successors

    # Enumerate paths using DFS
    paths = []
    start_ei = sorted_branches[0].id if sorted_branches else None

    if start_ei:
        _dfs_enumerate_paths(start_ei, adjacency, [], paths, max_depth=50)

    return paths


def _dfs_enumerate_paths(current: str, adjacency: dict[str, list[str]],
                         path: list[str], all_paths: list[list[str]],
                         max_depth: int) -> None:
    """DFS to enumerate all paths through the CFG."""
    path = path + [current]

    # Stop if path is too long
    if len(path) > max_depth:
        return

    # Get successors
    successors = adjacency.get(current, [])

    # If no successors (terminal or end), save path
    if not successors:
        all_paths.append(path)
        return

    # Recurse to successors
    for succ in successors:
        # Avoid cycles
        if succ not in path:
            _dfs_enumerate_paths(succ, adjacency, path, all_paths, max_depth)


def _add_integration_edges(cfg: nx.DiGraph, integration_data: dict[str, dict[str, Any]],
                           callable_inventory: dict[str, str],
                           ei_registry: dict[str, list[str]]) -> None:
    """Add inter-function call edges at integration points."""
    for category in ['interunit', 'extlib', 'stdlib', 'boundary', 'unknown']:
        integration_points: list[dict[str, Any]] = integration_data.get(category, [])

        for int_point in integration_points:
            if not int_point.get('id'):
                continue

            # For integration calls, add edge to target callable
            target_callable_id: str | None = None
            if category == 'interunit':
                target_fqn = int_point.get('target', '')
                target_callable_id = callable_inventory.get(target_fqn)

            add_integration_call_edges(
                cfg, category, int_point, target_callable_id, ei_registry
            )


# =============================================================================
# Graph Building
# =============================================================================

def build_cfg(ledger_paths: list[Path], callable_inventory: dict[str, str],
              verbose: bool = False) -> nx.DiGraph:
    """
    Build complete control flow graph from ledgers.

    Returns:
        NetworkX DiGraph with EI nodes and control flow edges
    """
    cfg = nx.DiGraph()
    ei_registry: dict[str, list[str]] = {}

    for ext_type in ExternalTargetType:
        cfg.add_node(ext_type.value, node_type="external_target")

    # First pass: create all EI nodes
    for ledger_path in ledger_paths:
        process_ledger(cfg, ledger_path, callable_inventory, ei_registry, verbose)

    if verbose:
        print(f"\nCFG statistics:")
        print(f"  Nodes (EIs): {cfg.number_of_nodes()}")
        print(f"  Edges: {cfg.number_of_edges()}")

        # Count nodes with decorators
        nodes_with_decorators = sum(
            1 for n, d in cfg.nodes(data=True)
            if d.get('decorators') and len(d.get('decorators', [])) > 0
        )
        print(f"  EIs with decorators: {nodes_with_decorators}")

        # Count terminal nodes
        terminal_nodes = sum(
            1 for n, d in cfg.nodes(data=True)
            if d.get('is_terminal')
        )
        print(f"  Terminal EIs: {terminal_nodes}")

    return cfg


# =============================================================================
# Serialization
# =============================================================================

def serialize_graph(cfg: nx.DiGraph, output: Path, ser_fmt: str) -> None:
    """Serialize graph to file."""
    # Auto-generate filename based on format
    base_name = 'stage1-full-call-graph'
    if ser_fmt == 'pickle':
        default_output = output / f'{base_name}.pkl'
        with open(default_output, 'wb') as f:
            pickle.dump(cfg, f)
    elif ser_fmt == 'yaml':
        default_output = output / f'{base_name}.yaml'
        graph_data = nx.node_link_data(cfg)
        with open(default_output, 'w', encoding='utf-8') as f:
            yaml.dump(graph_data, f, default_flow_style=False)
    elif ser_fmt == 'graphml':
        default_output = output / f'{base_name}.graphml'
        nx.write_graphml(cfg, default_output)
    else:
        raise ValueError(f"Unsupported serialization format: {ser_fmt}")


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        default=config.get_integration_output_dir(),
        help='Output dir (for serialization)'
    )
    ap.add_argument(
        '--format',
        choices=['pickle', 'yaml', 'graphml'],
        default='pickle',
        help='Output format (default: pickle)'
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

    if args.target_root:
        config.set_target_root(args.target_root)
        if args.verbose:
            print(f"Target root: {args.target_root}")

    if not args.ledgers_root.exists():
        print(f"ERROR: Ledgers root not found: {args.ledgers_root}",
              file=sys.stderr)
        return 1

    ledger_paths = discover_ledgers(args.ledgers_root)
    if not ledger_paths:
        print(f"ERROR: No *.ledger.yaml files found in {args.ledgers_root}",
              file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(ledger_paths)} ledger(s)")

    # Load callable inventory
    inventory_path = config.get_callable_inventory_path()
    callable_inventory = load_callable_inventory(inventory_path)

    if args.verbose:
        print(f"Loaded {len(callable_inventory)} callable inventory entries")

    # Build CFG
    if args.verbose:
        print("\nBuilding control flow graph...")

    cfg = build_cfg(ledger_paths, callable_inventory, verbose=args.verbose)

    # Serialize if output specified
    if args.output:
        serialize_graph(cfg, config.get_integration_output_dir(), args.format)
        print(f"\n✓ CFG saved to {config.get_integration_output_dir()} ({args.format} format)")

    print(f"\n✓ Control flow graph complete")
    print(f"  EI nodes: {cfg.number_of_nodes()}")
    print(f"  Control flow edges: {cfg.number_of_edges()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())