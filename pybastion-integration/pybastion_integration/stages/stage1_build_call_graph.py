#!/usr/bin/env python3
"""
Stage 1: Build Feasibility-Weighted Call Graph

Constructs a call graph from ledger and inventory files where:
  - Nodes are callables (functions, methods, classes)
  - Edges are calls between callables, gated by path feasibility
  - Edge weights reflect pre-computed feasible execution path counts (from inventories)
  - Edges crossing unit boundaries are flagged as integration seams

Input:  *.ledger.yaml  (callable nodes, classified integration points, branch constraints)
        *.inventory.yaml (pre-computed path feasibility counts, param type information)
Output: stage1-call-graph.yaml

DEFAULT BEHAVIOR (no args):
  - Reads ledgers from config.get_ledgers_root()
  - Reads inventories from config.get_inventories_root()
  - Outputs to config.get_stage_output(1)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

from pybastion_integration import config


# =============================================================================
# Data structures
# =============================================================================

class CallableNode:
    def __init__(self, node_id, unit_name, unit_id, callable_id, callable_name,
                 qualified_name, fully_qualified, kind, classification, is_mechanical=False,
                 signature=None, unknown=False):
        self.node_id = node_id
        self.unit_name = unit_name
        self.unit_id = unit_id
        self.callable_id = callable_id
        self.callable_name = callable_name
        self.qualified_name = qualified_name
        self.fully_qualified = fully_qualified
        self.kind = kind
        self.classification = classification
        self.is_mechanical = is_mechanical
        self.signature = signature
        self.unknown = unknown

    def to_dict(self) -> dict:
        return {
            'id': self.node_id,
            'unit': self.unit_name,
            'unit_id': self.unit_id,
            'callable_id': self.callable_id,
            'name': self.callable_name,
            'qualified_name': self.qualified_name,
            'fully_qualified': self.fully_qualified,
            'kind': self.kind,
            'classification': self.classification,
            'is_mechanical': self.is_mechanical,
            'signature': self.signature,
            'unknown': self.unknown,
        }


class CallEdge:
    def __init__(self, edge_id, from_node_id, to_node_id, from_callable, to_callable,
                 target_raw, is_integration_seam, feasible_path_count, total_path_count,
                 integration_id, integration_kind, integration_type='unknown',
                 signature=None, unknown=False, execution_paths=None):
        self.edge_id = edge_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.from_callable = from_callable
        self.to_callable = to_callable
        self.target_raw = target_raw
        self.is_integration_seam = is_integration_seam
        self.feasible_path_count = feasible_path_count
        self.total_path_count = total_path_count
        self.integration_id = integration_id
        self.integration_kind = integration_kind
        self.integration_type = integration_type
        self.signature = signature
        self.unknown = unknown
        self.execution_paths = execution_paths or []

    def to_dict(self) -> dict:
        return {
            'id': self.edge_id,
            'from': self.from_node_id,
            'to': self.to_node_id,
            'from_callable': self.from_callable,
            'to_callable': self.to_callable,
            'target_raw': self.target_raw,
            'is_integration_seam': self.is_integration_seam,
            'integration_type': self.integration_type,
            'feasible_path_count': self.feasible_path_count,
            'total_path_count': self.total_path_count,
            'integration_id': self.integration_id,
            'integration_kind': self.integration_kind,
            'signature': self.signature,
            'unknown': self.unknown,
            'execution_paths': self.execution_paths,
        }


class CallGraph:
    def __init__(self):
        self.nodes: list[CallableNode] = []
        self.edges: list[CallEdge] = []

    def to_dict(self) -> dict:
        return {
            'stage': 'call-graph',
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
            'stats': {
                'total_nodes': len(self.nodes),
                'unknown_nodes': sum(1 for n in self.nodes if n.unknown),
                'mechanical_nodes': sum(1 for n in self.nodes if n.is_mechanical),
                'total_edges': len(self.edges),
                'integration_seam_edges': sum(1 for e in self.edges if e.is_integration_seam),
                'interunit_edges': sum(1 for e in self.edges if e.integration_type == 'interunit'),
                'stdlib_edges': sum(1 for e in self.edges if e.integration_type == 'stdlib'),
                'extlib_edges': sum(1 for e in self.edges if e.integration_type == 'extlib'),
                'boundary_edges': sum(1 for e in self.edges if e.integration_type == 'boundary'),
                'unknown_edges': sum(1 for e in self.edges if e.integration_type == 'unknown'),
            }
        }


# =============================================================================
# File discovery and loading
# =============================================================================

def discover_ledgers(root: Path) -> list[Path]:
    return sorted(root.glob('**/*.ledger.yaml'))


def discover_inventories(root: Path) -> list[Path]:
    return sorted(root.glob('**/*.inventory.yaml'))


def load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_ledger_documents(path: Path) -> tuple[dict | None, dict | None]:
    with open(path, 'r', encoding='utf-8') as f:
        documents = list(yaml.safe_load_all(f))
    derived_doc = None
    ledger_doc = None
    for doc in documents:
        if not doc:
            continue
        kind = doc.get('docKind', '')
        if kind == 'derived-ids':
            derived_doc = doc
        elif kind == 'ledger':
            ledger_doc = doc
    return derived_doc, ledger_doc


# =============================================================================
# Inventory index
# =============================================================================

class InventoryIndex:
    """
    Index of pre-computed feasibility data from inventory files.
    Keyed by unit_id -> callable_id -> target -> path_analysis dict.
    """

    def __init__(self):
        self._index: dict[str, dict[str, dict[str, dict]]] = {}

    def load(self, inventory_paths: list[Path], verbose: bool = False) -> None:
        if verbose:
            print(f"  Loading {len(inventory_paths)} inventory file(s)...")

        for path in inventory_paths:
            try:
                data = load_yaml(path)
            except Exception as e:
                print(f"  WARNING: Could not load {path}: {e}", file=sys.stderr)
                continue

            unit_id = data.get('unit_id', '')
            if not unit_id:
                continue

            if unit_id not in self._index:
                self._index[unit_id] = {}

            for entry in data.get('entries', []):
                entry_id = entry.get('id', '')
                if not entry_id:
                    continue

                ast_analysis = entry.get('ast_analysis', {}) or {}
                candidates = ast_analysis.get('integration_candidates', []) or []

                if not candidates:
                    continue

                if entry_id not in self._index[unit_id]:
                    self._index[unit_id][entry_id] = {}

                for candidate in candidates:
                    target = candidate.get('target', '')
                    path_analysis = candidate.get('path_analysis', {})
                    if target and path_analysis:
                        self._index[unit_id][entry_id][target] = path_analysis

        if verbose:
            total_candidates = sum(
                len(targets)
                for callables in self._index.values()
                for targets in callables.values()
            )
            print(f"  Indexed {total_candidates} integration candidates across {len(self._index)} unit(s)")

    def get_path_analysis(self, unit_id: str, callable_id: str, target: str) -> dict | None:
        return self._index.get(unit_id, {}).get(callable_id, {}).get(target)


# =============================================================================
# Callable index
# =============================================================================

def make_node_id(fully_qualified: str) -> str:
    return 'CG_' + hashlib.sha256(fully_qualified.encode()).hexdigest()[:12].upper()


def make_edge_id(from_node_id: str, to_node_id: str, integration_id: str) -> str:
    key = f"{from_node_id}→{to_node_id}→{integration_id}"
    return 'CE_' + hashlib.sha256(key.encode()).hexdigest()[:12].upper()


def build_callable_index(
        ledger_paths: list[Path],
        verbose: bool = False
) -> tuple[dict[str, CallableNode], dict[str, CallableNode]]:
    nodes_by_fqn: dict[str, CallableNode] = {}
    nodes_by_callable_id: dict[str, CallableNode] = {}

    if verbose:
        print(f"  Building callable index from {len(ledger_paths)} ledger(s)...")

    for ledger_path in ledger_paths:
        _, ledger_doc = load_ledger_documents(ledger_path)
        if not ledger_doc:
            continue

        unit = ledger_doc.get('unit', {})
        unit_name = unit.get('name', 'unknown')
        unit_id = unit.get('id', '')

        def walk_entry(entry: dict, parent_qualified: str = '') -> None:
            entry_id = entry.get('id', '')
            entry_kind = entry.get('kind', '')
            entry_name = entry.get('name', '')
            entry_sig = entry.get('signature')

            if not entry_id or not entry_name:
                return

            qualified = f"{parent_qualified}.{entry_name}" if parent_qualified else entry_name
            fully_qualified = f"{unit_name}::{qualified}"

            is_mechanical = any(
                d.get('name') in ('MechanicalOperation', 'UtilityOperation')
                for d in entry.get('decorators', [])
            )

            if entry_kind in ('function', 'method', 'class', 'enum'):
                node = CallableNode(
                    node_id=make_node_id(fully_qualified),
                    unit_name=unit_name,
                    unit_id=unit_id,
                    callable_id=entry_id,
                    callable_name=entry_name,
                    qualified_name=qualified,
                    fully_qualified=fully_qualified,
                    kind=entry_kind,
                    classification=None,
                    is_mechanical=is_mechanical,
                    signature=entry_sig,
                )
                nodes_by_fqn[fully_qualified] = node
                nodes_by_callable_id[entry_id] = node

            new_parent = qualified if entry_kind == 'class' else parent_qualified
            for child in entry.get('children', []):
                walk_entry(child, new_parent)

        walk_entry(unit)

    if verbose:
        print(f"  Indexed {len(nodes_by_fqn)} callable nodes")

    return nodes_by_fqn, nodes_by_callable_id


# =============================================================================
# Target resolution
# =============================================================================

def resolve_fqn_to_node(
        fqn_target: str,
        nodes_by_fqn: dict[str, CallableNode],
) -> CallableNode | None:
    """
    Resolve a FQN target to a CallableNode by trying progressively shorter suffixes.

    e.g. "project_resolution_engine.model.keys.WheelKey.set_dependency_ids"
    matches node keyed as "keys::WheelKey.set_dependency_ids"
    """
    if not fqn_target:
        return None

    parts = fqn_target.split('.')
    for i in range(len(parts)):
        suffix = '.'.join(parts[i:])
        for fqn, node in nodes_by_fqn.items():
            if node.qualified_name == suffix or node.callable_name == suffix:
                return node
            if fqn.endswith(f"::{suffix}"):
                return node
            if '.' in suffix:
                first, rest = suffix.split('.', 1)
                if fqn == f"{first}::{rest}":
                    return node

    return None


# =============================================================================
# Graph construction
# =============================================================================

def _make_unknown_node(fqn: str, nodes_by_fqn: dict, integration_type: str = 'unknown') -> CallableNode:
    """
    Materialize an unresolved target as a proper graph node.
    Idempotent — returns existing node if already created for this FQN.
    The kind reflects the integration classification: interunit, stdlib, extlib, boundary, or unknown.
    """
    key = f'UNKNOWN::{fqn}'
    if key in nodes_by_fqn:
        return nodes_by_fqn[key]

    # Derive a readable name from the FQN tail
    name = fqn.split('.')[-1] if fqn else 'unknown'

    node = CallableNode(
        node_id=make_node_id(key),
        unit_name='unknown',
        unit_id=None,
        callable_id=None,
        callable_name=name,
        qualified_name=fqn,
        fully_qualified=fqn,
        kind='unknown',
        classification=integration_type,
        is_mechanical=False,
        signature=None,
        unknown=True,
    )
    nodes_by_fqn[key] = node
    return node


def _emit_edge(
        source_node: CallableNode,
        target_node: CallableNode | None,
        target_fqn: str,
        integ: dict,
        unit_id: str,
        entry_id: str,
        inventory_index: InventoryIndex,
        is_seam: bool,
        integration_type: str,
        stats: dict,
        edges: list,
        nodes_by_fqn: dict,
) -> None:
    """Resolve feasibility, materialize unknown nodes, build and append a CallEdge."""
    integ_id = integ.get('id', '')
    integ_kind = integ.get('kind', 'call')
    integ_sig = integ.get('signature')
    exec_paths = integ.get('execution_paths') or []
    total_paths = len(exec_paths)

    # Use pre-computed feasibility from inventory
    path_analysis = inventory_index.get_path_analysis(unit_id, entry_id, target_fqn)
    if path_analysis:
        feasible = path_analysis.get('feasible_paths', total_paths)
        total = path_analysis.get('total_syntactic_paths', total_paths)
    else:
        # Conservative fallback: assume all paths feasible
        feasible = total_paths
        total = total_paths
        stats['no_path_analysis'] += 1

    if feasible == 0 and total > 0:
        stats['infeasible_pruned'] += 1
        return

    if target_node:
        resolved_node = target_node
        unknown = False
    else:
        # Materialize unknown target as a proper graph node
        resolved_node = _make_unknown_node(target_fqn, nodes_by_fqn, integration_type)
        unknown = True

    edge_id = make_edge_id(source_node.node_id, resolved_node.node_id, integ_id)
    edges.append(
        CallEdge(
            edge_id=edge_id,
            from_node_id=source_node.node_id,
            to_node_id=resolved_node.node_id,
            from_callable=source_node.fully_qualified,
            to_callable=resolved_node.fully_qualified,
            target_raw=target_fqn,
            is_integration_seam=is_seam,
            integration_type=integration_type,
            feasible_path_count=feasible,
            total_path_count=total,
            integration_id=integ_id,
            integration_kind=integ_kind,
            signature=integ_sig,
            unknown=unknown,
            execution_paths=exec_paths,
        )
    )


def build_call_graph(
        ledger_paths: list[Path],
        inventory_index: InventoryIndex,
        verbose: bool = False
) -> CallGraph:
    """Build the feasibility-weighted call graph from ledgers + inventories."""

    nodes_by_fqn, _ = build_callable_index(ledger_paths, verbose=verbose)

    graph = CallGraph()
    # Note: graph.nodes is rebuilt after edge processing to include
    # any unknown nodes materialized during _emit_edge calls.

    if verbose:
        print(f"\n  Processing {len(ledger_paths)} ledger(s) for edges...")

    edges: list[CallEdge] = []
    stats = {
        'interunit_found': 0,
        'interunit_resolved': 0,
        'interunit_unresolved': 0,
        'stdlib_found': 0,
        'extlib_found': 0,
        'boundary_found': 0,
        'unknown_found': 0,
        'infeasible_pruned': 0,
        'no_path_analysis': 0,
    }

    for ledger_path in ledger_paths:
        _, ledger_doc = load_ledger_documents(ledger_path)
        if not ledger_doc:
            continue

        unit = ledger_doc.get('unit', {})
        unit_name = unit.get('name', 'unknown')
        unit_id = unit.get('id', '')

        def walk_entry_for_edges(entry: dict, parent_qualified: str = '') -> None:
            entry_id = entry.get('id', '')
            entry_name = entry.get('name', '')
            entry_kind = entry.get('kind', '')

            qualified = f"{parent_qualified}.{entry_name}" if parent_qualified else entry_name
            fully_qualified = f"{unit_name}::{qualified}"
            source_node = nodes_by_fqn.get(fully_qualified)

            callable_data = entry.get('callable', {})
            integration = callable_data.get('integration', {})

            if source_node and integration:
                # Skip emitting edges from mechanical/utility operations
                if source_node.is_mechanical:
                    return

                # Interunit edges — FQN targets, confirmed integration seams
                for integ in (integration.get('interunit') or []):
                    stats['interunit_found'] += 1
                    target_fqn = integ.get('target', '')
                    target_node = resolve_fqn_to_node(target_fqn, nodes_by_fqn)
                    if target_node:
                        stats['interunit_resolved'] += 1
                    else:
                        stats['interunit_unresolved'] += 1
                    _emit_edge(
                        source_node, target_node, target_fqn, integ,
                        unit_id, entry_id, inventory_index,
                        is_seam=True, integration_type='interunit',
                        stats=stats, edges=edges, nodes_by_fqn=nodes_by_fqn,
                    )

                # Stdlib edges
                for integ in (integration.get('stdlib') or []):
                    stats['stdlib_found'] += 1
                    target_raw = integ.get('target', '')
                    _emit_edge(
                        source_node, None, target_raw, integ,
                        unit_id, entry_id, inventory_index,
                        is_seam=False, integration_type='stdlib',
                        stats=stats, edges=edges, nodes_by_fqn=nodes_by_fqn,
                    )

                # Extlib edges
                for integ in (integration.get('extlib') or []):
                    stats['extlib_found'] += 1
                    target_raw = integ.get('target', '')
                    _emit_edge(
                        source_node, None, target_raw, integ,
                        unit_id, entry_id, inventory_index,
                        is_seam=False, integration_type='extlib',
                        stats=stats, edges=edges, nodes_by_fqn=nodes_by_fqn,
                    )

                # Boundary edges
                for integ in (integration.get('boundaries') or []):
                    stats['boundary_found'] += 1
                    target_raw = integ.get('target', '')
                    _emit_edge(
                        source_node, None, target_raw, integ,
                        unit_id, entry_id, inventory_index,
                        is_seam=True, integration_type='boundary',
                        stats=stats, edges=edges, nodes_by_fqn=nodes_by_fqn,
                    )

                # Unknown edges — unresolved targets, flagged for agent
                for integ in (integration.get('unknown') or []):
                    stats['unknown_found'] += 1
                    target_raw = integ.get('target', '')
                    _emit_edge(
                        source_node, None, target_raw, integ,
                        unit_id, entry_id, inventory_index,
                        is_seam=False, integration_type='unknown',
                        stats=stats, edges=edges, nodes_by_fqn=nodes_by_fqn,
                    )

            new_parent = qualified if entry_kind == 'class' else parent_qualified
            for child in entry.get('children', []):
                walk_entry_for_edges(child, new_parent)

        walk_entry_for_edges(unit)

    graph.edges = edges
    # Rebuild nodes list to include unknown nodes materialized during edge processing
    graph.nodes = list(nodes_by_fqn.values())

    if verbose:
        print(f"\n  Edge stats:")
        print(f"    Interunit found:               {stats['interunit_found']}")
        print(f"    Resolved to nodes:             {stats['interunit_resolved']}")
        print(f"    Unresolved (unknown target):   {stats['interunit_unresolved']}")
        print(f"    Stdlib found:                  {stats['stdlib_found']}")
        print(f"    Extlib found:                  {stats['extlib_found']}")
        print(f"    Boundary found:                {stats['boundary_found']}")
        print(f"    Unknown found:                 {stats['unknown_found']}")
        print(f"    Pruned (all paths infeasible): {stats['infeasible_pruned']}")
        print(f"    No inventory path analysis:    {stats['no_path_analysis']}")

    return graph


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
        '--inventories-root',
        type=Path,
        default=config.get_inventories_root(),
        help=f'Root directory for inventory discovery (default: {config.get_inventories_root()})'
    )
    ap.add_argument(
        '--output',
        type=Path,
        default=config.get_stage_output(1),
        help=f'Output file (default: {config.get_stage_output(1)})'
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
        print(f"ERROR: Ledgers root not found: {args.ledgers_root}", file=sys.stderr)
        return 1

    ledger_paths = discover_ledgers(args.ledgers_root)
    if not ledger_paths:
        print(f"ERROR: No *.ledger.yaml files found in {args.ledgers_root}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(ledger_paths)} ledger(s)")

    # Load inventories
    inventory_index = InventoryIndex()
    if args.inventories_root.exists():
        inventory_paths = discover_inventories(args.inventories_root)
        if args.verbose:
            print(f"Found {len(inventory_paths)} inventory file(s)")
        inventory_index.load(inventory_paths, verbose=args.verbose)
    else:
        print(f"WARNING: Inventories root not found: {args.inventories_root}", file=sys.stderr)
        print(f"         Continuing without pre-computed feasibility data", file=sys.stderr)

    # Build graph
    if args.verbose:
        print("\nBuilding call graph...")

    graph = build_call_graph(ledger_paths, inventory_index, verbose=args.verbose)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(
            graph.to_dict(),
            f,
            default_flow_style=False,
            sort_keys=config.get_yaml_sort_keys(),
            width=config.get_yaml_width(),
            indent=config.get_yaml_indent(),
        )

    stats = graph.to_dict()['stats']
    print(f"\n✓ Call graph complete → {args.output}")
    print(f"  Nodes:              {stats['total_nodes']} ({stats['unknown_nodes']} unknown)")
    print(f"  Mechanical nodes:   {stats['mechanical_nodes']}")
    print(f"  Edges:              {stats['total_edges']}")
    print(f"  Integration seams:  {stats['integration_seam_edges']}")
    print(f"  Interunit edges:    {stats['interunit_edges']}")
    print(f"  Stdlib edges:       {stats['stdlib_edges']}")
    print(f"  Extlib edges:       {stats['extlib_edges']}")
    print(f"  Boundary edges:     {stats['boundary_edges']}")
    print(f"  Unknown edges:      {stats['unknown_edges']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())