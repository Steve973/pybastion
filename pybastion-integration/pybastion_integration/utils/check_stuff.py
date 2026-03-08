#!/usr/bin/env python3
"""
CFG Path Diagnostics

Analyzes paths in the CFG to identify where paths break and why.
"""

import pickle
import re
import sys
from pathlib import Path

import networkx as nx


FUNC_ID_PATTERN: re.Pattern[str] = re.compile(r"^U[0-9A-F]{10}_[FM]\d{3}")


def load_cfg(cfg_path: Path) -> nx.DiGraph:
    """Load CFG from pickle."""
    with open(cfg_path, 'rb') as f:
        return pickle.load(f)


def trace_forward(cfg: nx.DiGraph, start: str, target: str, max_steps: int = 100):
    """
    Trace forward from start, following first available edge, reporting progress.

    Returns dict with:
    - path: list of node IDs
    - reached_target: bool
    - dead_end: node ID where path ended (if not target)
    - dead_end_reason: why it ended
    """
    current = start
    visited = set()
    path = [current]

    for i in range(max_steps):
        if current == target:
            return {
                'reached_target': True,
                'path': path,
                'steps': i
            }

        visited.add(current)
        edges = list(cfg.out_edges(current, data=True))

        if not edges:
            node = cfg.nodes[current]
            return {
                'reached_target': False,
                'path': path,
                'dead_end': current,
                'dead_end_reason': 'no_outgoing_edges',
                'is_terminal': node.get('is_terminal'),
                'terminates_via': node.get('terminates_via'),
                'outcome': node.get('outcome')
            }

        # Take first unvisited edge
        next_node = None
        for _, tgt, data in edges:
            if tgt not in visited:
                next_node = tgt
                break

        if not next_node:
            return {
                'reached_target': False,
                'path': path,
                'dead_end': current,
                'dead_end_reason': 'all_edges_visited'
            }

        path.append(next_node)
        current = next_node

    return {
        'reached_target': False,
        'path': path,
        'dead_end': current,
        'dead_end_reason': 'max_steps_exceeded'
    }


def analyze_ei_sequence(cfg: nx.DiGraph, callable_id: str, start_num: int, end_num: int):
    """
    Analyze a sequence of EIs in a callable.

    Returns list of dicts with EI info.
    """
    results = []

    # Extract unit and function IDs from callable_id
    # Format: UXXXXXXXXX_FXXXX or UXXXXXXXXX_CXXX_MXXX
    parts = callable_id.split('_')
    unit_id = parts[0]
    func_id = '_'.join(parts[1:])

    for i in range(start_num, end_num + 1):
        ei_id = f"{callable_id}_E{i:04d}"

        if not cfg.has_node(ei_id):
            results.append({'ei_id': ei_id, 'exists': False})
            continue

        node = cfg.nodes[ei_id]
        edges = list(cfg.out_edges(ei_id, data=True))

        ei_info = {
            'ei_id': ei_id,
            'exists': True,
            'outcome': node.get('outcome', ''),
            'is_terminal': node.get('is_terminal', False),
            'terminates_via': node.get('terminates_via'),
            'outgoing_edges': []
        }

        for _, target, data in edges:
            ei_info['outgoing_edges'].append({
                'target': target,
                'edge_type': data.get('edge_type')
            })

        results.append(ei_info)

    return results


def find_exit_eis(cfg: nx.DiGraph, callable_id: str):
    """Find all exit EIs for a callable (returns and yields)."""
    exits = []

    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get('callable_id') != callable_id:
            continue

        is_terminal = node_data.get('is_terminal', False)
        terminates_via = node_data.get('terminates_via', '')
        outcome = node_data.get('outcome', '')

        if is_terminal and terminates_via in ('return', 'implicit-return', 'yield'):
            exits.append({
                'ei_id': node_id,
                'type': 'return',
                'outcome': outcome
            })
        elif 'yield' in outcome.lower():
            exits.append({
                'ei_id': node_id,
                'type': 'yield',
                'outcome': outcome
            })

    return exits


def check_callable_integrity(cfg, callable_id):
    """Check if a callable has valid internal structure."""
    entry = f"{callable_id}_E000" + ("0" if FUNC_ID_PATTERN.match(callable_id) else "1")

    # Check entry exists
    if not cfg.has_node(entry):
        return {
            'callable_id': callable_id,
            'valid': False,
            'issue': 'no_entry_node',
            'entry': entry
        }

    # Find exit EIs (terminal returns)
    exits = []
    for nid, ndata in cfg.nodes(data=True):
        if (ndata.get('callable_id') == callable_id and
                ndata.get('is_terminal') and
                ndata.get('terminates_via') in ('return', 'implicit-return', 'yield')):
            exits.append(nid)

    if not exits:
        return {
            'callable_id': callable_id,
            'valid': False,
            'issue': 'no_exit_nodes',
            'entry': entry
        }

    # Check if path exists from entry to any exit
    has_path = False
    reachable_exit = None
    for exit_node in exits:
        try:
            nx.shortest_path(cfg, entry, exit_node)
            has_path = True
            reachable_exit = exit_node
            break
        except nx.NetworkXNoPath:
            pass

    if not has_path:
        return {
            'callable_id': callable_id,
            'valid': False,
            'issue': 'no_internal_path',
            'entry': entry,
            'exits': exits
        }

    return {
        'callable_id': callable_id,
        'valid': True,
        'entry': entry,
        'exits': exits,
        'reachable_exit': reachable_exit
    }


def check_return_edges(cfg, callable_id):
    """Check if callable's exits have return edges back to callers."""
    # Find all exit EIs
    exits = []
    for nid, ndata in cfg.nodes(data=True):
        if (ndata.get('callable_id') == callable_id and
                ndata.get('is_terminal') and
                ndata.get('terminates_via') in ('return', 'implicit-return', 'yield')):
            exits.append(nid)

    if not exits:
        return {
            'callable_id': callable_id,
            'has_exits': False,
            'exits': []
        }

    # Check each exit for return edges
    exit_info = []
    for exit_ei in exits:
        out_edges = list(cfg.out_edges(exit_ei, data=True))
        return_edges = [e for e in out_edges if e[2].get('edge_type') == 'return']

        exit_info.append({
            'exit_ei': exit_ei,
            'has_return_edges': len(return_edges) > 0,
            'return_count': len(return_edges),
            'returns_to': [tgt for _, tgt, _ in return_edges]
        })

    all_have_returns = all(e['has_return_edges'] for e in exit_info)

    return {
        'callable_id': callable_id,
        'has_exits': True,
        'exits': exit_info,
        'all_exits_have_returns': all_have_returns
    }


def check_call_coverage(cfg, callable_id):
    """Check if callable is ever called (has incoming call edges)."""
    entry = f"{callable_id}_E000" + ("0" if FUNC_ID_PATTERN.match(callable_id) else "1")

    if not cfg.has_node(entry):
        return {
            'callable_id': callable_id,
            'entry_exists': False
        }

    in_edges = list(cfg.in_edges(entry, data=True))
    call_edges = [e for e in in_edges if e[2].get('edge_type') == 'call']

    callers = []
    for src, _, _ in call_edges:
        caller_callable = cfg.nodes[src].get('callable_id')
        callers.append({
            'call_site': src,
            'caller_callable': caller_callable
        })

    return {
        'callable_id': callable_id,
        'entry_exists': True,
        'is_called': len(call_edges) > 0,
        'call_count': len(call_edges),
        'callers': callers
    }


def diagnose_all_callables(cfg):
    """Run all diagnostic checks on all callables in the graph."""
    # Get unique callable IDs
    callables = set()
    for _, node_data in cfg.nodes(data=True):
        cid = node_data.get('callable_id')
        if cid:
            callables.add(cid)

    results = []
    for callable_id in sorted(callables):
        integrity = check_callable_integrity(cfg, callable_id)
        returns = check_return_edges(cfg, callable_id)
        coverage = check_call_coverage(cfg, callable_id)

        results.append({
            'callable_id': callable_id,
            'integrity': integrity,
            'returns': returns,
            'coverage': coverage,
            'is_broken': not integrity['valid']
        })

    return results


def diagnose_callable_detail(cfg, callable_id):
    """
    Show detailed diagnostics for a single callable.

    Returns dict with:
    - all EIs (with their properties)
    - edges between them
    - entry/exit identification
    - path existence
    """
    # Gather all EIs for this callable
    eis = []
    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get('callable_id') == callable_id:
            eis.append({
                'id': node_id,
                'is_terminal': node_data.get('is_terminal', False),
                'terminates_via': node_data.get('terminates_via', ''),
                'outcome': node_data.get('outcome', ''),
                'condition': node_data.get('condition', '')
            })

    # Sort by EI number
    eis.sort(key=lambda e: int(e['id'].split('_E')[-1]))

    # Find entry (E0000)
    entry = f"{callable_id}_E000" + ("0" if FUNC_ID_PATTERN.match(callable_id) else "1")

    # Find exits
    exits = []
    for ei in eis:
        if (ei['is_terminal'] and
                ei['terminates_via'] in ('return', 'implicit-return', 'yield')):
            exits.append(ei['id'])

    # Get edges between EIs in this callable
    internal_edges = []
    for ei in eis:
        for _, target, edge_data in cfg.out_edges(ei['id'], data=True):
            target_callable = cfg.nodes[target].get('callable_id')
            edge_type = edge_data.get('edge_type')
            internal_edges.append({
                'from': ei['id'],
                'to': target,
                'edge_type': edge_type,
                'target_callable': target_callable,
                'is_internal': target_callable == callable_id
            })

    external_edges = []
    for node_id in cfg.nodes():
        if node_id.startswith('EXTERNAL'):
            node_callable = cfg.nodes[node_id].get('called_from', '')
            if callable_id in node_callable:
                for _, target, edge_data in cfg.out_edges(node_id, data=True):
                    external_edges.append({
                        'from': node_id,
                        'to': target,
                        'edge_type': edge_data.get('edge_type')
                    })

    # Get incoming edges to entry
    entry_in_edges = []
    if cfg.has_node(entry):
        for src, _, edge_data in cfg.in_edges(entry, data=True):
            src_callable = cfg.nodes[src].get('callable_id')
            entry_in_edges.append({
                'from': src,
                'from_callable': src_callable,
                'edge_type': edge_data.get('edge_type')
            })

    return {
        'callable_id': callable_id,
        'entry': entry,
        'exits': exits,
        'entry_incoming': entry_in_edges,
        'eis': eis,
        'edges': internal_edges,
        'external_edges': external_edges
    }


def analyze_broken_callable(cfg, callable_id):
    """
    Analyze why a callable is broken - look for same-line EIs without edges.
    """
    # Get all EIs
    eis = []
    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get('callable_id') == callable_id:
            constraint = node_data.get('constraint', {})
            eis.append({
                'id': node_id,
                'line': constraint.get('line') if isinstance(constraint, dict) else None,
                'is_terminal': node_data.get('is_terminal', False),
                'polarity': constraint.get('polarity') if isinstance(constraint, dict) else None,
                'expr': constraint.get('expr') if isinstance(constraint, dict) else None
            })

    # Group by line
    from collections import defaultdict
    by_line = defaultdict(list)
    for ei in eis:
        if ei['line']:
            by_line[ei['line']].append(ei)

    # Find lines with multiple EIs
    multi_ei_lines = {line: eis for line, eis in by_line.items() if len(eis) > 1}

    # Check which ones are connected
    results = []
    for line, line_eis in multi_ei_lines.items():
        # Check if these EIs are reachable from entry
        entry = f"{callable_id}_E000" + ("0" if FUNC_ID_PATTERN.match(callable_id) else "1")
        reachable = set()
        if cfg.has_node(entry):
            try:
                for ei in line_eis:
                    if nx.has_path(cfg, entry, ei['id']):
                        reachable.add(ei['id'])
            except:
                pass

        results.append({
            'line': line,
            'total_eis': len(line_eis),
            'reachable': len(reachable),
            'unreachable': [ei['id'] for ei in line_eis if ei['id'] not in reachable],
            'eis': line_eis
        })

    return results


def print_callable_detail(detail):
    """Pretty print callable detail."""
    print(f"\n{'=' * 80}")
    print(f"Callable: {detail['callable_id']}")
    print(f"Entry: {detail['entry']}")
    print(f"Exits: {detail['exits']}")
    print(f"\nEIs ({len(detail['eis'])}):")
    for ei in detail['eis']:
        term = f"[TERM:{ei['terminates_via']}]" if ei['is_terminal'] else ""
        print(f"  {ei['id']}")
        print(f"    condition: {ei['condition']}")
        print(f"    outcome: {ei['outcome']} {term}")

    print(f"\nEdges ({len(detail['edges'])}):")
    for edge in detail['edges']:
        internal = "INTERNAL" if edge['is_internal'] else f"EXTERNAL->{edge['target_callable']}"
        print(f"  {edge['from']} --[{edge['edge_type']}]--> {edge['to']} ({internal})")

    if detail.get('external_edges'):
        print(f"\nExternal node edges ({len(detail['external_edges'])}):")
        for edge in detail['external_edges']:
            print(f"  {edge['from']} --[{edge['edge_type']}]--> {edge['to']}")

    print(f"\nIncoming edges to entry ({len(detail['entry_incoming'])}):")
    for edge in detail['entry_incoming']:
        print(f"  {edge['from']} ({edge['from_callable']}) --[{edge['edge_type']}]--> {detail['entry']}")


def print_diagnostic_summary(results):
    """Print a summary of diagnostic results."""
    broken = [r for r in results if r['is_broken']]
    no_returns = [r for r in results if r['returns']['has_exits'] and not r['returns']['all_exits_have_returns']]
    never_called = [r for r in results if r['coverage']['entry_exists'] and not r['coverage']['is_called']]

    print(f"\n=== Diagnostic Summary ===")
    print(f"Total callables: {len(results)}")
    print(f"Broken callables (no internal path): {len(broken)}")
    print(f"Missing return edges: {len(no_returns)}")
    print(f"Never called: {len(never_called)}")

    if broken:
        print(f"\n=== Broken Callables ===")
        for r in broken[:10]:
            print(f"{r['callable_id']}: {r['integrity']['issue']}")

    if no_returns:
        print(f"\n=== Callables Missing Return Edges ===")
        for r in no_returns[:10]:
            exits_without_returns = [e for e in r['returns']['exits'] if not e['has_return_edges']]
            print(f"{r['callable_id']}: {len(exits_without_returns)} exits without returns")

    if never_called:
        print(f"\n=== Callables Never Called ===")
        for r in never_called[:10]:
            print(f"{r['callable_id']}")

    print("\n=== Callables NOT in Both Lists ===")
    no_returns_set = set(
        r['callable_id'] for r in results if r['returns']['has_exits'] and not r['returns']['all_exits_have_returns'])
    never_called_set = set(
        r['callable_id'] for r in results if r['coverage']['entry_exists'] and not r['coverage']['is_called'])
    not_in_both = no_returns_set ^ never_called_set
    print(f"Count: {len(not_in_both)}")
    for cid in sorted(not_in_both):
        if cid in no_returns_set:
            print(f"  {cid}: missing return edges")
        else:
            print(f"  {cid}: never called")


def main():
    if len(sys.argv) < 2:
        print("Usage: diagnose_cfg.py <cfg.pkl> [start_ei] [target_ei]")
        return 1

    cfg_path = Path(sys.argv[1])
    print(f"Loading CFG from {cfg_path}...")
    cfg = load_cfg(cfg_path)
    print(f"  {cfg.number_of_nodes()} nodes, {cfg.number_of_edges()} edges\n")

    # Run full diagnostics
    print("Running diagnostics...")
    results = diagnose_all_callables(cfg)
    print_diagnostic_summary(results)

    # If specific path requested, trace it
    if len(sys.argv) >= 4:
        start = sys.argv[2]
        target = sys.argv[3]

        print(f"\n=== Tracing Path: {start} → {target} ===")
        result = trace_forward(cfg, start, target, max_steps=200)

        if result['reached_target']:
            print(f"✓ Path found! {result['steps']} steps")
        else:
            print(f"✗ Path NOT found")
            print(f"  Dead end at: {result['dead_end']}")
            print(f"  Reason: {result['dead_end_reason']}")

    if len(sys.argv) >= 3 and sys.argv[2].startswith('U'):
        # Specific callable requested
        callable_id = sys.argv[2]
        detail = diagnose_callable_detail(cfg, callable_id)
        print_callable_detail(detail)
        broken = [r['callable_id'] for r in results if r['is_broken']]
        print(f"\n{callable_id} in broken list: {callable_id in broken}")

    return 0


if __name__ == '__main__':
    sys.exit(main())