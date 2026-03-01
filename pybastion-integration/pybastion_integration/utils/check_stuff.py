#!/usr/bin/env python3
"""
CFG Path Diagnostics

Analyzes paths in the CFG to identify where paths break and why.
"""

import pickle
import sys
from pathlib import Path

import networkx as nx


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

        if is_terminal and terminates_via == 'return':
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


def check_return_edges(cfg: nx.DiGraph, exit_ei_id: str):
    """Check what return edges exist from an exit EI."""
    edges = []

    for _, target, data in cfg.out_edges(exit_ei_id, data=True):
        if data.get('edge_type') == 'return':
            edges.append({
                'target': target,
                'returns_from': data.get('returns_from'),
                'call_site': data.get('original_call_site')
            })

    return edges


def main():
    if len(sys.argv) < 4:
        print("Usage: diagnose_cfg_path.py <cfg.pkl> <start_ei> <target_ei>")
        return 1

    cfg_path = Path(sys.argv[1])
    start = sys.argv[2]
    target = sys.argv[3]

    print(f"Loading CFG from {cfg_path}...")
    cfg = load_cfg(cfg_path)
    print(f"  {cfg.number_of_nodes()} nodes, {cfg.number_of_edges()} edges\n")

    # Trace forward
    print(f"=== Tracing from {start} to {target} ===")
    result = trace_forward(cfg, start, target, max_steps=200)

    if result['reached_target']:
        print(f"✓ Path found! {result['steps']} steps")
        print(f"  Path length: {len(result['path'])}")
    else:
        print(f"✗ Path NOT found")
        print(f"  Dead end at: {result['dead_end']}")
        print(f"  Reason: {result['dead_end_reason']}")
        if result['dead_end_reason'] == 'no_outgoing_edges':
            print(f"  is_terminal: {result['is_terminal']}")
            print(f"  terminates_via: {result['terminates_via']}")
            print(f"  outcome: {result['outcome']}")
        print(f"  Path so far: {len(result['path'])} steps")

    # Extract callable IDs from start and target
    start_callable = '_'.join(start.split('_')[:-1])
    target_callable = '_'.join(target.split('_')[:-1])

    if start_callable == target_callable:
        print(f"\n=== Analyzing EI sequence in {start_callable} ===")
        start_num = int(start.split('_E')[-1])
        target_num = int(target.split('_E')[-1])

        eis = analyze_ei_sequence(cfg, start_callable, start_num, min(target_num, start_num + 40))

        for ei in eis:
            if not ei['exists']:
                print(f"{ei['ei_id']}: DOES NOT EXIST")
                continue

            term_info = ""
            if ei['is_terminal']:
                term_info = f" [TERMINAL: {ei['terminates_via']}]"

            edges_info = ""
            if ei['outgoing_edges']:
                edge_strs = [f"{e['target']} ({e['edge_type']})" for e in ei['outgoing_edges']]
                edges_info = f" -> {', '.join(edge_strs)}"
            else:
                edges_info = " -> NONE"

            outcome = ei['outcome'][:80] if ei['outcome'] else ''
            print(f"{ei['ei_id']}: {outcome}{term_info}{edges_info}")

    # If we hit a dead end, analyze what happened
    if not result['reached_target'] and result['path']:
        last_ei = result['dead_end']

        # Check if it's a terminal return without return edges
        if result.get('is_terminal') and result.get('terminates_via') == 'return':
            node = cfg.nodes[last_ei]
            callable_id = node.get('callable_id')

            print(f"\n=== Dead end is a terminal return ===")
            print(f"  Callable: {callable_id}")
            print(f"  EI: {last_ei}")

            # Find who called this callable
            print(f"\n  Looking for calls TO this callable's entry...")
            entry_ei = None
            for nid in cfg.nodes():
                nd = cfg.nodes[nid]
                if nd.get('callable_id') == callable_id:
                    entry_ei = nid
                    break

            if entry_ei:
                print(f"  Entry EI: {entry_ei}")
                in_edges = list(cfg.in_edges(entry_ei, data=True))
                call_edges = [e for e in in_edges if e[2].get('edge_type') == 'call']
                print(f"  Call edges TO entry: {len(call_edges)}")
                for src, _, data in call_edges[:5]:
                    print(f"    Called from: {src}")
                    caller_callable = cfg.nodes[src].get('callable_id')
                    print(f"      Caller callable: {caller_callable}")

            # Check return edges FROM this exit
            print(f"\n  Return edges FROM this exit EI:")
            out_edges = list(cfg.out_edges(last_ei, data=True))
            return_edges = [e for e in out_edges if e[2].get('edge_type') == 'return']
            if return_edges:
                for _, tgt, data in return_edges:
                    print(f"    -> {tgt}")
            else:
                print(f"    NONE - this is the bug!")

        # Original call analysis
        edges = list(cfg.out_edges(last_ei, data=True))

        for _, target_node, data in edges:
            if data.get('edge_type') == 'call':
                target_callable = cfg.nodes[target_node].get('callable_id')
                if target_callable:
                    print(f"\n=== Call from {last_ei} to {target_callable} ===")
                    print(f"  Target entry: {target_node}")

                    exits = find_exit_eis(cfg, target_callable)
                    print(f"  Exit EIs: {len(exits)}")
                    for exit_info in exits:
                        print(f"    {exit_info['ei_id']} ({exit_info['type']}): {exit_info['outcome'][:60]}")

                        returns = check_return_edges(cfg, exit_info['ei_id'])
                        if returns:
                            for ret in returns:
                                print(f"      -> returns to {ret['target']}")
                        else:
                            print(f"      -> NO RETURN EDGES")

    broken = []
    checked = set()

    for node_id, node_data in cfg.nodes(data=True):
        cid = node_data.get('callable_id')
        if not cid or cid in checked:
            continue
        checked.add(cid)

        entry = f"{cid}_E0001"
        if not cfg.has_node(entry):
            continue

        exits = []
        for nid, ndata in cfg.nodes(data=True):
            if (ndata.get('callable_id') == cid and
                    ndata.get('is_terminal') and
                    ndata.get('terminates_via') == 'return'):
                exits.append(nid)

        if not exits:
            continue

        has_path = False
        for exit_node in exits:
            try:
                nx.shortest_path(cfg, entry, exit_node)
                has_path = True
                break
            except nx.NetworkXNoPath:
                pass

        if not has_path:
            broken.append(cid)

    print(f"Broken: {len(broken)}")
    for cid in broken[:15]:
        print(f"  {cid}")

    f033_f001_nodes = []
    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get('callable_id') == 'U5DABDB7A65_F033.F001':
            f033_f001_nodes.append(node_id)

    print(f"F033.F001 nodes in graph: {len(f033_f001_nodes)}")
    if f033_f001_nodes:
        print(f"Node IDs: {f033_f001_nodes[:5]}")
    else:
        print("F033.F001 has NO nodes in the graph at all!")

    return 0


if __name__ == '__main__':
    sys.exit(main())