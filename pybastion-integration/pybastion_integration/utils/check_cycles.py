#!/usr/bin/env python3
"""Analyze path structure between start and end nodes."""

import pickle
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx


def main():
    if len(sys.argv) < 2:
        print("Usage: check_path.py <cfg.pkl>")
        return 1

    cfg_path = Path(sys.argv[1])
    print(f"Loading CFG from {cfg_path}...")

    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)

    start = "U37D3513825_C001_M001_E0003"
    end = "U37D3513825_C001_M001_E0035"

    print(f"\nAnalyzing path from {start} to {end}")

    # Get shortest path
    try:
        path = nx.shortest_path(cfg, start, end)
        print(f"\nPath length: {len(path)} nodes")

        # Show first and last nodes
        print(f"\nFirst 20 nodes:")
        for i, node in enumerate(path[:20]):
            node_data = cfg.nodes[node]
            callable_id = node_data.get('callable_id', '')
            outcome = node_data.get('outcome', '')
            print(f"  {i}: {node} ({callable_id}) - {outcome}")

        print(f"\nLast 20 nodes:")
        for i, node in enumerate(path[-20:], start=len(path) - 20):
            node_data = cfg.nodes[node]
            callable_id = node_data.get('callable_id', '')
            outcome = node_data.get('outcome', '')
            print(f"  {i}: {node} ({callable_id}) - {outcome}")

        # Count edge types
        edge_types = defaultdict(int)
        for i in range(len(path) - 1):
            edge_data = cfg.get_edge_data(path[i], path[i + 1])
            edge_type = edge_data.get('edge_type', 'unknown') if edge_data else 'missing'
            edge_types[edge_type] += 1

        print(f"\nEdge types in path:")
        for etype, count in sorted(edge_types.items()):
            print(f"  {etype}: {count}")

        # Call/return balance
        call_count = edge_types.get('call', 0)
        return_count = edge_types.get('return', 0)
        print(f"\nCall/return balance: {call_count} calls, {return_count} returns")
        if call_count != return_count:
            print(f"  WARNING: Imbalance of {abs(call_count - return_count)}")

    except nx.NetworkXNoPath:
        print(f"\n✗ No path exists")

    return 0


if __name__ == '__main__':
    sys.exit(main())