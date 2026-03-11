#!/usr/bin/env python3
"""Check if every callable has an E0000 entry node."""

import pickle
import sys
from pathlib import Path
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: check_entries.py <cfg.pkl>")
        return 1

    cfg_path = Path(sys.argv[1])
    print(f"Loading CFG from {cfg_path}...")

    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)

    # Group nodes by callable
    by_callable = defaultdict(list)
    for node_id, node_data in cfg.nodes(data=True):
        callable_id = node_data.get('callable_id')
        if callable_id:
            by_callable[callable_id].append(node_id)

    print(f"\nTotal callables: {len(by_callable)}")

    # Check each callable for E0000
    missing_e0000 = []
    has_e0000 = []

    for callable_id, node_ids in sorted(by_callable.items()):
        entry_e0000 = f"{callable_id}_E0000"

        if entry_e0000 in node_ids:
            has_e0000.append(callable_id)
        else:
            missing_e0000.append(callable_id)

    print(f"Callables with E0000: {len(has_e0000)}")
    print(f"Callables missing E0000: {len(missing_e0000)}")

    if missing_e0000:
        print(f"\nCallables missing E0000 entry node:")
        for callable_id in missing_e0000[:20]:
            nodes = by_callable[callable_id]
            # Show first few nodes to see what it has instead
            first_nodes = sorted(nodes)[:5]
            print(f"  {callable_id}")
            print(f"    First nodes: {first_nodes}")

    # Also check: are there any E0001 being used as entries?
    e0001_entries = []
    for callable_id in by_callable:
        entry_e0001 = f"{callable_id}_E0001"
        entry_e0000 = f"{callable_id}_E0000"

        if entry_e0001 in by_callable[callable_id] and entry_e0000 not in by_callable[callable_id]:
            e0001_entries.append(callable_id)

    if e0001_entries:
        print(f"\nCallables starting at E0001 (no E0000):")
        for callable_id in e0001_entries[:20]:
            print(f"  {callable_id}")

    return 0


if __name__ == '__main__':
    sys.exit(main())