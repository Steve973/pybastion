#!/usr/bin/env python3
"""
Validate EI node IDs in a pickled NetworkX call graph.

Checks:
1. EI-shaped node IDs match the expected EI regex
2. Optionally, each callable has at least one entry EI candidate (E0000 or E0001)

Usage:
    python validate_graph_ei_ids.py graph.pkl
    python validate_graph_ei_ids.py graph.pkl --check-callable-entries
    python validate_graph_ei_ids.py graph.pkl --fail-on-mismatch
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# Callable IDs:
#   UXXXXXXXXXX_F001
#   UXXXXXXXXXX_F001.F002
#   UXXXXXXXXXX_C001_M001
#   UXXXXXXXXXX_C001.C002_M001
#   UXXXXXXXXXX_C001_M001.F001
#
# EI IDs:
#   <callable_id>_Edddd
#
# No end anchor on purpose.
CALLABLE_ID_PATTERN = re.compile(
    r"^U[0-9A-F]{10}(?:"
    r"_F\d{3}(?:\.F\d{3})*"
    r"|_C\d{3}(?:\.C\d{3})*_M\d{3}(?:\.F\d{3})*"
    r")"
)

EI_ID_PATTERN = re.compile(
    r"^U[0-9A-F]{10}(?:"
    r"_F\d{3}(?:\.F\d{3})*"
    r"|_C\d{3}(?:\.C\d{3})*_M\d{3}(?:\.F\d{3})*"
    r")_E\d{4}"
)

ENTRY_EI_ID_PATTERN = re.compile(
    r"^U[0-9A-F]{10}(?:"
    r"_F\d{3}(?:\.F\d{3})*"
    r"|_C\d{3}(?:\.C\d{3})*_M\d{3}(?:\.F\d{3})*"
    r")_E000[01]"
)


def load_graph(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def is_ei_like(node_id: Any) -> bool:
    return isinstance(node_id, str) and "_E" in node_id and node_id.startswith("U")


def callable_prefix_from_ei(ei_id: str) -> str | None:
    m = re.match(r"^(U[0-9A-F]{10}(?:_F\d{3}(?:\.F\d{3})*|_C\d{3}(?:\.C\d{3})*_M\d{3}(?:\.F\d{3})*))_E\d{4}", ei_id)
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EI IDs in a pickled call graph")
    parser.add_argument("graph", type=Path, help="Path to pickled NetworkX graph")
    parser.add_argument(
        "--check-callable-entries",
        action="store_true",
        help="Also check whether each callable has at least one E0000/E0001 EI in the graph",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero if any EI-shaped node fails validation",
    )
    args = parser.parse_args()

    graph = load_graph(args.graph)

    total_nodes = graph.number_of_nodes()
    ei_like_nodes: list[str] = []
    bad_ei_nodes: list[str] = []
    callable_to_eis: dict[str, list[str]] = defaultdict(list)

    for node_id, data in graph.nodes(data=True):
        if not is_ei_like(node_id):
            continue

        ei_like_nodes.append(node_id)

        if not EI_ID_PATTERN.match(node_id):
            bad_ei_nodes.append(node_id)
            continue

        callable_id = callable_prefix_from_ei(node_id)
        if callable_id:
            callable_to_eis[callable_id].append(node_id)

    print(f"Total graph nodes: {total_nodes}")
    print(f"EI-like nodes inspected: {len(ei_like_nodes)}")
    print(f"EI regex mismatches: {len(bad_ei_nodes)}")

    if bad_ei_nodes:
        print("\nEI regex mismatches:")
        for node_id in sorted(bad_ei_nodes):
            print(f"  {node_id}")

    missing_entry_callables: list[str] = []
    entry_zero_count = 0
    entry_one_count = 0
    if args.check_callable_entries:
        for callable_id, ei_ids in sorted(callable_to_eis.items()):
            has_entry = any(ENTRY_EI_ID_PATTERN.match(ei_id) for ei_id in ei_ids)
            if not has_entry:
                missing_entry_callables.append(callable_id)
            else:
                if ei_ids[0].endswith("_E0000"):
                    entry_zero_count += 1
                elif ei_ids[0].endswith("_E0001"):
                    entry_one_count += 1

        print(f"\nCallables discovered from EI IDs: {len(callable_to_eis)}")
        print(f"Callables missing entry-ish EI (E0000/E0001): {len(missing_entry_callables)}")
        print(f"  E0000 count: {entry_zero_count}")
        print(f"  E0001 count: {entry_one_count}")

        if missing_entry_callables:
            print("\nCallables missing entry-ish EI:")
            for callable_id in missing_entry_callables:
                print(f"  {callable_id}")

    if args.fail_on_mismatch and bad_ei_nodes:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
