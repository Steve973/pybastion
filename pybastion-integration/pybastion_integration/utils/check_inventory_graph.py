#!/usr/bin/env python3
"""
Diagnostics for inventory-first execution-instance control-flow graphs.

This checker is designed for graphs built by build_full_call_graph_from_inventory.py.
It avoids old EI numbering heuristics and instead relies on graph node metadata.

Checks included:
- callable integrity (entry exists, exits exist, entry can reach an exit)
- return-edge coverage for callable exits
- call coverage (which callables are never called)
- execution-path verification against recorded inventory paths
- detailed per-callable inspection
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

SUCCESS_EXIT_TERMINATORS: set[str] = {"return", "implicit-return"}
EXCEPTION_EXIT_TERMINATORS: set[str] = {"raise", "exception"}
CALLABLE_KINDS: set[str] = {"function", "method", "assignment"}
LOW_SIGNAL_DUNDER_NAMES: set[str] = {
    "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__",
    "__hash__", "__repr__", "__str__", "__bool__", "__len__",
    "__iter__", "__next__", "__contains__",
}



# =============================================================================
# Loading
# =============================================================================


def load_cfg(cfg_path: Path) -> nx.DiGraph:
    with open(cfg_path, "rb") as f:
        return pickle.load(f)



def discover_inventory_files(inventories_root: Path) -> list[Path]:
    inventory_files = list(inventories_root.rglob("*.inventory.yaml"))
    inventory_files.extend(inventories_root.rglob("*_inventory.yaml"))
    return sorted(set(inventory_files))


# =============================================================================
# Inventory indexing for execution-path verification
# =============================================================================


def build_callable_fqn(unit_fqn: str, ancestor_names: list[str], entry_name: str) -> str:
    parts = [unit_fqn, *ancestor_names, entry_name]
    return ".".join(part for part in parts if part)



def index_inventory_execution_paths(inventory_paths: list[Path]) -> dict[str, dict[str, Any]]:
    """
    Returns callable_id -> info containing execution path expectations.
    """
    indexed: dict[str, dict[str, Any]] = {}

    for path in inventory_paths:
        with open(path, "r", encoding="utf-8") as f:
            inventory = yaml.safe_load(f)
        if not inventory:
            continue

        unit_name = inventory["unit"]
        unit_fqn = inventory.get("fully_qualified_name", unit_name)

        def recurse(entries: list[dict[str, Any]], ancestors: list[str]) -> None:
            for entry in entries:
                kind = entry.get("kind", "unknown")
                if kind not in CALLABLE_KINDS:
                    recurse(entry.get("children", []) or [], [*ancestors, entry.get("name", "unknown")])
                    continue

                callable_id = entry["id"]
                callable_name = entry.get("name", "unknown")
                callable_fqn = build_callable_fqn(unit_fqn, ancestors, callable_name)
                branches = entry.get("branches", []) or []
                integration_candidates = ((entry.get("ast_analysis") or {}).get("integration_candidates")) or []

                indexed[callable_id] = {
                    "callable_id": callable_id,
                    "callable_name": callable_name,
                    "callable_fqn": callable_fqn,
                    "unit_name": unit_name,
                    "unit_fqn": unit_fqn,
                    "branches": branches,
                    "integration_candidates": integration_candidates,
                }

                recurse(entry.get("children", []) or [], [*ancestors, callable_name])

        recurse(inventory.get("entries", []) or [], [])

    return indexed


# =============================================================================
# Graph indexing
# =============================================================================


def collect_callables_from_graph(cfg: nx.DiGraph) -> dict[str, dict[str, Any]]:
    """
    Build callable metadata from EI nodes in graph.
    """
    callables: dict[str, dict[str, Any]] = {}

    for node_id, node_data in cfg.nodes(data=True):
        callable_id = node_data.get("callable_id")
        if not callable_id:
            continue

        entry = callables.setdefault(
            callable_id,
            {
                "callable_id": callable_id,
                "callable_name": node_data.get("callable_name", ""),
                "callable_kind": node_data.get("callable_kind", ""),
                "callable_fqn": node_data.get("callable_fqn", ""),
                "unit": node_data.get("unit", ""),
                "unit_fqn": node_data.get("unit_fqn", ""),
                "ei_ids": [],
            },
        )
        entry["ei_ids"].append(node_id)

    for entry in callables.values():
        entry["ei_ids"].sort(key=_ei_sort_key)

    return callables



def _ei_sort_key(ei_id: str) -> int:
    if "_E" not in ei_id:
        return 0
    try:
        return int(ei_id.rsplit("_E", 1)[1])
    except ValueError:
        return 0



def find_entry_ei(cfg: nx.DiGraph, callable_id: str) -> str | None:
    eis = [node_id for node_id, data in cfg.nodes(data=True) if data.get("callable_id") == callable_id]
    if not eis:
        return None
    return sorted(eis, key=_ei_sort_key)[0]



def find_success_exit_eis(cfg: nx.DiGraph, callable_id: str) -> list[str]:
    exits: list[str] = []
    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get("callable_id") != callable_id:
            continue
        if node_data.get("is_terminal") and node_data.get("terminates_via") in SUCCESS_EXIT_TERMINATORS:
            exits.append(node_id)
    return sorted(exits, key=_ei_sort_key)



def find_exception_exit_eis(cfg: nx.DiGraph, callable_id: str) -> list[str]:
    exits: list[str] = []
    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get("callable_id") != callable_id:
            continue
        if node_data.get("is_terminal") and node_data.get("terminates_via") in EXCEPTION_EXIT_TERMINATORS:
            exits.append(node_id)
    return sorted(exits, key=_ei_sort_key)


# =============================================================================
# Diagnostics
# =============================================================================


def check_callable_integrity(cfg: nx.DiGraph, callable_id: str) -> dict[str, Any]:
    entry = find_entry_ei(cfg, callable_id)
    if not entry:
        return {
            "callable_id": callable_id,
            "valid": False,
            "issue": "no_entry_ei",
        }

    success_exits = find_success_exit_eis(cfg, callable_id)
    exception_exits = find_exception_exit_eis(cfg, callable_id)
    all_exits = success_exits + exception_exits

    if not all_exits:
        return {
            "callable_id": callable_id,
            "valid": False,
            "issue": "no_exit_eis",
            "entry": entry,
        }

    reachable_exit = None
    for exit_ei in all_exits:
        try:
            nx.shortest_path(cfg, entry, exit_ei)
            reachable_exit = exit_ei
            break
        except nx.NetworkXNoPath:
            continue

    if not reachable_exit:
        return {
            "callable_id": callable_id,
            "valid": False,
            "issue": "no_path_from_entry_to_exit",
            "entry": entry,
            "success_exits": success_exits,
            "exception_exits": exception_exits,
        }

    return {
        "callable_id": callable_id,
        "valid": True,
        "entry": entry,
        "success_exits": success_exits,
        "exception_exits": exception_exits,
        "reachable_exit": reachable_exit,
    }



def check_return_edges(cfg: nx.DiGraph, callable_id: str) -> dict[str, Any]:
    success_exits = find_success_exit_eis(cfg, callable_id)
    exception_exits = find_exception_exit_eis(cfg, callable_id)

    exit_info: list[dict[str, Any]] = []
    for exit_ei in success_exits + exception_exits:
        out_edges = list(cfg.out_edges(exit_ei, data=True))
        return_edges = [edge for edge in out_edges if edge[2].get("edge_type") == "return"]
        exit_info.append(
            {
                "exit_ei": exit_ei,
                "exit_kind": "success" if exit_ei in success_exits else "exception",
                "has_return_edges": bool(return_edges),
                "return_count": len(return_edges),
                "returns_to": [target for _, target, _ in return_edges],
            }
        )

    return {
        "callable_id": callable_id,
        "has_exits": bool(exit_info),
        "exits": exit_info,
        "all_exits_have_returns": all(item["has_return_edges"] for item in exit_info) if exit_info else False,
    }



def is_low_signal_callable(callable_info: dict[str, Any]) -> bool:
    name = (callable_info.get("callable_name") or "").strip()
    decorators = callable_info.get("decorators") or []

    if name in LOW_SIGNAL_DUNDER_NAMES:
        return True

    for decorator in decorators:
        if isinstance(decorator, dict) and decorator.get("name") == "property":
            return True

    return False


def check_call_coverage(cfg: nx.DiGraph, callable_id: str) -> dict[str, Any]:
    entry = find_entry_ei(cfg, callable_id)
    if not entry:
        return {
            "callable_id": callable_id,
            "entry_exists": False,
        }

    in_edges = list(cfg.in_edges(entry, data=True))
    call_edges = [edge for edge in in_edges if edge[2].get("edge_type") == "call"]

    callers: list[dict[str, Any]] = []
    for src, _, edge_data in call_edges:
        caller_node = cfg.nodes[src]
        callers.append(
            {
                "call_site": src,
                "caller_callable_id": caller_node.get("callable_id"),
                "caller_callable_fqn": caller_node.get("callable_fqn"),
                "call_kind": edge_data.get("call_kind"),
            }
        )

    return {
        "callable_id": callable_id,
        "entry_exists": True,
        "is_called": bool(call_edges),
        "call_count": len(call_edges),
        "callers": callers,
    }



def path_exists_exactly(cfg: nx.DiGraph, path: list[str]) -> bool:
    if not path:
        return False
    for idx in range(len(path) - 1):
        if not cfg.has_edge(path[idx], path[idx + 1]):
            return False
    return True



def check_execution_paths(cfg: nx.DiGraph, inventory_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0

    for callable_id, info in inventory_index.items():
        for candidate in info.get("integration_candidates", []) or []:
            for path in candidate.get("execution_paths", []) or []:
                checked += 1
                if not path_exists_exactly(cfg, path):
                    failures.append(
                        {
                            "callable_id": callable_id,
                            "callable_fqn": info.get("callable_fqn"),
                            "integration_id": candidate.get("id"),
                            "ei_id": candidate.get("ei_id"),
                            "target": candidate.get("target"),
                            "path": path,
                        }
                    )

    return {
        "checked_paths": checked,
        "failure_count": len(failures),
        "failures": failures,
        "all_paths_valid": not failures,
    }



def diagnose_all_callables(cfg: nx.DiGraph) -> list[dict[str, Any]]:
    callables = collect_callables_from_graph(cfg)
    results: list[dict[str, Any]] = []

    for callable_id in sorted(callables):
        callable_info = callables[callable_id]
        integrity = check_callable_integrity(cfg, callable_id)
        returns = check_return_edges(cfg, callable_id)
        coverage = check_call_coverage(cfg, callable_id)
        low_signal = is_low_signal_callable(callable_info)
        results.append(
            {
                "callable_id": callable_id,
                "callable_name": callable_info.get("callable_name"),
                "callable_fqn": callable_info.get("callable_fqn"),
                "callable_kind": callable_info.get("callable_kind"),
                "low_signal": low_signal,
                "integrity": integrity,
                "returns": returns,
                "coverage": coverage,
                "is_broken": not integrity["valid"],
            }
        )

    return results



def analyze_broken_callable(cfg: nx.DiGraph, callable_id: str) -> dict[str, Any]:
    entry = find_entry_ei(cfg, callable_id)
    success_exits = find_success_exit_eis(cfg, callable_id)
    exception_exits = find_exception_exit_eis(cfg, callable_id)

    nodes = []
    reachable_from_entry: set[str] = set()
    if entry and cfg.has_node(entry):
        reachable_from_entry = nx.descendants(cfg, entry) | {entry}

    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get("callable_id") != callable_id:
            continue
        outgoing = []
        for _, tgt, edge_data in cfg.out_edges(node_id, data=True):
            outgoing.append(
                {
                    "target": tgt,
                    "edge_type": edge_data.get("edge_type"),
                    "call_kind": edge_data.get("call_kind"),
                    "return_kind": edge_data.get("return_kind"),
                    "target_callable_id": cfg.nodes[tgt].get("callable_id"),
                }
            )
        nodes.append(
            {
                "id": node_id,
                "line": node_data.get("line"),
                "stmt_type": node_data.get("stmt_type"),
                "condition": node_data.get("condition"),
                "description": node_data.get("description"),
                "is_terminal": node_data.get("is_terminal", False),
                "terminates_via": node_data.get("terminates_via"),
                "reachable_from_entry": node_id in reachable_from_entry,
                "outgoing": outgoing,
            }
        )

    nodes.sort(key=lambda item: _ei_sort_key(item["id"]))

    return {
        "callable_id": callable_id,
        "entry": entry,
        "success_exits": success_exits,
        "exception_exits": exception_exits,
        "reachable_node_count": len(reachable_from_entry),
        "total_node_count": len(nodes),
        "nodes": nodes,
    }


def diagnose_callable_detail(cfg: nx.DiGraph, callable_id: str) -> dict[str, Any]:
    callable_nodes = []
    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get("callable_id") == callable_id:
            callable_nodes.append(
                {
                    "id": node_id,
                    "line": node_data.get("line"),
                    "stmt_type": node_data.get("stmt_type"),
                    "condition": node_data.get("condition"),
                    "description": node_data.get("description"),
                    "is_terminal": node_data.get("is_terminal", False),
                    "terminates_via": node_data.get("terminates_via"),
                    "constraint": node_data.get("constraint"),
                    "owner_info": node_data.get("owner_info"),
                }
            )
    callable_nodes.sort(key=lambda item: _ei_sort_key(item["id"]))

    entry = find_entry_ei(cfg, callable_id)
    success_exits = find_success_exit_eis(cfg, callable_id)
    exception_exits = find_exception_exit_eis(cfg, callable_id)

    edges: list[dict[str, Any]] = []
    for node in callable_nodes:
        for _, target, edge_data in cfg.out_edges(node["id"], data=True):
            target_data = cfg.nodes[target]
            edges.append(
                {
                    "from": node["id"],
                    "to": target,
                    "edge_type": edge_data.get("edge_type"),
                    "call_kind": edge_data.get("call_kind"),
                    "return_kind": edge_data.get("return_kind"),
                    "target_callable_id": target_data.get("callable_id"),
                    "target_callable_fqn": target_data.get("callable_fqn"),
                    "target_category": target_data.get("category"),
                }
            )

    entry_incoming: list[dict[str, Any]] = []
    if entry:
        for src, _, edge_data in cfg.in_edges(entry, data=True):
            src_data = cfg.nodes[src]
            entry_incoming.append(
                {
                    "from": src,
                    "from_callable_id": src_data.get("callable_id"),
                    "from_callable_fqn": src_data.get("callable_fqn"),
                    "edge_type": edge_data.get("edge_type"),
                    "call_kind": edge_data.get("call_kind"),
                }
            )

    return {
        "callable_id": callable_id,
        "entry": entry,
        "success_exits": success_exits,
        "exception_exits": exception_exits,
        "entry_incoming": entry_incoming,
        "eis": callable_nodes,
        "edges": edges,
    }


# =============================================================================
# Output helpers
# =============================================================================


def print_diagnostic_summary(results: list[dict[str, Any]], path_check: dict[str, Any]) -> None:
    broken = [r for r in results if r["is_broken"] and not r["low_signal"]]
    broken_low_signal = [r for r in results if r["is_broken"] and r["low_signal"]]
    no_returns = [
        r for r in results
        if r["returns"]["has_exits"]
        and not r["returns"]["all_exits_have_returns"]
        and r["coverage"].get("is_called")
        and not r["low_signal"]
    ]
    never_called = [r for r in results if r["coverage"]["entry_exists"] and not r["coverage"]["is_called"] and not r["low_signal"]]
    never_called_low_signal = [r for r in results if r["coverage"]["entry_exists"] and not r["coverage"]["is_called"] and r["low_signal"]]

    print("\n=== Diagnostic Summary ===")
    print(f"Total callables: {len(results)}")
    print(f"Broken callables (no internal path / no exits, non-low-signal): {len(broken)}")
    print(f"Broken callables (low-signal bucket): {len(broken_low_signal)}")
    print(f"Callables missing return edges (called, non-low-signal): {len(no_returns)}")
    print(f"Callables never called (non-low-signal): {len(never_called)}")
    print(f"Callables never called (low-signal bucket): {len(never_called_low_signal)}")
    print(f"Recorded execution paths checked: {path_check['checked_paths']}")
    print(f"Execution path failures: {path_check['failure_count']}")

    if broken:
        print("\n=== Broken Callables ===")
        for item in broken[:10]:
            print(f"{item['callable_id']} ({item.get('callable_fqn', '')}): {item['integrity']['issue']}")

    if no_returns:
        print("\n=== Callables Missing Return Edges ===")
        for item in no_returns[:10]:
            exits_without_returns = [e for e in item["returns"]["exits"] if not e["has_return_edges"]]
            print(f"{item['callable_id']} ({item.get('callable_fqn', '')}): {len(exits_without_returns)} exits without returns")

    if never_called:
        print("\n=== Callables Never Called ===")
        for item in never_called[:10]:
            print(f"{item['callable_id']} ({item.get('callable_fqn', '')})")

    if never_called_low_signal:
        print("\n=== Callables Never Called (Low-Signal Bucket) ===")
        for item in never_called_low_signal[:10]:
            print(f"{item['callable_id']} ({item.get('callable_fqn', '')})")

    if path_check["failures"]:
        print("\n=== Execution Path Failures ===")
        for failure in path_check["failures"][:10]:
            print(
                f"{failure['callable_id']} ({failure.get('callable_fqn', '')}) "
                f"integration={failure.get('integration_id')} ei={failure.get('ei_id')}"
            )



def print_callable_detail(detail: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"Callable: {detail['callable_id']}")
    print(f"Entry: {detail['entry']}")
    print(f"Success exits: {detail['success_exits']}")
    print(f"Exception exits: {detail['exception_exits']}")

    print(f"\nEIs ({len(detail['eis'])}):")
    for ei in detail["eis"]:
        terminal = f" [TERM:{ei['terminates_via']}]" if ei["is_terminal"] else ""
        print(f"  {ei['id']} L{ei.get('line')} {ei.get('stmt_type')}{terminal}")
        print(f"    condition: {ei.get('condition')}")
        print(f"    description: {ei.get('description')}")
        print(f"    owner_info: {ei.get('owner_info')}")
        print(f"    constraint: {ei.get('constraint')}")

    print(f"\nEdges ({len(detail['edges'])}):")
    for edge in detail["edges"]:
        print(
            f"  {edge['from']} --[{edge['edge_type']}, call={edge.get('call_kind')}, return={edge.get('return_kind')}]--> "
            f"{edge['to']}"
        )
        print(
            f"    target_callable_id={edge.get('target_callable_id')} "
            f"target_callable_fqn={edge.get('target_callable_fqn')} target_category={edge.get('target_category')}"
        )

    print(f"\nIncoming edges to entry ({len(detail['entry_incoming'])}):")
    for edge in detail["entry_incoming"]:
        print(
            f"  {edge['from']} ({edge.get('from_callable_id')} / {edge.get('from_callable_fqn')}) "
            f"--[{edge['edge_type']}, call={edge.get('call_kind')}]--> {detail['entry']}"
        )


# =============================================================================
# Main
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnostics for inventory-first EI graphs")
    parser.add_argument("cfg", type=Path, help="Path to graph pickle")
    parser.add_argument("--inventories-root", type=Path, required=True, help="Root containing *.inventory.yaml files")
    parser.add_argument("--callable-id", type=str, help="Show detailed diagnostics for one callable ID")
    parser.add_argument("--broken-callable-id", type=str, help="Show focused broken-callable analysis for one callable ID")
    parser.add_argument("--write-report", type=Path, help="Write YAML report to this path")
    args = parser.parse_args(argv)

    print(f"Loading CFG from {args.cfg}...")
    cfg = load_cfg(args.cfg)
    print(f"  {cfg.number_of_nodes()} nodes, {cfg.number_of_edges()} edges")

    if not args.inventories_root.exists():
        print(f"ERROR: inventories root not found: {args.inventories_root}", file=sys.stderr)
        return 1

    inventory_paths = discover_inventory_files(args.inventories_root)
    if not inventory_paths:
        print(f"ERROR: no inventory files found under {args.inventories_root}", file=sys.stderr)
        return 1

    inventory_index = index_inventory_execution_paths(inventory_paths)

    print("Running diagnostics...")
    results = diagnose_all_callables(cfg)
    path_check = check_execution_paths(cfg, inventory_index)
    print_diagnostic_summary(results, path_check)

    if args.callable_id:
        detail = diagnose_callable_detail(cfg, args.callable_id)
        print_callable_detail(detail)

    if args.broken_callable_id:
        broken_detail = analyze_broken_callable(cfg, args.broken_callable_id)
        print("\n=== Broken Callable Analysis ===")
        print(yaml.dump(broken_detail, sort_keys=False, allow_unicode=True, width=float("inf")))

    if args.write_report:
        report = {
            "summary": {
                "total_callables": len(results),
                "broken_callables": len([r for r in results if r["is_broken"]]),
                "execution_paths_checked": path_check["checked_paths"],
                "execution_path_failures": path_check["failure_count"],
            },
            "results": results,
            "execution_path_check": path_check,
        }
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_report, "w", encoding="utf-8") as f:
            yaml.dump(report, f, sort_keys=False, allow_unicode=True, width=float("inf"))
        print(f"Wrote report to {args.write_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
