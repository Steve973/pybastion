#!/usr/bin/env python3
"""
Diagnostics for inventory-first execution-instance control-flow graphs.

This checker is designed for graphs built by build_full_call_graph_from_inventory.py.
It avoids old EI numbering heuristics and instead relies on graph node metadata.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from pybastion_integration.stages.build_full_call_graph_from_inventory import (
    analysis_info,
    hierarchy_info,
    signature_info,
)

SUCCESS_EXIT_TERMINATORS: set[str] = {
    "return",
    "implicit-return",
}

EXCEPTION_EXIT_TERMINATORS: set[str] = {
    "raise",
    "exception",
}

CALLABLE_KINDS: set[str] = {
    "function",
    "method",
}

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


def is_ei_like(node_id: Any) -> bool:
    return isinstance(node_id, str) and "_E" in node_id and node_id.startswith("U")


def callable_prefix_from_ei(ei_id: str) -> str | None:
    m = re.match(
        r"^(U[0-9A-F]{10}(?:_F\d{3}(?:\.F\d{3})*|_C\d{3}(?:\.C\d{3})*_M\d{3}(?:\.F\d{3})*))_E\d{4}",
        ei_id,
    )
    return m.group(1) if m else None


def validate_ei_ids(cfg: nx.DiGraph) -> dict[str, Any]:
    ei_like_nodes: list[str] = []
    bad_ei_nodes: list[str] = []
    callable_to_eis: dict[str, list[str]] = {}

    for node_id, _ in cfg.nodes(data=True):
        if not is_ei_like(node_id):
            continue

        ei_like_nodes.append(node_id)

        if not EI_ID_PATTERN.match(node_id):
            bad_ei_nodes.append(node_id)
            continue

        callable_id = callable_prefix_from_ei(node_id)
        if callable_id is not None:
            callable_to_eis.setdefault(callable_id, []).append(node_id)

    missing_entryish_callables: list[str] = []
    entry_e0000_count = 0
    entry_e0001_count = 0

    for callable_id, ei_ids in sorted(callable_to_eis.items()):
        has_entryish = any(ENTRY_EI_ID_PATTERN.match(ei_id) for ei_id in ei_ids)
        if not has_entryish:
            missing_entryish_callables.append(callable_id)

        sorted_eis = sorted(ei_ids, key=_ei_sort_key)
        if sorted_eis:
            first = sorted_eis[0]
            if first.endswith("_E0000"):
                entry_e0000_count += 1
            elif first.endswith("_E0001"):
                entry_e0001_count += 1

    return {
        "total_graph_nodes": cfg.number_of_nodes(),
        "ei_like_nodes_inspected": len(ei_like_nodes),
        "ei_regex_mismatches": sorted(bad_ei_nodes),
        "ei_regex_mismatch_count": len(bad_ei_nodes),
        "callables_discovered_from_eis": len(callable_to_eis),
        "callables_missing_entryish_ei": missing_entryish_callables,
        "callables_missing_entryish_ei_count": len(missing_entryish_callables),
        "entry_e0000_count": entry_e0000_count,
        "entry_e0001_count": entry_e0001_count,
    }


def analyze_collapsed_node_health(cfg: nx.DiGraph) -> dict[str, Any]:
    collapsed_without_return: list[dict[str, Any]] = []
    collapsed_without_incoming_call: list[dict[str, Any]] = []
    collapsed_missing_target_callable: list[dict[str, Any]] = []

    for node_id, data in cfg.nodes(data=True):
        if data.get("category") != "collapsed_internal_operation":
            continue

        incoming_calls = [
            (src, edge_data)
            for src, _, edge_data in cfg.in_edges(node_id, data=True)
            if edge_data.get("edge_type") == "call"
        ]
        outgoing_returns = [
            (dst, edge_data)
            for _, dst, edge_data in cfg.out_edges(node_id, data=True)
            if edge_data.get("edge_type") == "return"
        ]

        target_callable_id = data.get("target_callable_id") or data.get("callable_id")
        if not target_callable_id:
            collapsed_missing_target_callable.append(
                {
                    "node_id": node_id,
                    "callable_fqn": data.get("callable_fqn"),
                }
            )

        if not incoming_calls:
            collapsed_without_incoming_call.append(
                {
                    "node_id": node_id,
                    "callable_fqn": data.get("callable_fqn"),
                }
            )

        if not outgoing_returns:
            collapsed_without_return.append(
                {
                    "node_id": node_id,
                    "callable_fqn": data.get("callable_fqn"),
                }
            )

    return {
        "collapsed_without_return_count": len(collapsed_without_return),
        "collapsed_without_return": collapsed_without_return,
        "collapsed_without_incoming_call_count": len(collapsed_without_incoming_call),
        "collapsed_without_incoming_call": collapsed_without_incoming_call,
        "collapsed_missing_target_callable_count": len(collapsed_missing_target_callable),
        "collapsed_missing_target_callable": collapsed_missing_target_callable,
    }


def analyze_edge_target_health(cfg: nx.DiGraph) -> dict[str, Any]:
    bad_call_targets: list[dict[str, Any]] = []
    bad_return_targets: list[dict[str, Any]] = []
    self_loop_call_edges: list[dict[str, Any]] = []

    for src, dst, edge_data in cfg.edges(data=True):
        edge_type = edge_data.get("edge_type")
        src_data = cfg.nodes[src]
        dst_data = cfg.nodes[dst]

        if edge_type == "call":
            dst_category = dst_data.get("category")
            if dst_category not in {"execution_instance", "collapsed_internal_operation"}:
                bad_call_targets.append(
                    {
                        "from": src,
                        "to": dst,
                        "from_callable_fqn": src_data.get("callable_fqn"),
                        "to_category": dst_category,
                        "to_callable_fqn": dst_data.get("callable_fqn"),
                    }
                )

            if src_data.get("callable_id") == dst_data.get("callable_id"):
                self_loop_call_edges.append(
                    {
                        "from": src,
                        "to": dst,
                        "callable_fqn": src_data.get("callable_fqn"),
                    }
                )

        if edge_type == "return":
            dst_category = dst_data.get("category")
            if dst_category != "execution_instance":
                bad_return_targets.append(
                    {
                        "from": src,
                        "to": dst,
                        "from_callable_fqn": src_data.get("callable_fqn"),
                        "to_category": dst_category,
                        "to_callable_fqn": dst_data.get("callable_fqn"),
                    }
                )

    return {
        "bad_call_targets_count": len(bad_call_targets),
        "bad_call_targets": bad_call_targets,
        "bad_return_targets_count": len(bad_return_targets),
        "bad_return_targets": bad_return_targets,
        "self_loop_call_edges_count": len(self_loop_call_edges),
        "self_loop_call_edges": self_loop_call_edges,
    }


def build_callable_call_graph(cfg: nx.DiGraph) -> nx.DiGraph:
    call_graph = nx.DiGraph()

    for _, node_data in cfg.nodes(data=True):
        callable_id = node_data.get("callable_id")
        callable_fqn = node_data.get("callable_fqn")
        if callable_id:
            if not call_graph.has_node(callable_id):
                call_graph.add_node(
                    callable_id,
                    callable_fqn=callable_fqn,
                )

    for src, dst, edge_data in cfg.edges(data=True):
        if edge_data.get("edge_type") != "call":
            continue

        src_callable_id = cfg.nodes[src].get("callable_id")
        dst_callable_id = cfg.nodes[dst].get("callable_id")
        if not src_callable_id or not dst_callable_id:
            continue

        if call_graph.has_edge(src_callable_id, dst_callable_id):
            call_graph[src_callable_id][dst_callable_id]["count"] += 1
        else:
            call_graph.add_edge(src_callable_id, dst_callable_id, count=1)

    return call_graph


def analyze_callable_call_cycles(cfg: nx.DiGraph) -> dict[str, Any]:
    call_graph = build_callable_call_graph(cfg)

    self_recursive: list[dict[str, Any]] = []
    for node_id in sorted(call_graph.nodes()):
        if call_graph.has_edge(node_id, node_id):
            self_recursive.append(
                {
                    "callable_id": node_id,
                    "callable_fqn": call_graph.nodes[node_id].get("callable_fqn"),
                    "call_count": call_graph[node_id][node_id].get("count", 0),
                }
            )

    sccs = list(nx.strongly_connected_components(call_graph))
    multi_node_sccs: list[dict[str, Any]] = []

    for component in sccs:
        if len(component) <= 1:
            continue

        sorted_ids = sorted(component)
        members = [
            {
                "callable_id": callable_id,
                "callable_fqn": call_graph.nodes[callable_id].get("callable_fqn"),
            }
            for callable_id in sorted_ids
        ]

        internal_edges: list[dict[str, Any]] = []
        for src in sorted_ids:
            for dst in sorted_ids:
                if call_graph.has_edge(src, dst):
                    internal_edges.append(
                        {
                            "from_callable_id": src,
                            "to_callable_id": dst,
                            "from_callable_fqn": call_graph.nodes[src].get("callable_fqn"),
                            "to_callable_fqn": call_graph.nodes[dst].get("callable_fqn"),
                            "count": call_graph[src][dst].get("count", 0),
                        }
                    )

        multi_node_sccs.append(
            {
                "size": len(component),
                "members": members,
                "internal_edges": internal_edges,
            }
        )

    multi_node_sccs.sort(key=lambda item: (-item["size"], item["members"][0]["callable_fqn"] or ""))

    return {
        "callable_count": call_graph.number_of_nodes(),
        "call_edge_count": call_graph.number_of_edges(),
        "self_recursive_count": len(self_recursive),
        "self_recursive": self_recursive,
        "multi_node_scc_count": len(multi_node_sccs),
        "multi_node_sccs": multi_node_sccs,
    }


def analyze_external_seams(results: list[dict[str, Any]]) -> dict[str, Any]:
    external_seams: list[dict[str, Any]] = []

    for result in results:
        if not result.get("externally_reachable", False):
            continue

        coverage = result.get("coverage") or {}
        callers = coverage.get("callers", []) or []
        internal_callers = [
            caller for caller in callers
            if caller.get("caller_callable_id")
               and caller.get("caller_callable_id") != result.get("callable_id")
        ]

        external_seams.append(
            {
                "callable_id": result.get("callable_id"),
                "callable_fqn": result.get("callable_fqn"),
                "is_called": bool(coverage.get("is_called", False)),
                "internal_caller_count": len(internal_callers),
            }
        )

    external_seams.sort(key=lambda item: item.get("callable_fqn", "") or "")

    return {
        "count": len(external_seams),
        "full_list": external_seams,
    }


@dataclass(frozen=True)
class CheckerConfig:
    external_method_decorators: set[str]
    low_signal_decorators: set[str]
    collapsible_operation_decorators: set[str]
    low_signal_dunder_names: set[str]
    sections: list[dict[str, Any]]


def load_checker_config(config_path: Path) -> CheckerConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    marker_sets = raw.get("marker_sets", {}) or {}
    name_sets = raw.get("name_sets", {}) or {}

    return CheckerConfig(
        external_method_decorators=set(
            marker_sets.get("external_method_decorators", [])
        ),
        low_signal_decorators=set(marker_sets.get("low_signal_decorators", [])),
        collapsible_operation_decorators=set(
            marker_sets.get("collapsible_operation_decorators", [])
        ),
        low_signal_dunder_names=set(name_sets.get("low_signal_dunder_names", [])),
        sections=raw.get("sections", []) or [],
    )


def callable_marker_names(cfg: nx.DiGraph, callable_id: str) -> set[str]:
    names: set[str] = set()

    for _, data in cfg.nodes(data=True):
        if data.get("callable_id") != callable_id:
            continue

        decorators = data.get("callable_decorators")
        if decorators is None:
            decorators = data.get("decorators", [])

        for dec in decorators or []:
            name = (dec or {}).get("name")
            if name:
                names.add(str(name))

    return names


def externally_reachable_via_marker(
        cfg: nx.DiGraph,
        callable_id: str,
        config: CheckerConfig,
) -> bool:
    if not callable_id:
        return False
    return bool(callable_marker_names(cfg, callable_id) & config.external_method_decorators)


def load_cfg(cfg_path: Path) -> nx.DiGraph:
    with open(cfg_path, "rb") as f:
        return pickle.load(f)


def discover_inventory_files(inventories_root: Path) -> list[Path]:
    inventory_files = list(inventories_root.rglob("*.inventory.yaml"))
    inventory_files.extend(inventories_root.rglob("*_inventory.yaml"))
    return sorted(set(inventory_files))


def build_callable_fqn(unit_fqn: str, ancestor_names: list[str], entry_name: str) -> str:
    parts = [unit_fqn, *ancestor_names, entry_name]
    return ".".join(part for part in parts if part)


def index_inventory_execution_paths(inventory_paths: list[Path]) -> dict[str, dict[str, Any]]:
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
                name = entry.get("name", "unknown")
                children = entry.get("children", []) or []

                if kind not in CALLABLE_KINDS:
                    recurse(children, [*ancestors, name])
                    continue

                callable_id = entry["id"]
                callable_fqn = build_callable_fqn(unit_fqn, ancestors, name)
                ainfo = analysis_info(entry)
                sinfo = signature_info(entry)
                hinfo = hierarchy_info(entry)

                indexed[callable_id] = {
                    "callable_id": callable_id,
                    "callable_name": name,
                    "callable_fqn": callable_fqn,
                    "unit_name": unit_name,
                    "unit_fqn": unit_fqn,
                    "branches": ainfo.get("branches", []) or [],
                    "integration_candidates": ainfo.get("integration_candidates", []) or [],
                    "decorators": sinfo.get("decorators", []) or [],
                    "is_executable": entry.get("is_executable", True),
                    "is_contract_method": hinfo.get("is_contract_method", False),
                }

                recurse(children, [*ancestors, name])

        recurse(inventory.get("entries", []) or [], [])

    return indexed


def collect_callables_from_graph(cfg: nx.DiGraph) -> dict[str, dict[str, Any]]:
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
                "decorators": node_data.get("decorators", []) or [],
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
    eis = [
        node_id
        for node_id, data in cfg.nodes(data=True)
        if data.get("callable_id") == callable_id
    ]
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


def has_incoming_callable_call_edges(cfg: nx.DiGraph, callable_id: str) -> bool:
    entry = find_entry_ei(cfg, callable_id)
    if not entry:
        return False

    for src, _, edge_data in cfg.in_edges(entry, data=True):
        if edge_data.get("edge_type") != "call":
            continue

        src_callable_id = cfg.nodes[src].get("callable_id")
        if src_callable_id and src_callable_id != callable_id:
            return True

    return False


def find_first_failed_recorded_hop(
        cfg: nx.DiGraph,
        start_node: str,
        target_node: str,
        recorded_path: list[str],
) -> dict[str, Any] | None:
    if not recorded_path:
        return {"from": start_node, "to": target_node, "reason": "empty_recorded_path"}

    if recorded_path[0] != start_node and not nx.has_path(cfg, start_node, recorded_path[0]):
        return {
            "from": start_node,
            "to": recorded_path[0],
            "reason": "entry_cannot_reach_first_recorded_ei",
        }

    for current, nxt in zip(recorded_path, recorded_path[1:]):
        if not nx.has_path(cfg, current, nxt):
            return {"from": current, "to": nxt, "reason": "recorded_hop_not_reachable"}

    if recorded_path[-1] != target_node and not nx.has_path(cfg, recorded_path[-1], target_node):
        return {
            "from": recorded_path[-1],
            "to": target_node,
            "reason": "last_recorded_ei_cannot_reach_target",
        }

    return None


def path_contains_recorded_execution_path(
        cfg: nx.DiGraph,
        start_node: str,
        target_node: str,
        recorded_path: list[str],
) -> bool:
    if not recorded_path:
        return False

    if recorded_path[0] != start_node and not nx.has_path(cfg, start_node, recorded_path[0]):
        return False

    for current, nxt in zip(recorded_path, recorded_path[1:]):
        if not nx.has_path(cfg, current, nxt):
            return False

    if recorded_path[-1] != target_node and not nx.has_path(cfg, recorded_path[-1], target_node):
        return False

    return True


def check_callable_integrity(cfg: nx.DiGraph, callable_id: str) -> dict[str, Any]:
    entry = find_entry_ei(cfg, callable_id)
    if not entry:
        return {"callable_id": callable_id, "valid": False, "issue": "no_entry_ei"}

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


def decorator_names_for_callable(
        callable_info: dict[str, Any],
        inventory_info: dict[str, Any] | None = None,
) -> set[str]:
    decorators = callable_info.get("decorators") or []
    if inventory_info:
        inventory_decorators = inventory_info.get("decorators") or []
        if inventory_decorators:
            decorators = inventory_decorators

    names: set[str] = set()
    for decorator in decorators:
        if isinstance(decorator, dict):
            name = decorator.get("name")
            if name:
                names.add(str(name))
    return names


def is_collapsible_operation_callable(
        callable_info: dict[str, Any],
        inventory_info: dict[str, Any] | None,
        config: CheckerConfig,
) -> bool:
    return bool(
        decorator_names_for_callable(callable_info, inventory_info)
        & config.collapsible_operation_decorators
    )


def low_signal_category(
        callable_info: dict[str, Any],
        inventory_info: dict[str, Any] | None,
        config: CheckerConfig,
) -> str | None:
    name = (callable_info.get("callable_name") or "").strip()
    if name in config.low_signal_dunder_names:
        return "Dunder"

    decorator_names = decorator_names_for_callable(callable_info, inventory_info)

    matching_low_signal = sorted(decorator_names & config.low_signal_decorators)
    if matching_low_signal:
        return f"Decorator: {matching_low_signal[0]}"

    return None


def check_return_edges(
        cfg: nx.DiGraph,
        callable_id: str,
        inventory_info: dict[str, Any] | None,
        callable_info: dict[str, Any],
        config: CheckerConfig,
) -> dict[str, Any]:
    collapsible = is_collapsible_operation_callable(callable_info, inventory_info, config)

    if collapsible:
        collapsed_nodes: list[str] = []
        for node_id, node_data in cfg.nodes(data=True):
            if node_data.get("category") != "collapsed_internal_operation":
                continue

            if node_data.get("target_callable_id") == callable_id:
                collapsed_nodes.append(node_id)
                continue

            if node_data.get("callable_id") == callable_id:
                collapsed_nodes.append(node_id)

        exit_info: list[dict[str, Any]] = []
        for collapsed_node in collapsed_nodes:
            out_edges = list(cfg.out_edges(collapsed_node, data=True))
            return_edges = [
                edge for edge in out_edges if edge[2].get("edge_type") == "return"
            ]
            exit_info.append(
                {
                    "exit_ei": collapsed_node,
                    "exit_kind": "collapsed",
                    "has_return_edges": bool(return_edges),
                    "return_count": len(return_edges),
                    "returns_to": [target for _, target, _ in return_edges],
                }
            )

        return {
            "callable_id": callable_id,
            "has_exits": bool(exit_info),
            "exits": exit_info,
            "all_exits_have_returns": all(
                item["has_return_edges"] for item in exit_info
            )
            if exit_info
            else False,
        }

    success_exits = find_success_exit_eis(cfg, callable_id)
    exception_exits = find_exception_exit_eis(cfg, callable_id)

    exit_info: list[dict[str, Any]] = []
    for exit_ei in success_exits + exception_exits:
        out_edges = list(cfg.out_edges(exit_ei, data=True))
        return_edges = [
            edge for edge in out_edges if edge[2].get("edge_type") == "return"
        ]
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
        "all_exits_have_returns": all(item["has_return_edges"] for item in exit_info)
        if exit_info
        else False,
    }


def check_call_coverage(cfg: nx.DiGraph, callable_id: str) -> dict[str, Any]:
    entry = find_entry_ei(cfg, callable_id)

    target_nodes: list[str] = []
    if entry:
        target_nodes.append(entry)

    for node_id, node_data in cfg.nodes(data=True):
        if node_data.get("category") != "collapsed_internal_operation":
            continue

        if node_data.get("target_callable_id") == callable_id:
            target_nodes.append(node_id)
            continue

        if node_data.get("callable_id") == callable_id:
            target_nodes.append(node_id)

    seen_targets: set[str] = set()
    target_nodes = [
        n for n in target_nodes if not (n in seen_targets or seen_targets.add(n))
    ]

    if not target_nodes:
        return {"callable_id": callable_id, "entry_exists": False}

    call_edges: list[tuple[str, str, dict[str, Any]]] = []
    seen_edges: set[tuple[str, str]] = set()

    for target_node in target_nodes:
        for src, _, edge_data in cfg.in_edges(target_node, data=True):
            if edge_data.get("edge_type") != "call":
                continue

            edge_key = (src, target_node)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            call_edges.append((src, target_node, edge_data))

    callers: list[dict[str, Any]] = []
    for src, dst, edge_data in call_edges:
        caller_node = cfg.nodes[src]
        callers.append(
            {
                "call_site": src,
                "target_node": dst,
                "caller_callable_id": caller_node.get("callable_id"),
                "caller_callable_fqn": caller_node.get("callable_fqn"),
                "call_kind": edge_data.get("call_kind"),
            }
        )

    return {
        "callable_id": callable_id,
        "entry_exists": bool(entry),
        "is_called": bool(call_edges),
        "call_count": len(call_edges),
        "callers": callers,
    }


def check_execution_paths(
        cfg: nx.DiGraph,
        inventory_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0

    for callable_id, info in inventory_index.items():
        if not info.get("is_executable", True):
            continue
        if info.get("is_contract_method", False):
            continue

        entry_ei = find_entry_ei(cfg, callable_id)
        if not entry_ei:
            continue

        for candidate in info.get("integration_candidates", []) or []:
            target_ei = candidate.get("ei_id")
            if not target_ei or not cfg.has_node(target_ei):
                continue

            for recorded_path in candidate.get("execution_paths", []) or []:
                checked += 1
                ok = path_contains_recorded_execution_path(
                    cfg=cfg,
                    start_node=entry_ei,
                    target_node=target_ei,
                    recorded_path=recorded_path,
                )
                if not ok:
                    failure = find_first_failed_recorded_hop(
                        cfg=cfg,
                        start_node=entry_ei,
                        target_node=target_ei,
                        recorded_path=recorded_path,
                    )
                    failures.append(
                        {
                            "callable_id": callable_id,
                            "callable_fqn": info.get("callable_fqn"),
                            "integration_id": candidate.get("id"),
                            "ei_id": target_ei,
                            "target": candidate.get("target"),
                            "path": recorded_path,
                            "failed_hop": failure,
                        }
                    )

    return {
        "checked_paths": checked,
        "failure_count": len(failures),
        "failures": failures,
        "all_paths_valid": not failures,
    }


def diagnose_all_callables(
        cfg: nx.DiGraph,
        inventory_index: dict[str, dict[str, Any]],
        config: CheckerConfig,
) -> list[dict[str, Any]]:
    callables = collect_callables_from_graph(cfg)
    results: list[dict[str, Any]] = []

    for callable_id in sorted(callables):
        callable_info = callables[callable_id]
        inventory_info = inventory_index.get(callable_id, {})

        integrity = check_callable_integrity(cfg, callable_id)
        coverage = check_call_coverage(cfg, callable_id)
        returns = check_return_edges(
            cfg,
            callable_id,
            inventory_info=inventory_info,
            callable_info=callable_info,
            config=config,
        )

        is_executable = inventory_info.get("is_executable", True)
        is_contract_method = inventory_info.get("is_contract_method", False)

        low_signal_category_name = low_signal_category(callable_info, inventory_info, config)
        low_signal = low_signal_category_name is not None
        has_incoming_callable_calls = has_incoming_callable_call_edges(cfg, callable_id)
        externally_reachable = externally_reachable_via_marker(cfg, callable_id, config)

        results.append(
            {
                "callable_id": callable_id,
                "callable_name": callable_info.get("callable_name"),
                "callable_fqn": callable_info.get("callable_fqn"),
                "callable_kind": callable_info.get("callable_kind"),
                "is_executable": is_executable,
                "is_contract_method": is_contract_method,
                "has_incoming_callable_calls": has_incoming_callable_calls,
                "externally_reachable": externally_reachable,
                "low_signal": low_signal,
                "low_signal_category": low_signal_category_name,
                "integrity": integrity,
                "returns": returns,
                "coverage": coverage,
                "is_broken": is_executable and not integrity["valid"],
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


def is_private_module_helper(result: dict[str, Any]) -> bool:
    if result.get("callable_kind") != "function":
        return False

    name = (result.get("callable_name") or "").strip()
    if not name.startswith("_"):
        return False
    if name.startswith("__"):
        return False

    fqn = (result.get("callable_fqn") or "").strip()
    if not fqn:
        return False

    parts = fqn.split(".")
    if len(parts) < 2:
        return False

    penultimate = parts[-2]
    if penultimate[:1].isupper():
        return False

    return True


def count_same_unit_branch_references(
        result: dict[str, Any],
        inventory_info: dict[str, Any],
) -> int:
    helper_name = (result.get("callable_name") or "").strip()
    if not helper_name:
        return 0

    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(helper_name)}\s*\(")
    hits = 0

    for branch in inventory_info.get("branches", []) or []:
        texts: list[str] = []

        desc = branch.get("description")
        if isinstance(desc, str):
            texts.append(desc)

        constraint = branch.get("constraint") or {}
        expr = constraint.get("expr")
        if isinstance(expr, str):
            texts.append(expr)

        op_target = constraint.get("operation_target")
        if isinstance(op_target, str):
            texts.append(op_target)

        for target in branch.get("conditional_targets", []) or []:
            cond = target.get("target_condition")
            if isinstance(cond, str):
                texts.append(cond)

            hint = target.get("target_hint") or {}
            hint_expr = hint.get("expr")
            if isinstance(hint_expr, str):
                texts.append(hint_expr)

        for text in texts:
            if pattern.search(text):
                hits += 1
                break

    return hits


def find_same_unit_branch_references(
        result: dict[str, Any],
        inventory_index: dict[str, dict[str, Any]],
) -> list[str]:
    target_name = (result.get("callable_name") or "").strip()
    if not target_name:
        return []

    target_fqn = (result.get("callable_fqn") or "").strip()
    if not target_fqn:
        return []

    target_module = target_fqn.rsplit(".", 1)[0] if "." in target_fqn else ""
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(target_name)}\s*\(")

    referenced_by: list[str] = []

    for other_callable_id, other_info in inventory_index.items():
        if other_callable_id == result.get("callable_id"):
            continue

        other_fqn = other_info.get("callable_fqn", "") or ""
        other_module = other_fqn.rsplit(".", 1)[0] if "." in other_fqn else ""
        if other_module != target_module:
            continue

        matched = False
        for branch in other_info.get("branches", []) or []:
            texts: list[str] = []

            desc = branch.get("description")
            if isinstance(desc, str):
                texts.append(desc)

            constraint = branch.get("constraint") or {}
            expr = constraint.get("expr")
            if isinstance(expr, str):
                texts.append(expr)

            op_target = constraint.get("operation_target")
            if isinstance(op_target, str):
                texts.append(op_target)

            for conditional_target in branch.get("conditional_targets", []) or []:
                target_condition = conditional_target.get("target_condition")
                if isinstance(target_condition, str):
                    texts.append(target_condition)

                hint = conditional_target.get("target_hint") or {}
                hint_expr = hint.get("expr")
                if isinstance(hint_expr, str):
                    texts.append(hint_expr)

            if any(pattern.search(text) for text in texts):
                matched = True
                break

        if matched:
            referenced_by.append(other_fqn)

    return sorted(set(referenced_by))


def build_private_helper_suspect_items(
        results: list[dict[str, Any]],
        inventory_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    suspects: list[dict[str, Any]] = []

    for result in results:
        if result.get("is_contract_method", False):
            continue
        if result.get("low_signal", True):
            continue
        if result.get("externally_reachable", False):
            continue
        if (result.get("coverage") or {}).get("is_called", False):
            continue
        if not is_private_module_helper(result):
            continue

        helper_fqn = result.get("callable_fqn", "") or ""
        helper_module = helper_fqn.rsplit(".", 1)[0] if "." in helper_fqn else ""
        branch_ref_count = 0

        for other_callable_id, other_info in inventory_index.items():
            other_fqn = other_info.get("callable_fqn", "") or ""
            other_module = other_fqn.rsplit(".", 1)[0] if "." in other_fqn else ""
            if other_module != helper_module:
                continue
            if other_callable_id == result.get("callable_id"):
                continue

            branch_ref_count += count_same_unit_branch_references(result, other_info)

        if branch_ref_count > 0:
            enriched = dict(result)
            enriched["same_unit_branch_reference_count"] = branch_ref_count
            suspects.append(enriched)

    suspects.sort(
        key=lambda item: (
            -item["same_unit_branch_reference_count"],
            item.get("callable_fqn", "") or "",
        )
    )
    return suspects


def build_same_unit_reference_suspect_items(
        results: list[dict[str, Any]],
        inventory_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    suspects: list[dict[str, Any]] = []

    for result in results:
        if result.get("low_signal", True):
            continue
        if result.get("externally_reachable", False):
            continue
        if (result.get("coverage") or {}).get("is_called", False):
            continue

        referenced_by = find_same_unit_branch_references(result, inventory_index)
        if not referenced_by:
            continue

        enriched = dict(result)
        enriched["reference_count"] = len(referenced_by)
        enriched["referenced_by"] = referenced_by
        suspects.append(enriched)

    suspects.sort(
        key=lambda item: (
            -item["reference_count"],
            item.get("callable_fqn", "") or "",
        )
    )
    return suspects


def build_missing_return_edge_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for result in results:
        if result.get("low_signal", True):
            continue
        if not (result.get("coverage") or {}).get("is_called", False):
            continue
        if not (result.get("returns") or {}).get("has_exits", False):
            continue
        if (result.get("returns") or {}).get("all_exits_have_returns", False):
            continue

        enriched = dict(result)
        enriched["missing_exits"] = [
            exit_info
            for exit_info in (result.get("returns") or {}).get("exits", []) or []
            if not exit_info.get("has_return_edges", False)
        ]
        items.append(enriched)

    return items


def build_low_signal_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(result) for result in results if result.get("low_signal", False)]


def build_enriched_results(
        results: list[dict[str, Any]],
        private_helper_items: list[dict[str, Any]],
        same_unit_reference_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    private_ids = {item["callable_id"] for item in private_helper_items}
    same_unit_ids = {item["callable_id"] for item in same_unit_reference_items}

    enriched_results: list[dict[str, Any]] = []
    for item in results:
        enriched = dict(item)
        enriched["is_called"] = bool((item.get("coverage") or {}).get("is_called", False))
        enriched["has_missing_return_edges"] = bool(
            (item.get("returns") or {}).get("has_exits", False)
            and not (item.get("returns") or {}).get("all_exits_have_returns", False)
        )
        enriched["private_helper_suspect"] = item.get("callable_id") in private_ids
        enriched["same_unit_reference_suspect"] = item.get("callable_id") in same_unit_ids
        enriched_results.append(enriched)

    return enriched_results


def callable_module(item: dict[str, Any]) -> str:
    fqn = item.get("callable_fqn", "") or ""
    return fqn.rsplit(".", 1)[0] if "." in fqn else fqn


def group_items(items: list[dict[str, Any]], key_name: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        if key_name == "module":
            key = callable_module(item)
        else:
            key = item.get(key_name)

        key_str = str(key) if key is not None else "<none>"
        grouped.setdefault(key_str, []).append(item)

    return dict(sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def apply_grouping(items: list[dict[str, Any]], section: dict[str, Any]) -> dict[str, Any]:
    primary = section.get("group_by")
    secondary = section.get("secondary_group_by")

    if not primary:
        return {
            "count": len(items),
            "full_list": items,
        }

    primary_grouped = group_items(items, primary)

    payload: dict[str, Any] = {
        "count": len(items),
        "group_by": primary,
        "grouped": {},
    }

    for primary_key, primary_items in primary_grouped.items():
        bucket: dict[str, Any] = {"count": len(primary_items)}

        if secondary:
            secondary_grouped = group_items(primary_items, secondary)
            bucket["secondary_group_by"] = secondary
            bucket["grouped"] = {
                secondary_key: {
                    "count": len(secondary_items),
                    "items": secondary_items,
                }
                for secondary_key, secondary_items in secondary_grouped.items()
            }
        else:
            bucket["items"] = primary_items

        payload["grouped"][primary_key] = bucket

    return payload


def section_match(item: dict[str, Any], include_when: dict[str, Any]) -> bool:
    for key, expected in include_when.items():
        actual = item.get(key)
        if actual != expected:
            return False
    return True


def build_section_payloads(
    cfg: nx.DiGraph,
    results: list[dict[str, Any]],
    config: CheckerConfig,
    private_helper_items: list[dict[str, Any]],
    same_unit_reference_items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    missing_return_items = build_missing_return_edge_items(results)
    low_signal_items = build_low_signal_items(results)
    enriched_results = build_enriched_results(
        results,
        private_helper_items,
        same_unit_reference_items,
    )

    analysis_registry = build_analysis_registry(cfg, results)

    special_items_by_id: dict[str, list[dict[str, Any]]] = {
        "missing_return_edges": missing_return_items,
        "private_helper_suspects": private_helper_items,
        "same_unit_reference_suspects": same_unit_reference_items,
        "low_signal": low_signal_items,
    }

    section_payloads: dict[str, dict[str, Any]] = {}

    for section in config.sections:
        section_id = section.get("id")
        report_key = section.get("report_key", section_id)
        source = section.get("source", {}) or {}
        source_kind = source.get("kind", "results")

        if source_kind == "analysis":
            analysis_id = source.get("analysis_id")
            if not analysis_id:
                raise ValueError(
                    f"Section {section_id!r} has source.kind=analysis but no analysis_id"
                )
            analysis_payload = analysis_registry.get(analysis_id)
            if analysis_payload is None:
                raise ValueError(
                    f"Section {section_id!r} references unknown analysis_id {analysis_id!r}"
                )

            section_payloads[report_key] = normalize_analysis_section_payload(
                section,
                analysis_payload,
            )
            continue

        include_when = section.get("include_when", {}) or {}

        if section_id in special_items_by_id:
            items = special_items_by_id[section_id]
        else:
            items = [
                item
                for item in enriched_results
                if section_match(item, include_when)
            ]

        section_payloads[report_key] = {
            "id": section_id,
            "report_key": report_key,
            "section_header": section.get("section_header"),
            "description": section.get("description"),
            **apply_grouping(items, section),
        }

    return section_payloads


def build_low_signal_bins(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for item in results:
        category = item.get("low_signal_category")
        if category:
            counts[str(category)] += 1
    return dict(counts)


def parse_summary_codes(summary_value: str | None, config: CheckerConfig) -> tuple[set[str], bool]:
    if not summary_value:
        return set(), False

    codes = {ch for ch in summary_value.strip() if not ch.isspace()}
    valid_codes = {
        str(section.get("summary_code", "")).strip()
        for section in config.sections
        if section.get("summary_code")
    }

    unknown = sorted(code for code in codes if code not in valid_codes)
    if unknown:
        raise ValueError(
            f"Unknown --summary code(s): {', '.join(unknown)}. "
            f"Valid codes: {''.join(sorted(valid_codes))}"
        )

    # only "l" means additive low-signal on top of default behavior
    low_signal_additive = codes == {"l"}
    return codes, low_signal_additive


def should_include_section_in_console_summary(
        section_config: dict[str, Any],
        section_payload: dict[str, Any],
        requested_codes: set[str],
        low_signal_additive: bool,
) -> bool:
    if int(section_payload.get("count", 0) or 0) <= 0:
        return False

    section_code = str(section_config.get("summary_code", "")).strip()
    default_enabled = bool(section_config.get("include_in_console_summary", False))

    # no --summary at all: default configured behavior
    if not requested_codes:
        return default_enabled

    # special case: only "l" means additive low-signal
    if low_signal_additive:
        if section_code == "l":
            return True
        return default_enabled

    # otherwise include-only
    return section_code in requested_codes


def format_section_header_with_count(section_header: str, count: int) -> str:
    base = section_header.strip()
    if base.startswith("===") and base.endswith("==="):
        inner = base.strip("=").strip()
        return f"=== {inner} [{count}] ==="
    return f"{base} [{count}]"


def render_console_summary(
        config: CheckerConfig,
        section_payloads: dict[str, dict[str, Any]],
        path_check: dict[str, Any],
        requested_codes: set[str],
        low_signal_additive: bool,
) -> None:
    print("\n=== Diagnostic Summary ===")
    print(f"Recorded execution paths checked: {path_check['checked_paths']}")
    print(f"Execution path failures: {path_check['failure_count']}")

    for section in config.sections:
        report_key = section.get("report_key", section.get("id"))
        payload = section_payloads.get(report_key)
        if payload is None:
            continue

        if should_include_section_in_console_summary(
                section_config=section,
                section_payload=payload,
                requested_codes=requested_codes,
                low_signal_additive=low_signal_additive,
        ):
            print(
                format_section_header_with_count(
                    str(payload.get("section_header", report_key)),
                    int(payload.get("count", 0) or 0),
                )
            )


def build_analysis_registry(
        cfg: nx.DiGraph,
        results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "structural_health": build_structural_health_report(cfg, results),
        "ei_validation": validate_ei_ids(cfg),
        "collapsed_node_health": analyze_collapsed_node_health(cfg),
        "edge_target_health": analyze_edge_target_health(cfg),
        "callable_call_cycles": analyze_callable_call_cycles(cfg),
        "external_seams": analyze_external_seams(results),
    }


def normalize_analysis_section_payload(
        section: dict[str, Any],
        analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    count = analysis_payload.get("count")
    if count is None:
        count = analysis_payload.get("problem_count")
    if count is None:
        count = 0

    return {
        "id": section.get("id"),
        "report_key": section.get("report_key", section.get("id")),
        "section_header": section.get("section_header"),
        "description": section.get("description"),
        "count": int(count),
        "analysis_payload": analysis_payload,
    }


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


def print_analysis_payload(section: dict[str, Any]) -> None:
    print(f"\n{section.get('section_header', '=== Section ===')}")

    analysis_payload = section.get("analysis_payload", {}) or {}
    count = section.get("count", 0)
    print(f"Count: {count}")

    print(yaml.dump(analysis_payload, sort_keys=False, allow_unicode=True, width=float('inf')))


def build_structural_health_report(
        cfg: nx.DiGraph,
        results: list[dict[str, Any]],
) -> dict[str, Any]:
    ei_validation = validate_ei_ids(cfg)
    collapsed_health = analyze_collapsed_node_health(cfg)
    edge_target_health = analyze_edge_target_health(cfg)
    callable_cycles = analyze_callable_call_cycles(cfg)
    external_seams = analyze_external_seams(results)

    return {
        "ei_validation": ei_validation,
        "collapsed_node_health": collapsed_health,
        "edge_target_health": edge_target_health,
        "callable_call_cycles": callable_cycles,
        "external_seams": external_seams,
        "problem_count": (
                ei_validation["ei_regex_mismatch_count"]
                + ei_validation["callables_missing_entryish_ei_count"]
                + collapsed_health["collapsed_without_return_count"]
                + collapsed_health["collapsed_without_incoming_call_count"]
                + collapsed_health["collapsed_missing_target_callable_count"]
                + edge_target_health["bad_call_targets_count"]
                + edge_target_health["bad_return_targets_count"]
                + callable_cycles["multi_node_scc_count"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnostics for inventory-first EI graphs")
    parser.add_argument("cfg", type=Path, help="Path to graph pickle")
    parser.add_argument(
        "--inventories-root",
        type=Path,
        required=True,
        help="Root containing *.inventory.yaml files",
    )
    parser.add_argument(
        "--checker-config",
        type=Path,
        default=Path(__file__).with_name("graph_checker_config.json"),
        help="Path to checker configuration JSON",
    )
    parser.add_argument(
        "--callable-id",
        type=str,
        help="Show detailed diagnostics for one callable ID",
    )
    parser.add_argument(
        "--broken-callable-id",
        type=str,
        help="Show focused broken-callable analysis for one callable ID",
    )
    parser.add_argument(
        "--summary",
        metavar="CODES",
        default="aghprs",
        help=(
            "Console summary sections. Codes: "
            "a=abstraction, "
            "g=general, "
            "h=structural-health, "
            "l=low-signal, "
            "p=private-helpers, "
            "r=missing-returns, "
            "s=same-unit. "
            "If omitted, default configured sections with findings are shown. "
            "If set to only 'l', low-signal is added to the default summary. "
            "Any other value makes summary output include-only for the specified codes."
        ),
    )
    parser.add_argument("--write-report", type=Path, help="Write YAML report to this path")
    args = parser.parse_args(argv)

    print(f"Loading CFG from {args.cfg}...")
    cfg = load_cfg(args.cfg)
    print(f"  {cfg.number_of_nodes()} nodes, {cfg.number_of_edges()} edges")

    if not args.inventories_root.exists():
        print(f"ERROR: inventories root not found: {args.inventories_root}", file=sys.stderr)
        return 1

    if not args.checker_config.exists():
        print(f"ERROR: checker config not found: {args.checker_config}", file=sys.stderr)
        return 1

    checker_config = load_checker_config(args.checker_config)

    inventory_paths = discover_inventory_files(args.inventories_root)
    if not inventory_paths:
        print(f"ERROR: no inventory files found under {args.inventories_root}", file=sys.stderr)
        return 1

    inventory_index = index_inventory_execution_paths(inventory_paths)

    print("Running diagnostics...")
    results = diagnose_all_callables(cfg, inventory_index, checker_config)
    path_check = check_execution_paths(cfg, inventory_index)
    private_helper_items = build_private_helper_suspect_items(results, inventory_index)
    same_unit_reference_items = build_same_unit_reference_suspect_items(results, inventory_index)
    section_payloads = build_section_payloads(
        cfg,
        results,
        checker_config,
        private_helper_items,
        same_unit_reference_items,
    )

    low_signal_bins = build_low_signal_bins(results)

    try:
        requested_summary_codes, low_signal_additive = parse_summary_codes(
            args.summary,
            checker_config,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    render_console_summary(
        config=checker_config,
        section_payloads=section_payloads,
        path_check=path_check,
        requested_codes=requested_summary_codes,
        low_signal_additive=low_signal_additive,
    )

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
            "checker_config_path": str(args.checker_config),
            "execution_path_check": path_check,
            "low_signal_bins": low_signal_bins,
            "results": results,
            "sections": section_payloads,
        }
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_report, "w", encoding="utf-8") as f:
            yaml.dump(report, f, sort_keys=False, allow_unicode=True, width=float("inf"))
        print(f"Wrote report to {args.write_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
