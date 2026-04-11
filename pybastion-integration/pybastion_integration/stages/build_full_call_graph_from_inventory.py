#!/usr/bin/env python3
"""
Build an execution-instance-level control-flow graph directly from stage 3 inventory files.

This version is inventory-first:
- It consumes *.inventory.yaml files directly.
- It preserves explicit within-callable flow from branch metadata.
- It stitches local/interunit/external call edges from ast_analysis.integration_candidates.
- It verifies that recorded execution_paths are realizable in the assembled graph.

The goal is assembly, not reconstruction.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

# =============================================================================
# Constants
# =============================================================================

CALLABLE_KINDS: set[str] = {"function", "method", "assignment"}
TERMINAL_TERMINATORS: set[str] = {"return", "implicit-return", "raise", "exception"}
SUCCESS_EXIT_TERMINATORS: set[str] = {"return", "implicit-return"}
EXCEPTION_EXIT_TERMINATORS: set[str] = {"raise", "exception"}


# =============================================================================
# Data structures
# =============================================================================


@dataclass(slots=True)
class CallableContext:
    callable_id: str
    callable_name: str
    callable_kind: str
    callable_fqn: str
    unit_name: str
    unit_fqn: str
    entry: dict[str, Any]
    branches: list[dict[str, Any]]
    integration_by_ei: dict[str, list[dict[str, Any]]]


# =============================================================================
# File discovery
# =============================================================================


def discover_inventory_files(inventories_root: Path) -> list[Path]:
    inventory_files = list(inventories_root.rglob("*.inventory.yaml"))
    inventory_files.extend(inventories_root.rglob("*_inventory.yaml"))
    return sorted(set(inventory_files))


# =============================================================================
# Inventory traversal and indexing
# =============================================================================


def normalize_kind_for_callable(kind: str) -> str:
    if kind in {"function", "method", "assignment"}:
        return kind
    return kind


def build_callable_fqn(unit_fqn: str, ancestor_names: list[str], entry_name: str) -> str:
    parts = [unit_fqn, *ancestor_names, entry_name]
    return ".".join(part for part in parts if part)


def index_inventory(inventory: dict[str, Any]) -> tuple[dict[str, CallableContext], dict[str, str], dict[str, str]]:
    """
    Returns:
      callable_contexts: callable_id -> CallableContext
      fqn_to_callable_id: callable_fqn -> callable_id
      ei_to_callable_id: ei_id -> callable_id
    """
    unit_name = inventory["unit"]
    unit_fqn = inventory.get("fully_qualified_name", unit_name)

    callable_contexts: dict[str, CallableContext] = {}
    fqn_to_callable_id: dict[str, str] = {}
    ei_to_callable_id: dict[str, str] = {}

    def recurse(entries: list[dict[str, Any]], ancestors: list[str]) -> None:
        for entry in entries:
            kind = entry.get("kind", "unknown")
            name = entry.get("name", "unknown")
            entry_id = entry["id"]
            children = entry.get("children", []) or []
            branches = entry.get("branches", []) or []
            ast_analysis = entry.get("ast_analysis") or {}
            integration_candidates = ast_analysis.get("integration_candidates") or []

            callable_fqn = build_callable_fqn(unit_fqn, ancestors, name)

            if kind in CALLABLE_KINDS:
                if entry.get("is_executable") is False:
                    recurse(children, [*ancestors, name])
                    continue
                integration_by_ei: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for candidate in integration_candidates:
                    ei_id = candidate.get("ei_id")
                    if ei_id:
                        integration_by_ei[ei_id].append(candidate)

                context = CallableContext(
                    callable_id=entry_id,
                    callable_name=name,
                    callable_kind=normalize_kind_for_callable(kind),
                    callable_fqn=callable_fqn,
                    unit_name=unit_name,
                    unit_fqn=unit_fqn,
                    entry=entry,
                    branches=branches,
                    integration_by_ei=dict(integration_by_ei),
                )
                callable_contexts[entry_id] = context
                fqn_to_callable_id[callable_fqn] = entry_id
                for branch in branches:
                    ei_to_callable_id[branch["id"]] = entry_id

            recurse(children, [*ancestors, name])

    recurse(inventory.get("entries", []) or [], [])
    return callable_contexts, fqn_to_callable_id, ei_to_callable_id


def load_all_inventories(inventory_paths: list[Path]) -> tuple[
    dict[str, CallableContext], dict[str, str], dict[str, str]]:
    all_contexts: dict[str, CallableContext] = {}
    fqn_to_callable_id: dict[str, str] = {}
    ei_to_callable_id: dict[str, str] = {}

    for path in inventory_paths:
        with open(path, "r", encoding="utf-8") as f:
            inventory = yaml.safe_load(f)
        if not inventory:
            continue

        contexts, fqn_map, ei_map = index_inventory(inventory)
        all_contexts.update(contexts)
        fqn_to_callable_id.update(fqn_map)
        ei_to_callable_id.update(ei_map)

    return all_contexts, fqn_to_callable_id, ei_to_callable_id


# =============================================================================
# Graph helpers
# =============================================================================


def _decorator_name(decorator: dict[str, Any]) -> str:
    return str((decorator or {}).get("name", "")).strip()


def is_collapsible_operation_entry(entry: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    """
    Return (should_collapse, marker_name, marker_type).
    Collapse MechanicalOperation and UtilityOperation targets.
    """
    decorators = entry.get("decorators", []) or []
    for decorator in decorators:
        name = _decorator_name(decorator)
        if name in {"MechanicalOperation", "UtilityOperation"}:
            kwargs = decorator.get("kwargs") or {}
            op_type = kwargs.get("type")
            return True, name, op_type
    return False, None, None


def add_callable_nodes(cfg: nx.DiGraph, context: CallableContext) -> None:
    decorators = context.entry.get("decorators", []) or []

    for branch in context.branches:
        ei_id = branch["id"]
        statement_outcome = branch.get("statement_outcome") or {}
        cfg.add_node(
            ei_id,
            category="execution_instance",
            callable_id=context.callable_id,
            callable_name=context.callable_name,
            callable_kind=context.callable_kind,
            callable_fqn=context.callable_fqn,
            unit=context.unit_name,
            unit_fqn=context.unit_fqn,
            line=branch.get("line"),
            stmt_type=branch.get("stmt_type"),
            condition=branch.get("condition"),
            description=branch.get("description"),
            owner_info=branch.get("owner_info"),
            constraint=branch.get("constraint"),
            statement_outcome=statement_outcome,
            decorators=decorators,
            is_terminal=bool(statement_outcome.get("is_terminal", False)),
            terminates_via=statement_outcome.get("terminates_via"),
        )


def add_explicit_within_callable_edges(cfg: nx.DiGraph, context: CallableContext) -> None:
    for branch in context.branches:
        src = branch["id"]

        statement_outcome = branch.get("statement_outcome") or {}
        target_ei = statement_outcome.get("target_ei")
        if target_ei and not statement_outcome.get("is_terminal"):
            cfg.add_edge(
                src,
                target_ei,
                edge_type="statement_outcome",
                within_callable=True,
                callable_id=context.callable_id,
            )

        for conditional in branch.get("conditional_targets", []) or []:
            target_ei = conditional.get("target_ei")
            if target_ei and not conditional.get("is_terminal"):
                cfg.add_edge(
                    src,
                    target_ei,
                    edge_type="conditional_target",
                    within_callable=True,
                    callable_id=context.callable_id,
                    target_condition=conditional.get("target_condition"),
                    condition_result=conditional.get("condition_result"),
                    target_hint=conditional.get("target_hint"),
                )

        for disruptive in branch.get("disruptive_outcomes", []) or []:
            target_ei = disruptive.get("target_ei")
            if target_ei and not disruptive.get("is_terminal"):
                cfg.add_edge(
                    src,
                    target_ei,
                    edge_type="disruptive_outcome",
                    within_callable=True,
                    callable_id=context.callable_id,
                    outcome=disruptive.get("outcome"),
                )


def classify_integration_target_kind(candidate: dict[str, Any]) -> str:
    return (
            candidate.get("kind")
            or ((candidate.get("classification") or {}).get("kind"))
            or "unknown"
    )


def classify_call_site(branch: dict[str, Any]) -> bool:
    constraint = branch.get("constraint") or {}
    return bool(constraint.get("operation_target"))


def operation_target_for_branch(branch: dict[str, Any]) -> str | None:
    constraint = branch.get("constraint") or {}
    return constraint.get("operation_target")


def compute_success_exit_eis(context: CallableContext) -> list[str]:
    exits: list[str] = []
    for branch in context.branches:
        outcome = branch.get("statement_outcome") or {}
        if outcome.get("is_terminal") and outcome.get("terminates_via") in SUCCESS_EXIT_TERMINATORS:
            exits.append(branch["id"])
    return exits


def compute_exception_exit_eis(context: CallableContext) -> list[str]:
    exits: list[str] = []
    for branch in context.branches:
        outcome = branch.get("statement_outcome") or {}
        if outcome.get("is_terminal") and outcome.get("terminates_via") in EXCEPTION_EXIT_TERMINATORS:
            exits.append(branch["id"])
    return exits


def immediate_intra_callable_successors(cfg: nx.DiGraph, callable_id: str, ei_id: str) -> list[str]:
    successors: list[str] = []
    for succ in cfg.successors(ei_id):
        edge = cfg.edges[ei_id, succ]
        if edge.get("within_callable") and edge.get("callable_id") == callable_id:
            successors.append(succ)
    return successors


def resolve_return_destinations(cfg: nx.DiGraph, context: CallableContext, ei_id: str) -> list[str]:
    """
    Return destinations are the immediate within-callable successors of the call EI.
    """
    return immediate_intra_callable_successors(cfg, context.callable_id, ei_id)


def add_call_and_return_edges(
        cfg: nx.DiGraph,
        context: CallableContext,
        callable_contexts: dict[str, CallableContext],
        fqn_to_callable_id: dict[str, str],
) -> None:
    success_exit_cache: dict[str, list[str]] = {}
    exception_exit_cache: dict[str, list[str]] = {}

    for branch in context.branches:
        ei_id = branch["id"]
        if not classify_call_site(branch):
            continue

        integration_candidates = context.integration_by_ei.get(ei_id, [])
        operation_target = operation_target_for_branch(branch)
        return_targets = resolve_return_destinations(cfg, context, ei_id)

        if not integration_candidates:
            external_node_id = f"external::{ei_id}"
            cfg.add_node(
                external_node_id,
                category="external_call",
                external_kind="unresolved",
                operation_target=operation_target,
                called_from=ei_id,
                callable_id=context.callable_id,
                callable_fqn=context.callable_fqn,
            )
            cfg.add_edge(ei_id, external_node_id, edge_type="call", call_kind="unresolved")
            for return_target in return_targets:
                cfg.add_edge(external_node_id, return_target, edge_type="return", return_kind="external")
            continue

        for candidate in integration_candidates:
            kind = classify_integration_target_kind(candidate)
            target = candidate.get("target") or ((candidate.get("classification") or {}).get("resolved_target"))
            signature = candidate.get("signature")

            if kind == "interunit" and target and target in fqn_to_callable_id:
                target_callable_id = fqn_to_callable_id[target]
                target_context = callable_contexts[target_callable_id]
                collapse_target, marker_name, marker_type = is_collapsible_operation_entry(target_context.entry)

                if collapse_target:
                    collapsed_node_id = f"collapsed::{ei_id}::{target_callable_id}"

                    cfg.add_node(
                        collapsed_node_id,
                        category="collapsed_internal_operation",
                        callable_id=target_callable_id,
                        callable_fqn=target_context.callable_fqn,
                        callable_name=target_context.callable_name,
                        marker_name=marker_name,
                        marker_type=marker_type,
                        called_from=ei_id,
                    )

                    cfg.add_edge(
                        ei_id,
                        collapsed_node_id,
                        edge_type="call",
                        call_kind="collapsed_internal_operation",
                        target_callable_id=target_callable_id,
                        target_fqn=target,
                        marker_name=marker_name,
                        marker_type=marker_type,
                        signature=signature,
                    )

                    for return_target in return_targets:
                        cfg.add_edge(
                            collapsed_node_id,
                            return_target,
                            edge_type="return",
                            return_kind="collapsed_internal_operation",
                            original_call_site=ei_id,
                            returns_from=target_callable_id,
                        )

                    continue
                target_context = None
                # target_context lookup deferred through graph nodes; keep cheap map from callable ids present in graph later
                target_entry_ei = None
                for node_id, node_data in cfg.nodes(data=True):
                    if node_data.get("callable_id") == target_callable_id:
                        target_entry_ei = node_id
                        break

                if target_entry_ei is None:
                    # Target callable exists, but nodes not yet present would indicate bad build ordering.
                    # Since we add all nodes before edges, this should not happen.
                    raise ValueError(f"Could not find entry EI for interunit target {target} ({target_callable_id})")

                cfg.add_edge(
                    ei_id,
                    target_entry_ei,
                    edge_type="call",
                    call_kind="interunit",
                    target_callable_id=target_callable_id,
                    target_fqn=target,
                    signature=signature,
                )

                if target_callable_id not in success_exit_cache:
                    target_branches = [
                        {"id": node_id, "statement_outcome": {"is_terminal": data.get("is_terminal", False),
                                                              "terminates_via": data.get("terminates_via")}}
                        for node_id, data in cfg.nodes(data=True)
                        if data.get("callable_id") == target_callable_id
                    ]
                    fake_context = CallableContext(
                        callable_id=target_callable_id,
                        callable_name="",
                        callable_kind="",
                        callable_fqn=target,
                        unit_name="",
                        unit_fqn="",
                        entry={},
                        branches=target_branches,
                        integration_by_ei={},
                    )
                    success_exit_cache[target_callable_id] = compute_success_exit_eis(fake_context)
                    exception_exit_cache[target_callable_id] = compute_exception_exit_eis(fake_context)

                for return_target in return_targets:
                    for exit_ei in success_exit_cache[target_callable_id]:
                        cfg.add_edge(
                            exit_ei,
                            return_target,
                            edge_type="return",
                            return_kind="success",
                            original_call_site=ei_id,
                            returns_from=target_callable_id,
                        )
                    for exit_ei in exception_exit_cache[target_callable_id]:
                        cfg.add_edge(
                            exit_ei,
                            return_target,
                            edge_type="return",
                            return_kind="exception",
                            original_call_site=ei_id,
                            returns_from=target_callable_id,
                        )
            else:
                external_node_id = f"external::{ei_id}::{kind}::{len(return_targets)}::{target or 'unknown'}"
                cfg.add_node(
                    external_node_id,
                    category="external_call",
                    external_kind=kind,
                    target=target,
                    signature=signature,
                    called_from=ei_id,
                    callable_id=context.callable_id,
                    callable_fqn=context.callable_fqn,
                    execution_paths=candidate.get("execution_paths", []),
                    path_analysis=candidate.get("path_analysis"),
                )
                cfg.add_edge(
                    ei_id,
                    external_node_id,
                    edge_type="call",
                    call_kind=kind,
                    target=target,
                    signature=signature,
                )
                for return_target in return_targets:
                    cfg.add_edge(
                        external_node_id,
                        return_target,
                        edge_type="return",
                        return_kind="external",
                        original_call_site=ei_id,
                    )


# =============================================================================
# Verification
# =============================================================================


def path_exists_exactly(cfg: nx.DiGraph, path: list[str]) -> bool:
    if not path:
        return False
    for index in range(len(path) - 1):
        if not cfg.has_edge(path[index], path[index + 1]):
            return False
    return True


def verify_recorded_execution_paths(
        cfg: nx.DiGraph,
        callable_contexts: dict[str, CallableContext],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    for context in callable_contexts.values():
        for candidates in context.integration_by_ei.values():
            for candidate in candidates:
                for path in candidate.get("execution_paths", []) or []:
                    if not path_exists_exactly(cfg, path):
                        failures.append(
                            {
                                "callable_id": context.callable_id,
                                "callable_fqn": context.callable_fqn,
                                "integration_id": candidate.get("id"),
                                "ei_id": candidate.get("ei_id"),
                                "path": path,
                            }
                        )
    return failures


# =============================================================================
# Graph build orchestration
# =============================================================================


def build_graph_from_inventories(inventory_paths: list[Path]) -> tuple[
    nx.DiGraph, dict[str, CallableContext], list[dict[str, Any]]]:
    callable_contexts, fqn_to_callable_id, _ei_to_callable_id = load_all_inventories(inventory_paths)

    cfg = nx.DiGraph()

    # Pass 1: add all EI nodes.
    for context in callable_contexts.values():
        add_callable_nodes(cfg, context)

    # Pass 2: add all explicit within-callable edges.
    for context in callable_contexts.values():
        add_explicit_within_callable_edges(cfg, context)

    # Pass 3: add cross-call / external seam edges.
    for context in callable_contexts.values():
        add_call_and_return_edges(cfg, context, callable_contexts, fqn_to_callable_id)

    # Verification: recorded execution paths must be present.
    failures = verify_recorded_execution_paths(cfg, callable_contexts)
    return cfg, callable_contexts, failures


# =============================================================================
# Serialization
# =============================================================================


def serialize_graph(cfg: nx.DiGraph, output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "pickle":
        with open(output_path, "wb") as f:
            pickle.dump(cfg, f)
        return

    if fmt == "yaml":
        graph_data = nx.node_link_data(cfg)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(graph_data, f, sort_keys=False, allow_unicode=True, width=float("inf"))
        return

    if fmt == "graphml":
        nx.write_graphml(cfg, output_path)
        return

    raise ValueError(f"Unsupported format: {fmt}")


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build EI graph directly from inventory files")
    parser.add_argument("--inventories-root", type=Path, required=True, help="Root containing *.inventory.yaml files")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument("--format", choices=["pickle", "yaml", "graphml"], default="pickle")
    parser.add_argument("--fail-on-path-mismatch", action="store_true",
                        help="Exit nonzero if recorded execution_paths are not realizable")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.inventories_root.exists():
        print(f"ERROR: inventories root not found: {args.inventories_root}", file=sys.stderr)
        return 1

    inventory_paths = discover_inventory_files(args.inventories_root)
    if not inventory_paths:
        print(f"ERROR: no inventory files found under {args.inventories_root}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(inventory_paths)} inventory file(s)")

    cfg, callable_contexts, failures = build_graph_from_inventories(inventory_paths)

    if args.verbose:
        print(f"Callables indexed: {len(callable_contexts)}")
        print(f"Graph nodes: {cfg.number_of_nodes()}")
        print(f"Graph edges: {cfg.number_of_edges()}")
        edge_type_counts: dict[str, int] = defaultdict(int)
        for _, _, data in cfg.edges(data=True):
            edge_type_counts[data.get("edge_type", "unknown")] += 1
        print("Edge types:")
        for edge_type, count in sorted(edge_type_counts.items()):
            print(f"  {edge_type}: {count}")

    serialize_graph(cfg, args.output, args.format)

    if failures:
        report_path = args.output.parent / f"{args.output.stem}.execution_path_failures.yaml"
        with open(report_path, "w", encoding="utf-8") as f:
            yaml.dump({"failures": failures}, f, sort_keys=False, allow_unicode=True, width=float("inf"))
        print(f"Execution path verification failures: {len(failures)}")
        print(f"Wrote failure report to {report_path}")
        if args.fail_on_path_mismatch:
            return 2
    else:
        print("Execution path verification passed")

    print(f"Saved graph to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
