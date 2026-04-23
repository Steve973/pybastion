#!/usr/bin/env python3
"""
Build an execution-instance-level call graph directly from stage 3 inventory files.

This version is inventory-first, but preserves the important behavior from the
older graph builder:

- real project callables are linked whenever they can be resolved
- local same-unit callables are resolved by name when needed
- interunit/project callables are resolved by fully qualified target when needed
- external boundary nodes are fallback only when resolution truly fails
- return edges are wired from callee exit EIs back to caller continuations
- recorded execution_paths are verified against the assembled graph
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

CALLABLE_KINDS: set[str] = {"function", "method", "assignment"}
SUCCESS_EXIT_TERMINATORS: set[str] = {"return", "implicit-return", "yield"}
EXCEPTION_EXIT_TERMINATORS: set[str] = {"raise", "exception"}


def signature_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("signature_info", {}) or {}


def hierarchy_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("hierarchy_info", {}) or {}


def analysis_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("analysis_info", {}) or {}


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


@dataclass(slots=True)
class InventoryIndex:
    callable_contexts: dict[str, CallableContext]
    fqn_to_callable_id: dict[str, str]
    ei_to_callable_id: dict[str, str]
    type_hierarchy: dict[str, dict[str, Any]]
    contract_methods: dict[str, list[str]]

    def build_local_callables_by_unit(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = defaultdict(dict)
        for callable_id, context in self.callable_contexts.items():
            result[context.unit_fqn][context.callable_name] = callable_id
        return {unit_fqn: dict(name_map) for unit_fqn, name_map in result.items()}

    def build_contract_impl_index(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)

        for contract_method_fqn, impl_fqns in self.contract_methods.items():
            for impl_fqn in impl_fqns:
                impl_id = self.fqn_to_callable_id.get(impl_fqn)
                if impl_id is not None and impl_id not in result[contract_method_fqn]:
                    result[contract_method_fqn].append(impl_id)

        for callable_id, context in self.callable_contexts.items():
            hinfo = hierarchy_info(context.entry)
            if not hinfo.get("implements_contract_method", False):
                continue

            for overridden_fqn in hinfo.get("overrides", []) or []:
                if callable_id not in result[overridden_fqn]:
                    result[overridden_fqn].append(callable_id)

        return {k: v for k, v in result.items()}


def discover_inventory_files(inventories_root: Path) -> list[Path]:
    inventory_files = list(inventories_root.rglob("*.inventory.yaml"))
    inventory_files.extend(inventories_root.rglob("*_inventory.yaml"))
    return sorted(set(inventory_files))


def normalize_kind_for_callable(kind: str) -> str:
    if kind in {"function", "method", "assignment"}:
        return kind
    return kind


def build_callable_fqn(unit_fqn: str, ancestor_names: list[str], entry_name: str) -> str:
    parts = [unit_fqn, *ancestor_names, entry_name]
    return ".".join(part for part in parts if part)


def index_inventory(inventory: dict[str, Any]) -> InventoryIndex:
    unit_name = inventory["unit"]
    unit_fqn = inventory.get("fully_qualified_name", unit_name)

    callable_contexts: dict[str, CallableContext] = {}
    fqn_to_callable_id: dict[str, str] = {}
    ei_to_callable_id: dict[str, str] = {}

    type_hierarchy = inventory.get("type_hierarchy", {}) or {}
    contract_methods = inventory.get("contract_methods", {}) or {}

    def recurse(entries: list[dict[str, Any]], ancestors: list[str]) -> None:
        for entry in entries:
            kind = entry.get("kind", "unknown")
            name = entry.get("name", "unknown")
            entry_id = entry["id"]
            children = entry.get("children", []) or []

            ainfo = analysis_info(entry)
            branches = ainfo.get("branches", []) or []
            integration_candidates = ainfo.get("integration_candidates", []) or []

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

    return InventoryIndex(
        callable_contexts=callable_contexts,
        fqn_to_callable_id=fqn_to_callable_id,
        ei_to_callable_id=ei_to_callable_id,
        type_hierarchy=type_hierarchy,
        contract_methods=contract_methods,
    )


def load_all_inventories(inventory_paths: list[Path]) -> InventoryIndex:
    merged = InventoryIndex(
        callable_contexts={},
        fqn_to_callable_id={},
        ei_to_callable_id={},
        type_hierarchy={},
        contract_methods={},
    )

    for path in inventory_paths:
        with open(path, "r", encoding="utf-8") as f:
            inventory = yaml.safe_load(f)
        if not inventory:
            continue

        indexed = index_inventory(inventory)

        merged.callable_contexts.update(indexed.callable_contexts)
        merged.fqn_to_callable_id.update(indexed.fqn_to_callable_id)
        merged.ei_to_callable_id.update(indexed.ei_to_callable_id)
        merged.type_hierarchy.update(indexed.type_hierarchy)

        for contract_method_fqn, impl_fqns in indexed.contract_methods.items():
            bucket = merged.contract_methods.setdefault(contract_method_fqn, [])
            for impl_fqn in impl_fqns:
                if impl_fqn not in bucket:
                    bucket.append(impl_fqn)

    return merged


def parse_ei_num(ei_id: str) -> int:
    if "_E" not in ei_id:
        return 10 ** 9
    suffix = ei_id.rsplit("_E", 1)[1]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else 10 ** 9


def entry_ei_for_context(context: CallableContext) -> str:
    if not context.branches:
        raise ValueError(f"Callable {context.callable_id} has no branches")
    return min((branch["id"] for branch in context.branches), key=parse_ei_num)


def _decorator_name(decorator: dict[str, Any]) -> str:
    return str((decorator or {}).get("name", "")).strip()


def is_collapsible_operation_entry(entry: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    decorators = signature_info(entry).get("decorators", []) or []
    for decorator in decorators:
        name = _decorator_name(decorator)
        if name in {"MechanicalOperation", "UtilityOperation"}:
            kwargs = decorator.get("kwargs") or {}
            op_type = kwargs.get("type")
            return True, name, op_type
    return False, None, None


def add_callable_nodes(cfg: nx.DiGraph, context: CallableContext) -> None:
    callable_decorators = signature_info(context.entry).get("decorators", []) or []

    for branch in context.branches:
        ei_id = branch["id"]
        statement_outcome = branch.get("statement_outcome") or {}
        branch_decorators = branch.get("decorators", []) or []
        merged_decorators = [*callable_decorators, *branch_decorators]

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
            decorators=merged_decorators,
            callable_decorators=callable_decorators,
            branch_decorators=branch_decorators,
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
    return immediate_intra_callable_successors(cfg, context.callable_id, ei_id)


def resolve_project_callable_target(
        target: str | None,
        operation_target: str | None,
        context: CallableContext,
        fqn_to_callable_id: dict[str, str],
        local_callables_by_unit: dict[str, dict[str, str]],
) -> tuple[str | None, str | None]:
    if target and target in fqn_to_callable_id:
        return fqn_to_callable_id[target], target

    if target and "." not in target:
        same_unit_target = f"{context.unit_fqn}.{target}"
        if same_unit_target in fqn_to_callable_id:
            return fqn_to_callable_id[same_unit_target], same_unit_target

    if operation_target and "." not in operation_target:
        local_name_map = local_callables_by_unit.get(context.unit_fqn, {})
        local_callable_id = local_name_map.get(operation_target)
        if local_callable_id is not None:
            resolved_fqn = f"{context.unit_fqn}.{operation_target}"
            return local_callable_id, resolved_fqn

    return None, None


def resolve_contract_dispatch_targets(
        target: str | None,
        inventory_index: InventoryIndex,
        contract_impl_index: dict[str, list[str]],
) -> list[tuple[str, str]]:
    if not target:
        return []

    resolved: list[tuple[str, str]] = []
    for callable_id in contract_impl_index.get(target, []):
        context = inventory_index.callable_contexts.get(callable_id)
        if context is None:
            continue
        resolved.append((callable_id, context.callable_fqn))
    return resolved


def resolve_collapsible_decorated_target(
        operation_target: str | None,
        context: CallableContext,
        callable_contexts: dict[str, CallableContext],
        fqn_to_callable_id: dict[str, str],
) -> tuple[str | None, str | None]:
    if not operation_target:
        return None, None

    parts = operation_target.split(".")
    if len(parts) != 2:
        return None, None

    receiver, attr = parts
    if receiver not in {"self", "cls", "super"}:
        return None, None

    if "." not in context.callable_fqn:
        return None, None

    owner_prefix = context.callable_fqn.rsplit(".", 1)[0]
    candidate_fqn = f"{owner_prefix}.{attr}"
    candidate_id = fqn_to_callable_id.get(candidate_fqn)
    if candidate_id is None:
        return None, None

    candidate_context = callable_contexts.get(candidate_id)
    if candidate_context is None:
        return None, None

    collapse_target, _, _ = is_collapsible_operation_entry(candidate_context.entry)
    if not collapse_target:
        return None, None

    return candidate_id, candidate_fqn


def add_call_to_callable_with_returns(
        cfg: nx.DiGraph,
        *,
        call_site_ei: str,
        return_targets: list[str],
        target_callable_id: str,
        target_context: CallableContext,
        call_kind: str,
        resolved_target_repr: str | None,
        operation_target: str | None,
        signature: str | None,
        success_exit_cache: dict[str, list[str]],
        exception_exit_cache: dict[str, list[str]],
) -> None:
    target_entry_ei = entry_ei_for_context(target_context)

    cfg.add_edge(
        call_site_ei,
        target_entry_ei,
        edge_type="call",
        call_kind=call_kind,
        target_callable_id=target_callable_id,
        target_fqn=target_context.callable_fqn,
        resolved_target=resolved_target_repr,
        operation_target=operation_target,
        signature=signature,
    )

    if target_callable_id not in success_exit_cache:
        success_exit_cache[target_callable_id] = compute_success_exit_eis(target_context)
        exception_exit_cache[target_callable_id] = compute_exception_exit_eis(target_context)

    for return_target in return_targets:
        for exit_ei in success_exit_cache[target_callable_id]:
            cfg.add_edge(
                exit_ei,
                return_target,
                edge_type="return",
                return_kind="success",
                original_call_site=call_site_ei,
                returns_from=target_callable_id,
            )
        for exit_ei in exception_exit_cache[target_callable_id]:
            cfg.add_edge(
                exit_ei,
                return_target,
                edge_type="return",
                return_kind="exception",
                original_call_site=call_site_ei,
                returns_from=target_callable_id,
            )


def add_call_and_return_edges(
        cfg: nx.DiGraph,
        context: CallableContext,
        inventory_index: InventoryIndex,
        local_callables_by_unit: dict[str, dict[str, str]],
        contract_impl_index: dict[str, list[str]],
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
            resolved_callable_id, resolved_target_repr = resolve_project_callable_target(
                target=None,
                operation_target=operation_target,
                context=context,
                fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                local_callables_by_unit=local_callables_by_unit,
            )
            if resolved_callable_id is None:
                resolved_callable_id, resolved_target_repr = resolve_collapsible_decorated_target(
                    operation_target=operation_target,
                    context=context,
                    callable_contexts=inventory_index.callable_contexts,
                    fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                )

            if resolved_callable_id is not None:
                target_context = inventory_index.callable_contexts[resolved_callable_id]
                add_call_to_callable_with_returns(
                    cfg,
                    call_site_ei=ei_id,
                    return_targets=return_targets,
                    target_callable_id=resolved_callable_id,
                    target_context=target_context,
                    call_kind="local",
                    resolved_target_repr=resolved_target_repr,
                    operation_target=operation_target,
                    signature=None,
                    success_exit_cache=success_exit_cache,
                    exception_exit_cache=exception_exit_cache,
                )
                continue

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
            cfg.add_edge(
                ei_id,
                external_node_id,
                edge_type="call",
                call_kind="unresolved",
                operation_target=operation_target,
            )
            for return_target in return_targets:
                cfg.add_edge(
                    external_node_id,
                    return_target,
                    edge_type="return",
                    return_kind="external",
                )
            continue

        for candidate in integration_candidates:
            kind = classify_integration_target_kind(candidate)
            target = candidate.get("resolved_target") or candidate.get("target")
            signature = candidate.get("signature")

            contract_targets = resolve_contract_dispatch_targets(
                target=target,
                inventory_index=inventory_index,
                contract_impl_index=contract_impl_index,
            )

            if contract_targets:
                if len(contract_targets) == 1:
                    resolved_callable_id, resolved_target_repr = contract_targets[0]
                else:
                    resolved_callable_id, resolved_target_repr = None, None
            else:
                resolved_callable_id, resolved_target_repr = resolve_project_callable_target(
                    target=target,
                    operation_target=operation_target,
                    context=context,
                    fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                    local_callables_by_unit=local_callables_by_unit,
                )

            if resolved_callable_id is None and not contract_targets:
                resolved_callable_id, resolved_target_repr = resolve_collapsible_decorated_target(
                    operation_target=operation_target,
                    context=context,
                    callable_contexts=inventory_index.callable_contexts,
                    fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                )

            if resolved_callable_id is None and contract_targets:
                for contract_callable_id, contract_target_fqn in contract_targets:
                    target_context = inventory_index.callable_contexts[contract_callable_id]
                    add_call_to_callable_with_returns(
                        cfg,
                        call_site_ei=ei_id,
                        return_targets=return_targets,
                        target_callable_id=contract_callable_id,
                        target_context=target_context,
                        call_kind=kind,
                        resolved_target_repr=contract_target_fqn,
                        operation_target=operation_target,
                        signature=signature,
                        success_exit_cache=success_exit_cache,
                        exception_exit_cache=exception_exit_cache,
                    )
                continue

            if resolved_callable_id is not None:
                target_context = inventory_index.callable_contexts[resolved_callable_id]
                collapse_target, marker_name, marker_type = is_collapsible_operation_entry(target_context.entry)

                if collapse_target:
                    collapsed_node_id = f"collapsed::{ei_id}::{resolved_callable_id}"
                    if not cfg.has_node(collapsed_node_id):
                        target_sig = signature_info(target_context.entry)
                        cfg.add_node(
                            collapsed_node_id,
                            category="collapsed_internal_operation",
                            callable_id=resolved_callable_id,
                            callable_fqn=target_context.callable_fqn,
                            callable_name=target_context.callable_name,
                            marker_name=marker_name,
                            marker_type=marker_type,
                            decorators=target_sig.get("decorators", []) or [],
                            callable_decorators=target_sig.get("decorators", []) or [],
                            called_from=ei_id,
                        )

                    cfg.add_edge(
                        ei_id,
                        collapsed_node_id,
                        edge_type="call",
                        call_kind=kind,
                        target_callable_id=resolved_callable_id,
                        target_fqn=target_context.callable_fqn,
                        resolved_target=resolved_target_repr,
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
                            returns_from=resolved_callable_id,
                        )
                    continue

                add_call_to_callable_with_returns(
                    cfg,
                    call_site_ei=ei_id,
                    return_targets=return_targets,
                    target_callable_id=resolved_callable_id,
                    target_context=target_context,
                    call_kind=kind,
                    resolved_target_repr=resolved_target_repr,
                    operation_target=operation_target,
                    signature=signature,
                    success_exit_cache=success_exit_cache,
                    exception_exit_cache=exception_exit_cache,
                )
                continue

            external_node_id = f"external::{ei_id}::{kind}::{target or operation_target or 'unknown'}"
            if not cfg.has_node(external_node_id):
                cfg.add_node(
                    external_node_id,
                    category="external_call",
                    external_kind=kind,
                    target=target,
                    operation_target=operation_target,
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
                operation_target=operation_target,
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


def build_graph_from_inventories(
        inventory_paths: list[Path],
) -> tuple[nx.DiGraph, InventoryIndex, list[dict[str, Any]]]:
    inventory_index = load_all_inventories(inventory_paths)
    local_callables_by_unit = inventory_index.build_local_callables_by_unit()
    contract_impl_index = inventory_index.build_contract_impl_index()

    cfg = nx.DiGraph()

    for context in inventory_index.callable_contexts.values():
        add_callable_nodes(cfg, context)

    for context in inventory_index.callable_contexts.values():
        add_explicit_within_callable_edges(cfg, context)

    for context in inventory_index.callable_contexts.values():
        add_call_and_return_edges(
            cfg,
            context,
            inventory_index,
            local_callables_by_unit,
            contract_impl_index,
        )

    failures = verify_recorded_execution_paths(cfg, inventory_index.callable_contexts)
    return cfg, inventory_index, failures


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build EI call graph directly from inventory files")
    parser.add_argument("--inventories-root", type=Path, required=True, help="Root containing *.inventory.yaml files")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument("--format", choices=["pickle", "yaml", "graphml"], default="pickle")
    parser.add_argument(
        "--fail-on-path-mismatch",
        action="store_true",
        help="Exit nonzero if recorded execution_paths are not realizable",
    )
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

    cfg, inventory_index, failures = build_graph_from_inventories(inventory_paths)

    if args.verbose:
        print(f"Callables indexed: {len(inventory_index.callable_contexts)}")
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
