#!/usr/bin/env python3
"""
Build an execution-instance-level call graph directly from stage 3 inventory files.

This version is inventory-first, but preserves the important behavior from the
older graph builder:

- real project callables are linked whenever they can be resolved
- local same-unit callables are resolved by name when needed
- interunit/project callables are resolved by fully qualified target when needed
- return edges are wired from callee exit EIs back to caller continuations
- recorded execution_paths are verified against the assembled graph
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import sys
import yaml

from pybastion_integration.utils.check_ei_successors import run_preflight
from pybastion_integration.utils.check_inventory_graph import path_contains_recorded_execution_path
from pybastion_integration.utils.inventory_index import (
    CallableContext,
    InventoryIndex,
    discover_inventory_files,
    load_all_inventories,
    signature_info,
)

SUCCESS_EXIT_TERMINATORS: set[str] = {"return", "implicit-return", "yield"}
EXCEPTION_EXIT_TERMINATORS: set[str] = {"raise", "exception"}


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


def safe_node_id_part(value: str | None) -> str:
    if not value:
        return "unknown"
    return (
        value
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("|", "_")
    )


def inventory_successor_targets(branch: dict[str, Any]) -> list[str]:
    outcome = branch.get("statement_outcome") or {}

    if not isinstance(outcome, dict):
        return []

    if outcome.get("is_terminal"):
        return []

    target_ei = outcome.get("target_ei")
    if not target_ei:
        return []

    return [str(target_ei)]


def add_external_placeholder_node(
        cfg: nx.DiGraph,
        *,
        call_site_ei: str,
        candidate: dict[str, Any],
        operation_target: str | None,
        context: CallableContext,
) -> str:
    """
    Adds an external placeholder node to the given directed graph. The node represents
    a call target outside the analyzed code unit, allowing for the modeling of call
    and return edges within the graph. Without this node, there could be no outgoing
    and return edges to represent external calls. That would mean that the call graph
    would be incomplete and unable to accurately model the flow of control and data
    between different parts of the system when external operations are involved.

    Parameters:
    cfg: nx.DiGraph
        The directed graph to which the node will be added.
    call_site_ei: str
        The unique identifier of the calling site from which the external target is
        invoked.
    candidate: dict[str, Any]
        A dictionary containing information about the external call target, such
        as the resolved target, classification, and other integration details.
    operation_target: str | None
        An optional operation target associated with the external call.
    callable_context: CallableContext
        The callable execution context containing metadata such as the current
        callable's ID and fully qualified name.

    Returns:
    str
        The unique identifier of the newly added or already existing external placeholder
        node.
    """
    kind = classify_integration_target_kind(candidate)
    target = (
            candidate.get("resolved_target")
            or candidate.get("target")
            or operation_target
            or "unknown"
    )
    node_id = (
        f"placeholder::{call_site_ei}"
        f"::{safe_node_id_part(kind)}"
        f"::{safe_node_id_part(target)}"
    )

    if not cfg.has_node(node_id):
        cfg.add_node(
            node_id,
            category="external_placeholder",
            placeholder=True,
            reason=(
                "Represents an outbound integration target that is not modeled "
                "as an analyzed project callable."
            ),
            external_kind=kind,
            target=target,
            operation_target=operation_target,
            signature=candidate.get("signature"),
            called_from=call_site_ei,
            source_callable_id=context.callable_id,
            source_callable_fqn=context.callable_fqn,
            integration_kind=kind,
            integration_target=candidate.get("target"),
            integration_resolved_target=candidate.get("resolved_target"),
            integration_signature=candidate.get("signature"),
            integration_classification=candidate.get("classification"),
            integration_resolution_kind=candidate.get("resolution_kind"),
            integration_suppressed_by=candidate.get("suppressed_by"),
        )

    return node_id


def add_callable_nodes(cfg: nx.DiGraph, context: CallableContext) -> None:
    callable_decorators = signature_info(context.entry).get("decorators", []) or []

    for branch in context.branches:
        ei_id = branch["id"]
        statement_outcome = branch.get("statement_outcome") or {}
        branch_decorators = branch.get("decorators", []) or []
        merged_decorators = [*callable_decorators, *branch_decorators]
        integration_candidates = context.integration_by_ei.get(ei_id, [])
        integration_candidate = integration_candidates[0] if integration_candidates else {}

        integration_info: dict[str, Any] = {}
        if integration_candidate:
            integration_info = {
                "is_integration_point": True,
                "integration_kind": integration_candidate.get("kind"),
                "integration_target": integration_candidate.get("target"),
                "integration_resolved_target": integration_candidate.get("resolved_target"),
                "integration_signature": integration_candidate.get("signature"),
                "integration_classification": integration_candidate.get("classification"),
                "integration_resolution_kind": integration_candidate.get("resolution_kind"),
                "integration_suppressed_by": integration_candidate.get("suppressed_by"),
                "execution_paths": integration_candidate.get("execution_paths", []) or [],
                "path_analysis": integration_candidate.get("path_analysis"),
            }

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
            **integration_info,
        )


def add_explicit_within_callable_edges(
        cfg: nx.DiGraph,
        context: CallableContext,
        modeled_call_site_eis: set[str],
) -> None:
    for branch in context.branches:
        src = branch["id"]

        if src in modeled_call_site_eis:
            continue

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


def add_call_to_placeholder_with_return(
        cfg: nx.DiGraph,
        *,
        call_site_ei: str,
        return_targets: list[str],
        placeholder_node_id: str,
        candidate: dict[str, Any],
        operation_target: str | None,
) -> None:
    kind = classify_integration_target_kind(candidate)
    target = candidate.get("resolved_target") or candidate.get("target")

    cfg.add_edge(
        call_site_ei,
        placeholder_node_id,
        edge_type="call",
        call_kind=kind,
        target=target,
        operation_target=operation_target,
        signature=candidate.get("signature"),
        placeholder_target=True,
    )

    for return_target in return_targets:
        cfg.add_edge(
            placeholder_node_id,
            return_target,
            edge_type="return",
            return_kind="external_placeholder",
            original_call_site=call_site_ei,
        )


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
) -> set[str]:
    modeled_call_site_eis: set[str] = set()
    success_exit_cache: dict[str, list[str]] = {}
    exception_exit_cache: dict[str, list[str]] = {}

    for branch in context.branches:
        ei_id = branch["id"]
        integration_candidates = context.integration_by_ei.get(ei_id, [])
        operation_target = operation_target_for_branch(branch)

        if not integration_candidates and not operation_target:
            continue

        return_targets = inventory_successor_targets(branch)

        if not integration_candidates:
            resolved_callable_id, resolved_target_repr = resolve_project_callable_target(
                target=None,
                operation_target=operation_target,
                context=context,
                fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                local_callables_by_unit=local_callables_by_unit,
            )

            if resolved_callable_id is None:
                resolved_callable_id, resolved_target_repr = (
                    resolve_collapsible_decorated_target(
                        operation_target=operation_target,
                        context=context,
                        callable_contexts=inventory_index.callable_contexts,
                        fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                    )
                )

            if resolved_callable_id is None:
                continue

            target_context = inventory_index.callable_contexts[resolved_callable_id]
            collapse_target, marker_name, marker_type = is_collapsible_operation_entry(
                target_context.entry
            )

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
                    call_kind="local",
                    target_callable_id=resolved_callable_id,
                    target_fqn=target_context.callable_fqn,
                    resolved_target=resolved_target_repr,
                    operation_target=operation_target,
                    marker_name=marker_name,
                    marker_type=marker_type,
                    signature=None,
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

                modeled_call_site_eis.add(ei_id)
                continue

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
            modeled_call_site_eis.add(ei_id)
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
                resolved_callable_id, resolved_target_repr = (
                    resolve_collapsible_decorated_target(
                        operation_target=operation_target,
                        context=context,
                        callable_contexts=inventory_index.callable_contexts,
                        fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                    )
                )

            if resolved_callable_id is None and contract_targets:
                for contract_callable_id, contract_target_fqn in contract_targets:
                    target_context = inventory_index.callable_contexts[
                        contract_callable_id
                    ]
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

                modeled_call_site_eis.add(ei_id)
                continue

            if resolved_callable_id is not None:
                target_context = inventory_index.callable_contexts[resolved_callable_id]
                collapse_target, marker_name, marker_type = is_collapsible_operation_entry(
                    target_context.entry
                )

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
                        operation_target=operation_target,
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

                    modeled_call_site_eis.add(ei_id)
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
                modeled_call_site_eis.add(ei_id)
                continue

            placeholder_node_id = add_external_placeholder_node(
                cfg,
                call_site_ei=ei_id,
                candidate=candidate,
                operation_target=operation_target,
                context=context,
            )

            add_call_to_placeholder_with_return(
                cfg,
                call_site_ei=ei_id,
                return_targets=return_targets,
                placeholder_node_id=placeholder_node_id,
                candidate=candidate,
                operation_target=operation_target,
            )
            modeled_call_site_eis.add(ei_id)

    return modeled_call_site_eis


def verify_recorded_execution_paths(
        cfg: nx.DiGraph,
        callable_contexts: dict[str, CallableContext],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    for context in callable_contexts.values():
        try:
            entry_ei = entry_ei_for_context(context)
        except ValueError as exc:
            failures.append(
                {
                    "callable_id": context.callable_id,
                    "callable_fqn": context.callable_fqn,
                    "reason": "missing_entry_ei",
                    "message": str(exc),
                }
            )
            continue

        for candidates in context.integration_by_ei.values():
            for candidate in candidates:
                target_ei = candidate.get("ei_id")
                if not target_ei:
                    failures.append(
                        {
                            "callable_id": context.callable_id,
                            "callable_fqn": context.callable_fqn,
                            "integration_id": candidate.get("id"),
                            "reason": "missing_integration_ei_id",
                            "path": None,
                        }
                    )
                    continue

                if not cfg.has_node(entry_ei):
                    failures.append(
                        {
                            "callable_id": context.callable_id,
                            "callable_fqn": context.callable_fqn,
                            "integration_id": candidate.get("id"),
                            "ei_id": target_ei,
                            "reason": "entry_ei_not_in_graph",
                            "entry_ei": entry_ei,
                        }
                    )
                    continue

                if not cfg.has_node(target_ei):
                    failures.append(
                        {
                            "callable_id": context.callable_id,
                            "callable_fqn": context.callable_fqn,
                            "integration_id": candidate.get("id"),
                            "ei_id": target_ei,
                            "reason": "integration_ei_not_in_graph",
                            "entry_ei": entry_ei,
                        }
                    )
                    continue

                for path in candidate.get("execution_paths", []) or []:
                    if not path_contains_recorded_execution_path(
                            cfg=cfg,
                            start_node=entry_ei,
                            target_node=target_ei,
                            recorded_path=path,
                    ):
                        failures.append(
                            {
                                "callable_id": context.callable_id,
                                "callable_fqn": context.callable_fqn,
                                "integration_id": candidate.get("id"),
                                "ei_id": target_ei,
                                "entry_ei": entry_ei,
                                "path": path,
                                "reason": "recorded_path_not_reachable_in_graph",
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

    modeled_call_site_eis: set[str] = set()

    for context in inventory_index.callable_contexts.values():
        modeled_call_site_eis.update(
            add_call_and_return_edges(
                cfg,
                context,
                inventory_index,
                local_callables_by_unit,
                contract_impl_index,
            )
        )

    for context in inventory_index.callable_contexts.values():
        add_explicit_within_callable_edges(
            cfg,
            context,
            modeled_call_site_eis,
        )

    failures = verify_recorded_execution_paths(
        cfg,
        inventory_index.callable_contexts,
    )
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

    preflight_report = args.output.parent / "ei-successor-preflight.yaml"
    preflight_ok = run_preflight(
        inventories_root=args.inventories_root,
        report_path=preflight_report,
        fail_on_error=True,
        verbose=args.verbose,
    )

    if not preflight_ok:
        print(
            f"ERROR: EI successor preflight failed. See report: {preflight_report}",
            file=sys.stderr,
        )
        return 2

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
