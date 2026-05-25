#!/usr/bin/env python3
"""
Build an execution-instance-level call graph directly from stage 3 inventory files.

This version is inventory-first and preserves the important graph behavior:

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
from pybastion_integration.utils.check_inventory_graph import (
    path_contains_recorded_execution_path,
)
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
        return 10**9
    suffix = ei_id.rsplit("_E", 1)[1]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else 10**9


def entry_ei_for_context(context: CallableContext) -> str:
    execution_items = context_execution_items(context)
    if not execution_items:
        raise ValueError(f"Callable {context.callable_id} has no execution items")
    return min((item["id"] for item in execution_items), key=parse_ei_num)


def _decorator_name(decorator: dict[str, Any]) -> str:
    return str((decorator or {}).get("name", "")).strip()


def is_collapsible_operation_entry(
    entry: dict[str, Any],
) -> tuple[bool, str | None, str | None]:
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
        value.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("|", "_")
    )


def analysis_info(entry: dict[str, Any]) -> dict[str, Any]:
    value = entry.get("analysis_info") or {}
    return value if isinstance(value, dict) else {}


def context_execution_items(context: CallableContext) -> list[dict[str, Any]]:
    """Return the current Stage 3 execution items for a callable context."""
    items = analysis_info(context.entry).get("execution_items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def line_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def execution_item_ids_by_line(context: CallableContext) -> dict[int, list[str]]:
    by_line: dict[int, list[str]] = defaultdict(list)

    for execution_item in context_execution_items(context):
        ei_id = execution_item.get("id")
        line = line_number(execution_item.get("line"))
        if ei_id and line is not None:
            by_line[line].append(str(ei_id))

    return dict(by_line)


def execution_item_ids_in_line_range(
    context: CallableContext,
    *,
    start_line: int | None,
    end_line: int | None,
) -> list[str]:
    if start_line is None:
        return []

    resolved_end_line = end_line if end_line is not None else start_line
    matches: list[str] = []

    for ei in context_execution_items(context):
        line = line_number(ei.get("line"))
        ei_id = ei.get("id")
        if line is None or not ei_id:
            continue

        if start_line <= line <= resolved_end_line:
            matches.append(str(ei_id))

    return matches


def execution_item_ids_for_region(
    region: dict[str, Any],
    context: CallableContext,
) -> list[str]:
    start_line = line_number(region.get("start_line"))
    end_line = line_number(region.get("end_line"))

    return execution_item_ids_in_line_range(
        context,
        start_line=start_line,
        end_line=end_line,
    )


def callable_node_id(context: CallableContext) -> str:
    return f"callable::{context.callable_id}"


def control_region_node_id(context: CallableContext, region_id: str) -> str:
    return f"{context.callable_id}::control_region::{region_id}"


def control_policy_node_id(
    context: CallableContext, index: int, policy: dict[str, Any]
) -> str:
    owner_id = safe_node_id_part(str(policy.get("owner_id") or "unknown"))
    mechanism = safe_node_id_part(str(policy.get("mechanism_kind") or "unknown"))
    return f"{context.callable_id}::control_policy::{owner_id}::{mechanism}::{index}"


def control_line_target_node_id(context: CallableContext, route: dict[str, Any]) -> str:
    route_id = safe_node_id_part(str(route.get("id") or "unknown"))
    target_line = safe_node_id_part(str(route.get("target_line") or "unknown"))
    return f"{context.callable_id}::control_line_target::{target_line}::{route_id}"


def context_control_flow(context: CallableContext) -> dict[str, Any]:
    control_flow = analysis_info(context.entry).get("control_flow") or {}
    return control_flow if isinstance(control_flow, dict) else {}


def add_callable_container_node(cfg: nx.DiGraph, context: CallableContext) -> None:
    control_flow = context_control_flow(context)

    cfg.add_node(
        callable_node_id(context),
        category="callable",
        callable_id=context.callable_id,
        callable_name=context.callable_name,
        callable_kind=context.callable_kind,
        callable_fqn=context.callable_fqn,
        unit=context.unit_name,
        unit_fqn=context.unit_fqn,
        has_control_flow=bool(control_flow),
        control_flow_summary={
            "regions": len(control_flow.get("regions", []) or []),
            "routes": len(control_flow.get("routes", []) or []),
            "policies": len(control_flow.get("policies", []) or []),
        },
    )

    for item in context_execution_items(context):
        ei_id = item.get("id")
        if not ei_id:
            continue

        cfg.add_edge(
            callable_node_id(context),
            str(ei_id),
            edge_type="owns_execution_item",
            callable_id=context.callable_id,
        )


def add_control_flow_nodes_and_edges(cfg: nx.DiGraph, context: CallableContext) -> None:
    """
    Structural control-flow layer.

    Adds callable-owned control-region and post-execution-policy nodes, then adds
    route edges from the current Stage 3 control-flow model. When a route targets
    a source line that maps to execution item nodes, the route is wired directly
    to those execution items. Otherwise, the route falls back to a synthetic line
    target or terminal target node.
    """
    control_flow = context_control_flow(context)
    if not control_flow:
        return

    regions = [
        region
        for region in control_flow.get("regions", []) or []
        if isinstance(region, dict) and region.get("id")
    ]
    routes = [
        route
        for route in control_flow.get("routes", []) or []
        if isinstance(route, dict) and route.get("id")
    ]
    policies = [
        policy
        for policy in control_flow.get("policies", []) or []
        if isinstance(policy, dict)
    ]

    region_ids = {str(region["id"]) for region in regions}
    regions_by_id = {str(region["id"]): region for region in regions}
    execution_item_ids_by_region_id = {
        region_id: execution_item_ids_for_region(region, context)
        for region_id, region in regions_by_id.items()
    }
    ei_ids_by_line = execution_item_ids_by_line(context)

    for region in regions:
        region_id = str(region["id"])
        node_id = control_region_node_id(context, region_id)

        cfg.add_node(
            node_id,
            category="control_region",
            callable_id=context.callable_id,
            callable_fqn=context.callable_fqn,
            region_id=region_id,
            owner_id=region.get("owner_id"),
            region_kind=region.get("kind"),
            source_construct=region.get("source_construct"),
            start_line=region.get("start_line"),
            end_line=region.get("end_line"),
            ordinal=region.get("ordinal"),
            metadata=region.get("metadata") or {},
        )

        cfg.add_edge(
            callable_node_id(context),
            node_id,
            edge_type="owns_control_region",
            callable_id=context.callable_id,
        )

        region_start_line = line_number(region.get("start_line"))
        region_end_line = line_number(region.get("end_line"))

        for ei_id in execution_item_ids_in_line_range(
            context,
            start_line=region_start_line,
            end_line=region_end_line,
        ):
            cfg.add_edge(
                node_id,
                ei_id,
                edge_type="control_region_contains_execution_item",
                callable_id=context.callable_id,
                region_id=region_id,
            )

            cfg.add_edge(
                ei_id,
                node_id,
                edge_type="execution_item_in_control_region",
                callable_id=context.callable_id,
                region_id=region_id,
            )

    for index, policy in enumerate(policies):
        node_id = control_policy_node_id(context, index, policy)
        cfg.add_node(
            node_id,
            category="post_execution_policy",
            callable_id=context.callable_id,
            callable_fqn=context.callable_fqn,
            owner_id=policy.get("owner_id"),
            mechanism_kind=policy.get("mechanism_kind"),
            binding_mode=policy.get("binding_mode"),
            binding_scope=policy.get("binding_scope"),
            trigger_event=policy.get("trigger_event"),
            region_id=policy.get("region_id"),
            target_region_id=policy.get("target_region_id"),
            target_line=policy.get("target_line"),
            applies_to=policy.get("applies_to") or [],
            preserves_prior_outcome=policy.get("preserves_prior_outcome"),
            source_construct=policy.get("source_construct"),
            detail=policy.get("detail"),
        )

        cfg.add_edge(
            callable_node_id(context),
            node_id,
            edge_type="owns_post_execution_policy",
            callable_id=context.callable_id,
        )

        region_id = policy.get("region_id")
        if region_id and str(region_id) in region_ids:
            cfg.add_edge(
                node_id,
                control_region_node_id(context, str(region_id)),
                edge_type="policy_applies_to_region",
                callable_id=context.callable_id,
                owner_id=policy.get("owner_id"),
            )

        target_region_id = policy.get("target_region_id")
        if target_region_id and str(target_region_id) in region_ids:
            cfg.add_edge(
                node_id,
                control_region_node_id(context, str(target_region_id)),
                edge_type="policy_targets_region",
                callable_id=context.callable_id,
                owner_id=policy.get("owner_id"),
            )

    for route in routes:
        source_region_id = route.get("source_region_id")
        target_region_id = route.get("target_region_id")
        target_line = line_number(route.get("target_line"))

        source_node: str
        if source_region_id and str(source_region_id) in region_ids:
            source_node = control_region_node_id(context, str(source_region_id))
        else:
            source_node = callable_node_id(context)

        target_nodes: list[str]
        resolved_target_kind: str

        if target_region_id and str(target_region_id) in region_ids:
            target_nodes = [control_region_node_id(context, str(target_region_id))]
            resolved_target_kind = "control_region"
        elif target_line is not None and target_line in ei_ids_by_line:
            target_nodes = ei_ids_by_line[target_line]
            resolved_target_kind = "execution_item"
        elif route.get("target_line") is not None:
            target_node = control_line_target_node_id(context, route)
            if not cfg.has_node(target_node):
                cfg.add_node(
                    target_node,
                    category="control_line_target",
                    callable_id=context.callable_id,
                    callable_fqn=context.callable_fqn,
                    target_line=route.get("target_line"),
                    route_id=route.get("id"),
                    route_kind=route.get("kind"),
                    owner_id=route.get("owner_id"),
                    synthetic=True,
                )
                cfg.add_edge(
                    callable_node_id(context),
                    target_node,
                    edge_type="owns_control_line_target",
                    callable_id=context.callable_id,
                )
            target_nodes = [target_node]
            resolved_target_kind = "line_placeholder"
        else:
            target_node = f"{context.callable_id}::control_terminal::{safe_node_id_part(str(route.get('id')))}"
            if not cfg.has_node(target_node):
                cfg.add_node(
                    target_node,
                    category="control_terminal_target",
                    callable_id=context.callable_id,
                    callable_fqn=context.callable_fqn,
                    route_id=route.get("id"),
                    route_kind=route.get("kind"),
                    owner_id=route.get("owner_id"),
                    exit_kind=route.get("exit_kind"),
                    synthetic=True,
                )
                cfg.add_edge(
                    callable_node_id(context),
                    target_node,
                    edge_type="owns_control_terminal_target",
                    callable_id=context.callable_id,
                )
            target_nodes = [target_node]
            resolved_target_kind = "terminal_placeholder"

        for target_node in target_nodes:
            cfg.add_edge(
                source_node,
                target_node,
                edge_type="control_route",
                callable_id=context.callable_id,
                route_id=route.get("id"),
                route_kind=route.get("kind"),
                owner_id=route.get("owner_id"),
                exit_kind=route.get("exit_kind"),
                condition=route.get("condition"),
                condition_result=route.get("condition_result"),
                implicit=route.get("implicit"),
                synthetic=route.get("synthetic"),
                preserves_prior_outcome=route.get("preserves_prior_outcome"),
                target_line=route.get("target_line"),
                resolved_target_kind=resolved_target_kind,
                metadata=route.get("metadata") or {},
            )

        if (
            source_region_id
            and target_region_id
            and str(source_region_id) in execution_item_ids_by_region_id
            and str(target_region_id) in execution_item_ids_by_region_id
        ):
            source_eis = execution_item_ids_by_region_id[str(source_region_id)]
            target_eis = execution_item_ids_by_region_id[str(target_region_id)]

            for source_ei in source_eis:
                for target_ei in target_eis:
                    cfg.add_edge(
                        source_ei,
                        target_ei,
                        edge_type="derived_control_route_execution_item",
                        callable_id=context.callable_id,
                        route_id=route.get("id"),
                        route_kind=route.get("kind"),
                        owner_id=route.get("owner_id"),
                        source_region_id=str(source_region_id),
                        target_region_id=str(target_region_id),
                    )

        if (
            source_region_id
            and resolved_target_kind == "execution_item"
            and str(source_region_id) in execution_item_ids_by_region_id
        ):
            source_eis = execution_item_ids_by_region_id[str(source_region_id)]

            for source_ei in source_eis:
                for target_ei in target_nodes:
                    cfg.add_edge(
                        source_ei,
                        target_ei,
                        edge_type="derived_control_route_execution_item",
                        callable_id=context.callable_id,
                        route_id=route.get("id"),
                        route_kind=route.get("kind"),
                        owner_id=route.get("owner_id"),
                        source_region_id=str(source_region_id),
                        target_line=route.get("target_line"),
                        resolved_target_kind="execution_item",
                    )

        if (
            source_region_id
            and resolved_target_kind == "terminal_placeholder"
            and str(source_region_id) in execution_item_ids_by_region_id
        ):
            source_eis = execution_item_ids_by_region_id[str(source_region_id)]

            for source_ei in source_eis:
                for target_node in target_nodes:
                    cfg.add_edge(
                        source_ei,
                        target_node,
                        edge_type="derived_control_route_execution_item_terminal",
                        callable_id=context.callable_id,
                        route_id=route.get("id"),
                        route_kind=route.get("kind"),
                        owner_id=route.get("owner_id"),
                        source_region_id=str(source_region_id),
                        exit_kind=route.get("exit_kind"),
                        resolved_target_kind="terminal_placeholder",
                    )


def inventory_successor_targets(execution_item: dict[str, Any]) -> list[str]:
    targets: list[str] = []

    def add_target(target_ei: Any, is_terminal: Any) -> None:
        if target_ei and not is_terminal:
            targets.append(str(target_ei))

    statement_outcome = execution_item.get("statement_outcome") or {}
    if isinstance(statement_outcome, dict):
        add_target(
            statement_outcome.get("target_ei"),
            statement_outcome.get("is_terminal"),
        )

    for conditional in execution_item.get("conditional_targets", []) or []:
        if not isinstance(conditional, dict):
            continue

        add_target(
            conditional.get("target_ei"),
            conditional.get("is_terminal"),
        )

    for disruptive in execution_item.get("disruptive_outcomes", []) or []:
        if not isinstance(disruptive, dict):
            continue

        add_target(
            disruptive.get("target_ei"),
            disruptive.get("is_terminal"),
        )

    return list(dict.fromkeys(targets))


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

    for execution_item in context_execution_items(context):
        ei_id = execution_item["id"]
        statement_outcome = execution_item.get("statement_outcome") or {}
        ei_decorators = execution_item.get("decorators", []) or []
        merged_decorators = [*callable_decorators, *ei_decorators]
        integration_candidates = context.integration_by_ei.get(ei_id, [])
        integration_candidate = (
            integration_candidates[0] if integration_candidates else {}
        )

        integration_info: dict[str, Any] = {}
        if integration_candidate:
            integration_info = {
                "is_integration_point": True,
                "integration_kind": integration_candidate.get("kind"),
                "integration_target": integration_candidate.get("target"),
                "integration_resolved_target": integration_candidate.get(
                    "resolved_target"
                ),
                "integration_signature": integration_candidate.get("signature"),
                "integration_classification": integration_candidate.get(
                    "classification"
                ),
                "integration_resolution_kind": integration_candidate.get(
                    "resolution_kind"
                ),
                "integration_suppressed_by": integration_candidate.get("suppressed_by"),
                "execution_paths": integration_candidate.get("execution_paths", [])
                or [],
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
            line=execution_item.get("line"),
            stmt_type=execution_item.get("stmt_type"),
            condition=execution_item.get("condition"),
            description=execution_item.get("description"),
            owner_info=execution_item.get("owner_info"),
            constraint=execution_item.get("constraint"),
            statement_outcome=statement_outcome,
            decorators=merged_decorators,
            callable_decorators=callable_decorators,
            ei_decorators=ei_decorators,
            is_terminal=bool(statement_outcome.get("is_terminal", False)),
            terminates_via=statement_outcome.get("terminates_via"),
            **integration_info,
        )


def add_explicit_within_callable_edges(
    cfg: nx.DiGraph,
    context: CallableContext,
    modeled_call_site_eis: set[str],
) -> None:
    edge_type, conditional_edge_type, disruptive_edge_type = (
        "diagnostic_statement_outcome",
        "diagnostic_conditional_target",
        "diagnostic_disruptive_outcome",
    )

    for ei in context_execution_items(context):
        src = ei["id"]

        if src in modeled_call_site_eis:
            continue

        statement_outcome = ei.get("statement_outcome") or {}
        target_ei = statement_outcome.get("target_ei")
        if target_ei and not statement_outcome.get("is_terminal"):
            cfg.add_edge(
                src,
                target_ei,
                edge_type=edge_type,
                within_callable=True,
                callable_id=context.callable_id,
            )

        for conditional in ei.get("conditional_targets", []) or []:
            target_ei = conditional.get("target_ei")
            if target_ei and not conditional.get("is_terminal"):
                cfg.add_edge(
                    src,
                    target_ei,
                    edge_type=conditional_edge_type,
                    within_callable=True,
                    callable_id=context.callable_id,
                    target_condition=conditional.get("target_condition"),
                    condition_result=conditional.get("condition_result"),
                    target_hint=conditional.get("target_hint"),
                )

        for disruptive in ei.get("disruptive_outcomes", []) or []:
            target_ei = disruptive.get("target_ei")
            if target_ei and not disruptive.get("is_terminal"):
                cfg.add_edge(
                    src,
                    target_ei,
                    edge_type=disruptive_edge_type,
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


def classify_call_site(execution_item: dict[str, Any]) -> bool:
    constraint = execution_item.get("constraint") or {}
    return bool(constraint.get("operation_target"))


def operation_target_for_execution_item(execution_item: dict[str, Any]) -> str | None:
    constraint = execution_item.get("constraint") or {}
    return constraint.get("operation_target")


def resolved_target_for_execution_item(execution_item: dict[str, Any]) -> str | None:
    constraint = execution_item.get("constraint") or {}
    resolved_target = constraint.get("resolved_target")

    if resolved_target:
        return str(resolved_target)

    return None


def compute_success_exit_eis(context: CallableContext) -> list[str]:
    exits: list[str] = []
    for ei in context_execution_items(context):
        outcome = ei.get("statement_outcome") or {}
        if (
            outcome.get("is_terminal")
            and outcome.get("terminates_via") in SUCCESS_EXIT_TERMINATORS
        ):
            exits.append(ei["id"])
    return exits


def compute_exception_exit_eis(context: CallableContext) -> list[str]:
    exits: list[str] = []
    for ei in context_execution_items(context):
        outcome = ei.get("statement_outcome") or {}
        if (
            outcome.get("is_terminal")
            and outcome.get("terminates_via") in EXCEPTION_EXIT_TERMINATORS
        ):
            exits.append(ei["id"])
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
        success_exit_cache[target_callable_id] = compute_success_exit_eis(
            target_context
        )
        exception_exit_cache[target_callable_id] = compute_exception_exit_eis(
            target_context
        )

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


def add_collapsed_internal_operation_node(
    ei_id: str,
    resolved_callable_id: str,
    target_context: CallableContext,
    cfg: nx.DiGraph,
    marker_name: str | None = None,
    marker_type: str | None = None,
) -> str:
    """
    Adds a collapsed internal operation node to the call graph for a given execution item.

    Args:
        ei_id (str): The ID of the execution item.
        resolved_callable_id (str): The resolved callable ID for the collapsed operation.
        target_context (CallableContext): The context of the target callable.
        cfg (nx.DiGraph): The call graph to which the node will be added.
        marker_name (str): The name of the marker associated with the collapsed operation.
        marker_type (str): The type of the marker associated with the collapsed operation.
    """
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
    return collapsed_node_id


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

    for ei in context_execution_items(context):
        ei_id = ei["id"]
        integration_candidates = context.integration_by_ei.get(ei_id, [])
        operation_target = operation_target_for_execution_item(ei)
        resolved_target = resolved_target_for_execution_item(ei)

        if not integration_candidates and not operation_target and not resolved_target:
            continue

        return_targets = inventory_successor_targets(ei)

        if not integration_candidates:
            resolved_callable_id, resolved_target_repr = (
                resolve_project_callable_target(
                    target=resolved_target,
                    operation_target=operation_target,
                    context=context,
                    fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                    local_callables_by_unit=local_callables_by_unit,
                )
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
                collapsed_node_id = add_collapsed_internal_operation_node(
                    ei_id,
                    resolved_callable_id,
                    target_context,
                    cfg,
                    marker_name,
                    marker_type,
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
                resolved_callable_id, resolved_target_repr = (
                    resolve_project_callable_target(
                        target=target,
                        operation_target=operation_target,
                        context=context,
                        fqn_to_callable_id=inventory_index.fqn_to_callable_id,
                        local_callables_by_unit=local_callables_by_unit,
                    )
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
                collapse_target, marker_name, marker_type = (
                    is_collapsible_operation_entry(target_context.entry)
                )

                if collapse_target:
                    collapsed_node_id = add_collapsed_internal_operation_node(
                        ei_id,
                        resolved_callable_id,
                        target_context,
                        cfg,
                        marker_name,
                        marker_type,
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

    cfg = nx.MultiDiGraph()

    for context in inventory_index.callable_contexts.values():
        add_callable_container_node(cfg, context)
        add_callable_nodes(cfg, context)
        add_control_flow_nodes_and_edges(cfg, context)

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
            yaml.dump(
                graph_data, f, sort_keys=False, allow_unicode=True, width=float("inf")
            )
        return

    if fmt == "graphml":
        nx.write_graphml(cfg, output_path)
        return

    raise ValueError(f"Unsupported format: {fmt}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build EI call graph directly from inventory files"
    )
    parser.add_argument(
        "--inventories-root",
        type=Path,
        required=True,
        help="Root containing *.inventory.yaml files",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument(
        "--format", choices=["pickle", "yaml", "graphml"], default="pickle"
    )
    parser.add_argument(
        "--fail-on-path-mismatch",
        action="store_true",
        help="Exit nonzero if recorded execution_paths are not realizable",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.inventories_root.exists():
        print(
            f"ERROR: inventories root not found: {args.inventories_root}",
            file=sys.stderr,
        )
        return 1

    inventory_paths = discover_inventory_files(args.inventories_root)
    if not inventory_paths:
        print(
            f"ERROR: no inventory files found under {args.inventories_root}",
            file=sys.stderr,
        )
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
        report_path = (
            args.output.parent / f"{args.output.stem}.execution_path_failures.yaml"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {"failures": failures},
                f,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            )
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
