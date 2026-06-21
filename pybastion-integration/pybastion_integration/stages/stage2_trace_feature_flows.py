#!/usr/bin/env python3
"""
Feature tracing helpers.

This module inventories feature-flow markers from PyBastion unit inventory YAML
files.

Feature markers provide feature intent. Execution and control-flow facts come
from inventory EI metadata. When a marker is attached to a StatementAnchor, this
helper resolves the full same-line EI decomposition for the marked statement.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from pybastion_integration.utils.graph_loader import (
    FeatureFlowGraphs,
    load_graph,
)

# =============================================================================
# Marker and EI classification constants
# =============================================================================

FEATURE_MARKER_NAMES: set[str] = {
    "FeatureStart",
    "FeatureTrace",
    "FeatureBranch",
    "FeatureConverge",
    "FeatureEnd",
    "FeatureEndConditional",
}

CONTROL_STMT_TYPES: set[str] = {
    "If",
    "For",
    "AsyncFor",
    "While",
    "Match",
    "Try",
}

CONTROL_CONSTRAINT_TYPES: set[str] = {
    "condition",
    "iteration",
    "match_case",
}

TERMINAL_VIA_VALUES: set[str] = {
    "return",
    "implicit-return",
    "raise",
    "exception",
}

TERMINAL_REPRESENTATIVE_ROUTE_KINDS: set[str] = {
    "loop_continue",
    "loop_break",
}

TERMINAL_REPRESENTATIVE_EXIT_KINDS: set[str] = {
    "continue",
    "break",
}


# =============================================================================
# Feature flow enums
# =============================================================================


class FeatureFlowCaseStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    UNRESOLVED = "unresolved"


class FeatureFlowEndKind(StrEnum):
    FEATURE_END = "feature_end"
    FEATURE_END_CONDITIONAL = "feature_end_conditional"
    UNRESOLVED = "unresolved"


class FeatureFlowOutcomeKind(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


class FeatureFlowUnresolvedReason(StrEnum):
    NO_VALID_CONVERGE_PATH = "no_valid_converge_path"
    NO_VALID_END_PATH = "no_valid_end_path"


class FeaturePathSegmentDispositionReason(StrEnum):
    EXACT_GRAPH_PATH = "exact_graph_path"
    REPAIRED_GRAPH_PATH = "repaired_graph_path"
    INTENTIONAL_SHARED_PATH = "intentional_shared_path"
    PENDING_BRANCH_UNIQUENESS_CHECK = "pending_branch_uniqueness_check"
    NO_GRAPH_PATH = "no_graph_path"
    INVALID_GRAPH_PATH = "invalid_graph_path"
    AMBIGUOUS_REPAIR_CANDIDATES = "ambiguous_repair_candidates"
    COLLAPSED_TO_UNRELATED_BRANCH = "collapsed_to_unrelated_branch"


class FeaturePathSegmentDisposition(StrEnum):
    CANDIDATE = "candidate"
    REVIEW = "review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class FeaturePathSegmentRole(StrEnum):
    START_TO_END = "start_to_end"
    START_TO_BRANCH = "start_to_branch"
    BRANCH_TARGET = "branch_target"
    BRANCH_TO_CONVERGE = "branch_to_converge"
    CONVERGE_TO_BRANCH = "converge_to_branch"
    SHARED_TAIL = "shared_tail"
    TO_END = "to_end"
    TO_CONDITIONAL_END = "to_conditional_end"


# =============================================================================
# Marker inventory models
# =============================================================================


@dataclass(frozen=True)
class EiExecutionMetadata:
    ei_id: str
    line: int | None
    stmt_type: str | None
    description: str | None
    condition: str | None
    statement_outcome: dict[str, Any] | None
    conditional_targets: list[dict[str, Any]]
    disruptive_outcomes: list[dict[str, Any]]
    constraint: dict[str, Any] | None
    is_terminal: bool
    terminates_via: str | None


@dataclass(frozen=True)
class MarkedStatementMetadata:
    entry_ei: EiExecutionMetadata
    eis: list[EiExecutionMetadata]
    terminal_eis: list[EiExecutionMetadata]
    control_eis: list[EiExecutionMetadata]


@dataclass(frozen=True)
class FeatureBranchSelection:
    branch: str | None
    control_polarity: bool | None
    selected_target_eis: list[str]
    selected_conditions: list[str]


@dataclass(frozen=True)
class FeatureMarkerRecord:
    feature_name: str
    marker_name: str
    inventory_path: str
    unit: str | None
    unit_fqn: str | None
    callable_id: str
    callable_name: str | None
    callable_fqn: str | None
    node_id: str
    kwargs: dict[str, Any]
    line: int | None
    stmt_type: str | None
    description: str | None
    condition: str | None
    marker_ei: EiExecutionMetadata
    marked_statement: MarkedStatementMetadata | None
    branch_selection: FeatureBranchSelection | None = None


@dataclass
class FeatureMarkerInventory:
    feature_name: str
    starts: list[FeatureMarkerRecord] = field(default_factory=list)
    traces: list[FeatureMarkerRecord] = field(default_factory=list)
    branches: list[FeatureMarkerRecord] = field(default_factory=list)
    converges: list[FeatureMarkerRecord] = field(default_factory=list)
    ends: list[FeatureMarkerRecord] = field(default_factory=list)
    conditional_ends: list[FeatureMarkerRecord] = field(default_factory=list)

    def all_records(self) -> list[FeatureMarkerRecord]:
        return [
            *self.starts,
            *self.traces,
            *self.branches,
            *self.converges,
            *self.ends,
            *self.conditional_ends,
        ]


# =============================================================================
# Segment/case models for the real branch-aware expansion engine
# =============================================================================


@dataclass(frozen=True)
class FeaturePathSegment:
    feature_name: str
    segment_branch_path: tuple[str, ...]
    start_ei: str
    end_ei: str
    path: list[str]
    role: FeaturePathSegmentRole
    disposition: FeaturePathSegmentDisposition
    disposition_reason: FeaturePathSegmentDispositionReason


@dataclass(frozen=True)
class FeatureFlowCase:
    feature_name: str
    case_branch_path: tuple[str, ...]
    active_branch_path: tuple[str, ...]
    current_eis: tuple[str, ...]
    segments: tuple[FeaturePathSegment, ...]
    status: FeatureFlowCaseStatus = FeatureFlowCaseStatus.ACTIVE
    end_kind: FeatureFlowEndKind | None = None
    outcome_kind: FeatureFlowOutcomeKind | None = None
    end_marker_node_id: str | None = None


@dataclass(frozen=True)
class UnresolvedFeatureFlowCase:
    feature_name: str
    case_branch_path: tuple[str, ...]
    active_branch_path: tuple[str, ...]
    current_eis: tuple[str, ...]
    segments: tuple[FeaturePathSegment, ...]
    reason: FeatureFlowUnresolvedReason
    expected_converge_point: FeatureConvergePoint | None = None
    expected_end_marker: FeatureMarkerRecord | None = None


@dataclass(frozen=True)
class FeatureFlowTraceResult:
    completed_cases: list[FeatureFlowCase]
    unresolved_cases: list[UnresolvedFeatureFlowCase]


@dataclass(frozen=True)
class FeatureBranchPoint:
    feature_name: str
    branch_point_id: str
    marker_node_id: str
    control_ei_id: str
    branch_markers: tuple[FeatureMarkerRecord, ...]


@dataclass(frozen=True)
class FeatureConvergePoint:
    feature_name: str
    converge_point_id: str
    marker_node_id: str
    converge_ei_id: str
    source_branches: tuple[str, ...]
    into_branch: str
    marker: FeatureMarkerRecord


# =============================================================================
# Inventory loading and traversal
# =============================================================================


def load_inventory(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def find_inventory_files(inventory_root: Path) -> list[Path]:
    return sorted(inventory_root.rglob("*.inventory.yaml"))


def iter_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(items: list[dict[str, Any]]) -> None:
        for item in items:
            result.append(item)
            visit(item.get("children", []) or [])

    visit(entries)
    return result


# =============================================================================
# EI metadata extraction
# =============================================================================


def ei_execution_metadata(execution_item: dict[str, Any]) -> EiExecutionMetadata:
    statement_outcome = execution_item.get("statement_outcome")
    disruptive_outcomes = execution_item.get("disruptive_outcomes", []) or []

    terminates_via = execution_item.get("terminates_via")
    is_terminal = bool(execution_item.get("is_terminal", False))

    if statement_outcome:
        terminates_via = terminates_via or statement_outcome.get("terminates_via")
        is_terminal = is_terminal or bool(statement_outcome.get("is_terminal", False))

    for outcome in disruptive_outcomes:
        terminates_via = terminates_via or outcome.get("terminates_via")
        is_terminal = is_terminal or bool(outcome.get("is_terminal", False))

    return EiExecutionMetadata(
        ei_id=str(execution_item.get("id")),
        line=execution_item.get("line"),
        stmt_type=execution_item.get("stmt_type"),
        description=execution_item.get("description"),
        condition=execution_item.get("condition"),
        statement_outcome=statement_outcome,
        conditional_targets=execution_item.get("conditional_targets", []) or [],
        disruptive_outcomes=disruptive_outcomes,
        constraint=execution_item.get("constraint"),
        is_terminal=is_terminal,
        terminates_via=terminates_via,
    )


def parse_control_polarity(value: Any) -> bool | None:
    if value is True or value == "true":
        return True
    if value is False or value == "false":
        return False
    return None


def parse_ei_sort_key(ei_id: str) -> tuple[str, int, str]:
    if "_E" not in ei_id:
        return ei_id, 10**9, ei_id

    prefix, suffix = ei_id.rsplit("_E", 1)
    digits: list[str] = []

    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break

    number = int("".join(digits)) if digits else 10**9
    return prefix, number, ei_id


def is_terminal_ei(ei: EiExecutionMetadata) -> bool:
    if ei.is_terminal:
        return True

    if ei.terminates_via in TERMINAL_VIA_VALUES:
        return True

    if ei.statement_outcome is not None:
        if ei.statement_outcome.get("terminates_via") in TERMINAL_VIA_VALUES:
            return True

    for outcome in ei.disruptive_outcomes:
        if outcome.get("terminates_via") in TERMINAL_VIA_VALUES:
            return True

    if ei.description and ei.description.startswith("raises "):
        return True

    return False


def is_control_ei(ei: EiExecutionMetadata) -> bool:
    if ei.conditional_targets:
        return True

    if ei.constraint is not None:
        if ei.constraint.get("constraint_type") in CONTROL_CONSTRAINT_TYPES:
            return True

    if ei.stmt_type in CONTROL_STMT_TYPES:
        return True

    return False


# =============================================================================
# Marker attachment and marked-statement resolution
# =============================================================================


def statement_anchor_target_ei_id(execution_item: dict[str, Any]) -> str | None:
    if execution_item.get("stmt_type") != "StatementAnchor":
        return None

    statement_outcome = execution_item.get("statement_outcome") or {}
    target_ei = statement_outcome.get("target_ei")

    return str(target_ei) if target_ei else None


def marked_statement_metadata(
    *,
    marker_ei: dict[str, Any],
    execution_items_by_id: dict[str, dict[str, Any]],
    execution_items: list[dict[str, Any]],
) -> MarkedStatementMetadata | None:
    entry_ei_id = statement_anchor_target_ei_id(marker_ei)

    if entry_ei_id is None:
        return None

    entry_ei = execution_items_by_id.get(entry_ei_id)

    if entry_ei is None:
        return None

    entry_ei_metadata = ei_execution_metadata(entry_ei)
    line = entry_ei.get("line")

    statement_eis = [
        ei_execution_metadata(ei)
        for ei in execution_items
        if ei.get("line") == line and ei.get("stmt_type") != "StatementAnchor"
    ]

    statement_eis.sort(key=lambda item: parse_ei_sort_key(item.ei_id))

    terminal_eis = [item for item in statement_eis if is_terminal_ei(item)]

    control_eis = [item for item in statement_eis if is_control_ei(item)]

    return MarkedStatementMetadata(
        entry_ei=entry_ei_metadata,
        eis=statement_eis,
        terminal_eis=terminal_eis,
        control_eis=control_eis,
    )


def add_unique_text(target: list[str], value: Any) -> None:
    if value is None:
        return

    text = str(value)
    if text not in target:
        target.append(text)


def normalize_branch_name(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def try_success_branch_name(value: Any) -> bool:
    normalized = normalize_branch_name(value)
    return normalized in {
        "trysuccess",
        "success",
        "normal",
        "normalcompletion",
        "trynormal",
    }


def text_mentions_branch(text: Any, branch: Any) -> bool:
    normalized_text = normalize_branch_name(text)
    normalized_branch = normalize_branch_name(branch)

    if not normalized_text or not normalized_branch:
        return False

    if normalized_branch in normalized_text:
        return True

    if normalized_branch.endswith("error"):
        exception_name = normalized_branch.removesuffix("error") + "error"
        return exception_name in normalized_text

    return False


def derive_try_branch_targets(
    *,
    branch: Any,
    marked_statement: MarkedStatementMetadata,
) -> tuple[list[str], list[str]]:
    selected_target_eis: list[str] = []
    selected_conditions: list[str] = []

    for control_ei in marked_statement.control_eis:
        if control_ei.stmt_type != "Try":
            continue

        if try_success_branch_name(branch):
            statement_outcome = control_ei.statement_outcome or {}
            target_ei = statement_outcome.get("target_ei")

            if target_ei is not None:
                add_unique_text(selected_target_eis, target_ei)
                add_unique_text(
                    selected_conditions,
                    control_ei.condition
                    or control_ei.description
                    or statement_outcome.get("outcome")
                    or "try_success",
                )

            continue

        for outcome in control_ei.disruptive_outcomes:
            outcome_text = " ".join(
                str(item)
                for item in [
                    control_ei.condition,
                    control_ei.description,
                    outcome.get("outcome"),
                    outcome.get("exception_type"),
                    outcome.get("handler_type"),
                ]
                if item
            )

            if not text_mentions_branch(outcome_text, branch):
                continue

            add_unique_text(selected_target_eis, outcome.get("target_ei"))
            add_unique_text(selected_conditions, outcome_text)

    return selected_target_eis, selected_conditions


def control_route_owner_id(control_ei: EiExecutionMetadata) -> str | None:
    if control_ei.line is None or control_ei.stmt_type is None:
        return None

    stmt_type = control_ei.stmt_type.lower()
    return f"{stmt_type}:{control_ei.line}"


def ei_id_for_route_target_line(
    *,
    execution_items: list[dict[str, Any]],
    target_line: Any,
) -> str | None:
    if target_line is None:
        return None

    try:
        line = int(target_line)
    except (TypeError, ValueError):
        return None

    candidates = [
        item for item in execution_items if item.get("line") == line and item.get("id")
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda item: parse_ei_sort_key(str(item.get("id"))))

    # Route target lines frequently point to a feature marker anchor.  Preserve
    # that anchor so Stage 4 can consume the convergence/branch event instead
    # of jumping past it to the executable statement on the same line.
    for item in candidates:
        if item.get("stmt_type") == "StatementAnchor":
            return str(item.get("id"))

    return str(candidates[0].get("id"))


def route_mentions_branch(route: dict[str, Any], branch: Any) -> bool:
    metadata = route.get("metadata") or {}
    route_text = " ".join(
        str(item)
        for item in [
            route.get("condition"),
            route.get("kind"),
            metadata.get("handler_expr"),
            metadata.get("handler_type"),
        ]
        if item
    )
    return text_mentions_branch(route_text, branch)


def derive_route_branch_targets(
    *,
    branch: Any,
    control_polarity: bool | None,
    marked_statement: MarkedStatementMetadata,
    control_flow_routes: list[dict[str, Any]],
    execution_items: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    selected_target_eis: list[str] = []
    selected_conditions: list[str] = []

    for control_ei in marked_statement.control_eis:
        owner_id = control_route_owner_id(control_ei)
        if owner_id is None:
            continue

        owner_routes = [
            route for route in control_flow_routes if route.get("owner_id") == owner_id
        ]

        if control_ei.stmt_type == "Try":
            if try_success_branch_name(branch):
                # Prefer the first explicit Try EI successor because it enters
                # the protected body and therefore preserves nested feature
                # branches inside a successful try path.  Later same-line Try
                # EIs model else/finally transfer points and must not become
                # alternate branch starts.
                statement_outcome = control_ei.statement_outcome or {}
                target_ei = statement_outcome.get("target_ei")

                if target_ei is not None:
                    add_unique_text(selected_target_eis, target_ei)
                    add_unique_text(
                        selected_conditions,
                        statement_outcome.get("outcome") or control_ei.description,
                    )
                    break

            for route in owner_routes:
                if route.get("kind") != "handler_match":
                    continue

                if not route_mentions_branch(route, branch):
                    continue

                target_ei = ei_id_for_route_target_line(
                    execution_items=execution_items,
                    target_line=route.get("target_line"),
                )
                add_unique_text(selected_target_eis, target_ei)
                add_unique_text(selected_conditions, route.get("condition"))

            continue

        if control_polarity is None:
            continue

        for route in owner_routes:
            if route.get("condition_result") != control_polarity:
                continue

            target_ei = ei_id_for_route_target_line(
                execution_items=execution_items,
                target_line=route.get("target_line"),
            )
            add_unique_text(selected_target_eis, target_ei)
            add_unique_text(selected_conditions, route.get("condition"))

    return selected_target_eis, selected_conditions


def derive_feature_branch_selection(
    *,
    marker_kwargs: dict[str, Any],
    marked_statement: MarkedStatementMetadata | None,
    control_flow_routes: list[dict[str, Any]],
    execution_items: list[dict[str, Any]],
) -> FeatureBranchSelection | None:
    if marked_statement is None:
        return None

    branch = marker_kwargs.get("branch")
    control_polarity = parse_control_polarity(marker_kwargs.get("control_polarity"))

    selected_target_eis: list[str] = []
    selected_conditions: list[str] = []

    route_target_eis, route_conditions = derive_route_branch_targets(
        branch=branch,
        control_polarity=control_polarity,
        marked_statement=marked_statement,
        control_flow_routes=control_flow_routes,
        execution_items=execution_items,
    )

    selected_target_eis.extend(route_target_eis)
    selected_conditions.extend(route_conditions)

    if not selected_target_eis:
        try_target_eis, try_conditions = derive_try_branch_targets(
            branch=branch,
            marked_statement=marked_statement,
        )

        selected_target_eis.extend(try_target_eis)
        selected_conditions.extend(try_conditions)

    if not selected_target_eis and control_polarity is not None:
        for control_ei in marked_statement.control_eis:
            for target in control_ei.conditional_targets:
                if target.get("condition_result") != control_polarity:
                    continue

                add_unique_text(selected_target_eis, target.get("target_ei"))
                add_unique_text(selected_conditions, target.get("target_condition"))

    return FeatureBranchSelection(
        branch=branch,
        control_polarity=control_polarity,
        selected_target_eis=selected_target_eis,
        selected_conditions=selected_conditions,
    )


# =============================================================================
# Feature marker inventory construction
# =============================================================================


def iter_feature_marker_records_for_callable(
    *,
    inventory_path: Path,
    inventory: dict[str, Any],
    entry: dict[str, Any],
) -> list[FeatureMarkerRecord]:
    analysis_info = entry.get("analysis_info", {}) or {}
    execution_items = analysis_info.get("execution_items", []) or []

    if not execution_items:
        return []

    eis_by_id = {str(ei.get("id")): ei for ei in execution_items if ei.get("id")}

    records: list[FeatureMarkerRecord] = []

    for ei in execution_items:
        decorators = ei.get("decorators", []) or []

        if not decorators:
            continue

        marker_ei = ei_execution_metadata(ei)
        marked_statement = marked_statement_metadata(
            marker_ei=ei,
            execution_items_by_id=eis_by_id,
            execution_items=execution_items,
        )

        for decorator in decorators:
            if not isinstance(decorator, dict):
                continue

            marker_name: str = decorator.get("name", "")
            if marker_name not in FEATURE_MARKER_NAMES:
                continue

            kwargs = decorator.get("kwargs", {}) or {}
            if not isinstance(kwargs, dict):
                kwargs = {}

            feature_name: str = kwargs.get("name", "")
            if not feature_name:
                continue

            branch_selection = (
                derive_feature_branch_selection(
                    marker_kwargs=kwargs,
                    marked_statement=marked_statement,
                    control_flow_routes=(
                        (analysis_info.get("control_flow") or {}).get("routes", [])
                        or []
                    ),
                    execution_items=execution_items,
                )
                if marker_name == "FeatureBranch"
                else None
            )

            records.append(
                FeatureMarkerRecord(
                    feature_name=feature_name,
                    marker_name=marker_name,
                    inventory_path=str(inventory_path),
                    unit=inventory.get("unit"),
                    unit_fqn=inventory.get("fully_qualified_name"),
                    callable_id=str(entry.get("id")),
                    callable_name=entry.get("name"),
                    callable_fqn=entry.get("_fqn"),
                    node_id=str(ei.get("id")),
                    kwargs=kwargs,
                    line=ei.get("line"),
                    stmt_type=ei.get("stmt_type"),
                    description=ei.get("description"),
                    condition=ei.get("condition"),
                    marker_ei=marker_ei,
                    marked_statement=marked_statement,
                    branch_selection=branch_selection,
                )
            )

    return records


def iter_feature_marker_records_from_inventory(
    inventory_path: Path,
) -> list[FeatureMarkerRecord]:
    inventory = load_inventory(inventory_path)
    entries = iter_entries(inventory.get("entries", []) or [])

    records: list[FeatureMarkerRecord] = []

    for entry in entries:
        records.extend(
            iter_feature_marker_records_for_callable(
                inventory_path=inventory_path,
                inventory=inventory,
                entry=entry,
            )
        )

    return records


def build_feature_marker_inventory_from_paths(
    inventory_paths: list[Path],
) -> dict[str, FeatureMarkerInventory]:
    inventories: dict[str, FeatureMarkerInventory] = {}

    for inventory_path in inventory_paths:
        for record in iter_feature_marker_records_from_inventory(inventory_path):
            inventory = inventories.setdefault(
                record.feature_name,
                FeatureMarkerInventory(feature_name=record.feature_name),
            )

            match record.marker_name:
                case "FeatureStart":
                    inventory.starts.append(record)
                case "FeatureTrace":
                    inventory.traces.append(record)
                case "FeatureBranch":
                    inventory.branches.append(record)
                case "FeatureConverge":
                    inventory.converges.append(record)
                case "FeatureEnd":
                    inventory.ends.append(record)
                case "FeatureEndConditional":
                    inventory.conditional_ends.append(record)

    return inventories


def feature_marker_inventory_to_dict(
    inventories: dict[str, FeatureMarkerInventory],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []

    for feature_name, inventory in sorted(inventories.items()):
        features.append(
            {
                "feature_name": feature_name,
                "summary": {
                    "starts": len(inventory.starts),
                    "traces": len(inventory.traces),
                    "branches": len(inventory.branches),
                    "converges": len(inventory.converges),
                    "ends": len(inventory.ends),
                    "conditional_ends": len(inventory.conditional_ends),
                    "total_markers": len(inventory.all_records()),
                },
                "markers": {
                    "starts": [asdict(record) for record in inventory.starts],
                    "traces": [asdict(record) for record in inventory.traces],
                    "branches": [asdict(record) for record in inventory.branches],
                    "converges": [asdict(record) for record in inventory.converges],
                    "ends": [asdict(record) for record in inventory.ends],
                    "conditional_ends": [
                        asdict(record) for record in inventory.conditional_ends
                    ],
                },
            }
        )

    return {
        "feature_count": len(features),
        "features": features,
    }


# =============================================================================
# Graph path helpers
# =============================================================================


def append_path(base: list[str], addition: list[str]) -> list[str]:
    if not base:
        return list(addition)
    if not addition:
        return base
    if base[-1] == addition[0]:
        return [*base, *addition[1:]]
    return [*base, *addition]


class SegmentPathSearchMode(StrEnum):
    FIRST = "first"
    SHORTEST = "shortest"
    ALL = "all"
    LIMITED = "limited"


def node_callable_name(
    cfg: nx.MultiDiGraph,
    node_id: str,
) -> str | None:
    return cfg.nodes.get(node_id, {}).get("callable_name")


def case_callable_name(
    cfg: nx.MultiDiGraph,
    case: FeatureFlowCase,
) -> str | None:
    for current_ei in case.current_eis:
        callable_name = node_callable_name(cfg, current_ei)

        if callable_name is not None:
            return callable_name

    return None


def is_statement_anchor_node(
    cfg: nx.MultiDiGraph,
    node_id: str,
) -> bool:
    node_data = cfg.nodes.get(node_id, {})

    if node_data.get("stmt_type") == "StatementAnchor":
        return True

    if node_data.get("condition") == "statement anchor":
        return True

    if node_data.get("description") == "statement anchor":
        return True

    return False


def node_has_feature_marker(
    cfg: nx.MultiDiGraph,
    node_id: str,
) -> bool:
    decorators = cfg.nodes.get(node_id, {}).get("decorators", []) or []

    for decorator in decorators:
        if not isinstance(decorator, dict):
            continue

        if decorator.get("name") in FEATURE_MARKER_NAMES:
            return True

    return False


def marker_callables_for_path(
    cfg: nx.MultiDiGraph,
    path: list[str],
) -> set[str]:
    result: set[str] = set()

    for node_id in path:
        if not node_has_feature_marker(cfg, node_id):
            continue

        callable_name = node_callable_name(cfg, node_id)
        if callable_name is not None:
            result.add(callable_name)

    return result


NAVIGATION_EDGE_TYPES: set[str] = {
    "diagnostic_statement_outcome",
    "diagnostic_conditional_target",
    "derived_control_route_execution_item",
    "derived_control_route_execution_item_terminal",
    "call",
    "return",
}


def has_navigation_edge(
    cfg: nx.MultiDiGraph,
    source: str,
    target: str,
) -> bool:
    edge_data = cfg.get_edge_data(source, target)

    if not edge_data:
        return False

    # MultiDiGraph returns a key->attrs mapping. DiGraph-like test doubles may
    # return the attrs directly, so support both shapes.
    if isinstance(edge_data, dict) and "edge_type" in edge_data:
        return edge_data.get("edge_type") in NAVIGATION_EDGE_TYPES

    if isinstance(edge_data, dict):
        for attrs in edge_data.values():
            if not isinstance(attrs, dict):
                continue

            if attrs.get("edge_type") in NAVIGATION_EDGE_TYPES:
                return True

    return False


def graph_path_rejection_reason(
    cfg: nx.MultiDiGraph,
    path: list[str],
) -> str | None:
    if not path:
        return "empty_path"

    if len(path) == 1:
        if path[0] not in cfg:
            return "single_node_missing_from_graph"

        return None

    missing_edges = [
        (source, target)
        for source, target in zip(path, path[1:])
        if not cfg.has_edge(source, target)
    ]

    if missing_edges:
        return f"missing_edges={missing_edges}"

    non_navigation_edges = [
        (source, target)
        for source, target in zip(path, path[1:])
        if not has_navigation_edge(cfg, source, target)
    ]

    if non_navigation_edges:
        return f"non_navigation_edges={non_navigation_edges}"

    return None


def is_navigation_valid_graph_path(
    cfg: nx.MultiDiGraph,
    path: list[str],
) -> bool:
    if graph_path_rejection_reason(cfg, path) is not None:
        return False

    source_callable = node_callable_name(cfg, path[0])
    target_callable = node_callable_name(cfg, path[-1])

    if source_callable is None or target_callable is None:
        return True

    if source_callable != target_callable:
        return True

    marker_callables = marker_callables_for_path(cfg, path)
    marker_callables.discard(source_callable)

    return not marker_callables


def find_valid_segment_paths(
    graphs: FeatureFlowGraphs,
    *,
    sources: list[str],
    targets: list[str],
    mode: SegmentPathSearchMode = SegmentPathSearchMode.SHORTEST,
    max_paths: int | None = None,
    cutoff: int | None = None,
    forbidden_nodes: set[str] | None = None,
    max_shortest_candidates: int = 25,
) -> list[list[str]]:
    paths: list[list[str]] = []
    forbidden_nodes = forbidden_nodes or set()
    target_set = set(targets)

    def is_allowed_path(path: list[str]) -> bool:
        if cutoff is not None and len(path) > cutoff:
            return False

        blocked_nodes = forbidden_nodes - target_set
        interior_nodes = set(path[1:-1])

        if interior_nodes & blocked_nodes:
            return False

        return is_navigation_valid_graph_path(graphs.cfg, path)

    def iter_shortest_candidate_paths(
        source: str,
        target: str,
    ) -> Iterable[list[str]]:
        try:
            candidate_paths = nx.shortest_simple_paths(
                graphs.cfg_nav,
                source,
                target,
            )

            for index, candidate_path in enumerate(candidate_paths):
                if index >= max_shortest_candidates:
                    break

                yield candidate_path

        except nx.NetworkXNoPath:
            return
        except nx.NodeNotFound:
            return

    for source in sources:
        if source not in graphs.cfg:
            continue

        for target in targets:
            if target not in graphs.cfg:
                continue

            if mode == SegmentPathSearchMode.SHORTEST:
                for candidate_path in iter_shortest_candidate_paths(
                    source,
                    target,
                ):
                    if not is_allowed_path(candidate_path):
                        continue

                    paths.append(candidate_path)
                    break

                continue

            try:
                candidate_paths = nx.all_simple_paths(
                    graphs.cfg_nav,
                    source,
                    target,
                    cutoff=cutoff,
                )
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue

            for path in candidate_paths:
                if not is_allowed_path(path):
                    continue

                paths.append(path)

                if mode == SegmentPathSearchMode.FIRST:
                    return paths

                if mode == SegmentPathSearchMode.LIMITED:
                    if max_paths is not None and len(paths) >= max_paths:
                        return paths

    if mode == SegmentPathSearchMode.SHORTEST:
        paths.sort(key=len)

        if paths:
            return [paths[0]]

    return paths


def trace_feature_path_segment(
    graphs: FeatureFlowGraphs,
    *,
    feature_name: str,
    segment_branch_path: tuple[str, ...],
    start_eis: list[str],
    end_eis: list[str],
    role: FeaturePathSegmentRole,
    disposition: FeaturePathSegmentDisposition,
    disposition_reason: FeaturePathSegmentDispositionReason,
    forbidden_nodes: set[str] | None = None,
) -> FeaturePathSegment | None:
    paths = find_valid_segment_paths(
        graphs,
        sources=start_eis,
        targets=end_eis,
        mode=SegmentPathSearchMode.SHORTEST,
        forbidden_nodes=forbidden_nodes,
    )

    if not paths:
        return None

    path = paths[0]

    return FeaturePathSegment(
        feature_name=feature_name,
        segment_branch_path=segment_branch_path,
        start_ei=path[0],
        end_ei=path[-1],
        path=path,
        role=role,
        disposition=disposition,
        disposition_reason=disposition_reason,
    )


# =============================================================================
# Marker endpoint helpers
# =============================================================================


def marker_exit_eis(record: FeatureMarkerRecord) -> list[str]:
    if record.marked_statement is not None:
        return [record.marked_statement.entry_ei.ei_id]

    return [record.node_id]


def marker_end_eis(record: FeatureMarkerRecord) -> list[str]:
    if record.marked_statement is None:
        return [record.node_id]

    terminal_eis = [item.ei_id for item in record.marked_statement.terminal_eis]

    if terminal_eis:
        return terminal_eis

    return [record.marked_statement.entry_ei.ei_id]


# =============================================================================
# Segment/case helpers
# =============================================================================

BRANCH_PATH_SEPARATOR = "::"


def branch_path_to_key(branch_path: tuple[str, ...]) -> str:
    return BRANCH_PATH_SEPARATOR.join(branch_path)


def feature_path_segment_id(segment: FeaturePathSegment) -> str:
    branch_path_key = branch_path_to_key(segment.segment_branch_path)

    return (
        f"{segment.feature_name}"
        f"{BRANCH_PATH_SEPARATOR}{branch_path_key}"
        f"{BRANCH_PATH_SEPARATOR}{segment.role.value}"
        f"{BRANCH_PATH_SEPARATOR}{segment.start_ei}"
        f"{BRANCH_PATH_SEPARATOR}{segment.end_ei}"
    )


def initial_feature_flow_case(
    *,
    feature_name: str,
    start_eis: list[str],
) -> FeatureFlowCase:
    return FeatureFlowCase(
        feature_name=feature_name,
        case_branch_path=("main",),
        active_branch_path=("main",),
        current_eis=tuple(start_eis),
        segments=(),
    )


def append_segment_to_case(
    case: FeatureFlowCase,
    segment: FeaturePathSegment,
    *,
    current_eis: list[str],
    active_branch_path: tuple[str, ...] | None = None,
) -> FeatureFlowCase:
    return FeatureFlowCase(
        feature_name=case.feature_name,
        case_branch_path=case.case_branch_path,
        active_branch_path=active_branch_path or case.active_branch_path or (),
        current_eis=tuple(current_eis),
        segments=(*case.segments, segment),
        status=case.status,
        end_kind=case.end_kind,
        outcome_kind=case.outcome_kind,
        end_marker_node_id=case.end_marker_node_id,
    )


def complete_feature_flow_case(
    case: FeatureFlowCase,
    *,
    end_kind: FeatureFlowEndKind,
    outcome_kind: FeatureFlowOutcomeKind,
    current_eis: list[str],
    segment: FeaturePathSegment,
    end_marker_node_id: str,
) -> FeatureFlowCase:
    return FeatureFlowCase(
        feature_name=case.feature_name,
        case_branch_path=case.case_branch_path,
        active_branch_path=case.active_branch_path,
        current_eis=tuple(current_eis),
        segments=(*case.segments, segment),
        status=FeatureFlowCaseStatus.COMPLETED,
        end_kind=end_kind,
        outcome_kind=outcome_kind,
        end_marker_node_id=end_marker_node_id,
    )


def assemble_case_path(case: FeatureFlowCase) -> list[str]:
    path: list[str] = []

    for segment in case.segments:
        path = append_path(path, segment.path)

    return path


def unresolved_case_for_converge_failure(
    case: FeatureFlowCase,
    *,
    converge_point: FeatureConvergePoint,
) -> UnresolvedFeatureFlowCase:
    return UnresolvedFeatureFlowCase(
        feature_name=case.feature_name,
        case_branch_path=case.case_branch_path,
        active_branch_path=case.active_branch_path,
        current_eis=case.current_eis,
        segments=case.segments,
        reason=FeatureFlowUnresolvedReason.NO_VALID_CONVERGE_PATH,
        expected_converge_point=converge_point,
        expected_end_marker=None,
    )


def unresolved_case_for_end_failure(
    case: FeatureFlowCase,
    *,
    end: FeatureMarkerRecord,
) -> UnresolvedFeatureFlowCase:
    return UnresolvedFeatureFlowCase(
        feature_name=case.feature_name,
        case_branch_path=case.case_branch_path,
        active_branch_path=case.active_branch_path,
        current_eis=case.current_eis,
        segments=case.segments,
        reason=FeatureFlowUnresolvedReason.NO_VALID_END_PATH,
        expected_converge_point=None,
        expected_end_marker=end,
    )


def feature_flow_case_id(case: FeatureFlowCase) -> str:
    branch_path_key = branch_path_to_key(case.case_branch_path)

    if case.end_marker_node_id is None:
        return f"{case.feature_name}{BRANCH_PATH_SEPARATOR}{branch_path_key}"

    return (
        f"{case.feature_name}"
        f"{BRANCH_PATH_SEPARATOR}{branch_path_key}"
        f"{BRANCH_PATH_SEPARATOR}end"
        f"{BRANCH_PATH_SEPARATOR}{case.end_marker_node_id}"
    )


def unresolved_feature_flow_case_id(
    case: UnresolvedFeatureFlowCase,
) -> str:
    branch_path_key = branch_path_to_key(case.case_branch_path)

    if case.expected_converge_point is not None:
        return (
            f"{case.feature_name}"
            f"{BRANCH_PATH_SEPARATOR}{branch_path_key}"
            f"{BRANCH_PATH_SEPARATOR}converge"
            f"{BRANCH_PATH_SEPARATOR}{case.expected_converge_point.marker_node_id}"
        )

    if case.expected_end_marker is not None:
        return (
            f"{case.feature_name}"
            f"{BRANCH_PATH_SEPARATOR}{branch_path_key}"
            f"{BRANCH_PATH_SEPARATOR}end"
            f"{BRANCH_PATH_SEPARATOR}{case.expected_end_marker.node_id}"
        )

    return (
        f"{case.feature_name}"
        f"{BRANCH_PATH_SEPARATOR}{branch_path_key}"
        f"{BRANCH_PATH_SEPARATOR}unresolved"
    )


# =============================================================================
# Branch point discovery helpers
# =============================================================================


def branch_point_key(record: FeatureMarkerRecord) -> tuple[str, str]:
    if record.marked_statement is not None:
        return (
            record.callable_id,
            record.marked_statement.entry_ei.ei_id,
        )

    return (
        record.callable_id,
        record.node_id,
    )


def branch_point_control_ei_id(record: FeatureMarkerRecord) -> str:
    if record.marked_statement is not None:
        return record.marked_statement.entry_ei.ei_id

    return record.node_id


def marked_statement_has_control_stmt_type(
    marked_statement: MarkedStatementMetadata | None,
    stmt_type: str,
) -> bool:
    if marked_statement is None:
        return False

    return any(
        control_ei.stmt_type == stmt_type for control_ei in marked_statement.control_eis
    )


def ordered_conditional_targets_for_marked_statement(
    marked_statement: MarkedStatementMetadata | None,
) -> list[tuple[str, str]]:
    if marked_statement is None:
        return []

    result: list[tuple[str, str]] = []
    seen_target_eis: set[str] = set()

    for control_ei in marked_statement.control_eis:
        for target in control_ei.conditional_targets:
            target_ei = target.get("target_ei")
            if target_ei is None:
                continue

            target_text = str(target_ei)
            if target_text in seen_target_eis:
                continue

            seen_target_eis.add(target_text)
            result.append(
                (
                    target_text,
                    str(target.get("target_condition") or ""),
                )
            )

    return result


def feature_marker_record_with_branch_targets(
    record: FeatureMarkerRecord,
    *,
    selected_target_eis: list[str],
    selected_conditions: list[str],
) -> FeatureMarkerRecord:
    branch_selection = record.branch_selection or FeatureBranchSelection(
        branch=record.kwargs.get("branch"),
        control_polarity=parse_control_polarity(record.kwargs.get("control_polarity")),
        selected_target_eis=[],
        selected_conditions=[],
    )

    return replace(
        record,
        branch_selection=FeatureBranchSelection(
            branch=branch_selection.branch,
            control_polarity=branch_selection.control_polarity,
            selected_target_eis=selected_target_eis,
            selected_conditions=selected_conditions,
        ),
    )


def normalize_match_branch_marker_records(
    records: list[FeatureMarkerRecord],
) -> list[FeatureMarkerRecord]:
    if not records:
        return records

    first = records[0]

    if not marked_statement_has_control_stmt_type(first.marked_statement, "Match"):
        return records

    ordered_targets = ordered_conditional_targets_for_marked_statement(
        first.marked_statement,
    )

    if len(ordered_targets) != len(records):
        # Leave the original selection visible rather than guessing silently.
        return records

    normalized_records: list[FeatureMarkerRecord] = []

    for record, (target_ei, condition) in zip(records, ordered_targets, strict=True):
        normalized_records.append(
            feature_marker_record_with_branch_targets(
                record,
                selected_target_eis=[target_ei],
                selected_conditions=[condition] if condition else [],
            )
        )

    return normalized_records


def normalize_branch_marker_records(
    records: list[FeatureMarkerRecord],
) -> list[FeatureMarkerRecord]:
    records = normalize_match_branch_marker_records(records)
    return records


def build_feature_branch_points(
    feature: FeatureMarkerInventory,
) -> list[FeatureBranchPoint]:
    grouped: dict[tuple[str, str], list[FeatureMarkerRecord]] = {}

    for branch_marker in feature.branches:
        grouped.setdefault(
            branch_point_key(branch_marker),
            [],
        ).append(branch_marker)

    branch_points: list[FeatureBranchPoint] = []

    for index, raw_records in enumerate(grouped.values(), start=1):
        records = normalize_branch_marker_records(raw_records)
        first = records[0]
        control_ei_id = branch_point_control_ei_id(first)

        branch_points.append(
            FeatureBranchPoint(
                feature_name=feature.feature_name,
                branch_point_id=f"{feature.feature_name}::branch_point::{index}",
                marker_node_id=first.node_id,
                control_ei_id=control_ei_id,
                branch_markers=tuple(records),
            )
        )

    return branch_points


def branch_name_for_marker(branch_marker: FeatureMarkerRecord) -> str:
    if branch_marker.branch_selection is None:
        return "main"

    return branch_marker.branch_selection.branch or "main"


# =============================================================================
# Converge point discovery helpers
# =============================================================================


def converge_point_ei_id(record: FeatureMarkerRecord) -> str:
    if record.marked_statement is not None:
        return record.marked_statement.entry_ei.ei_id

    return record.node_id


def parse_converge_source_branches(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())

    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())

    return ()


def converge_into_branch(record: FeatureMarkerRecord) -> str:
    value = record.kwargs.get("into")

    if value is None:
        return "main"

    text = str(value).strip()
    return text if text else "main"


def build_feature_converge_points(
    feature: FeatureMarkerInventory,
) -> list[FeatureConvergePoint]:
    converge_points: list[FeatureConvergePoint] = []

    for index, converge_marker in enumerate(feature.converges, start=1):
        converge_points.append(
            FeatureConvergePoint(
                feature_name=feature.feature_name,
                converge_point_id=f"{feature.feature_name}::converge_point::{index}",
                marker_node_id=converge_marker.node_id,
                converge_ei_id=converge_point_ei_id(converge_marker),
                source_branches=parse_converge_source_branches(
                    converge_marker.kwargs.get("branches")
                ),
                into_branch=converge_into_branch(converge_marker),
                marker=converge_marker,
            )
        )

    return converge_points


def branch_leaf_name(branch_path: tuple[str, ...]) -> str:
    if not branch_path:
        return "main"

    return branch_path[-1]


def branch_path_for_converge_target(
    converge_point: FeatureConvergePoint,
) -> tuple[str, ...]:
    if converge_point.into_branch == "main":
        return ("main",)

    return tuple(
        item.strip()
        for item in converge_point.into_branch.split(BRANCH_PATH_SEPARATOR)
        if item.strip()
    )


def converge_applies_to_case(
    converge_point: FeatureConvergePoint,
    case: FeatureFlowCase,
) -> bool:
    if not converge_point.source_branches:
        return True

    active_branches = set(case.active_branch_path)
    source_branches = set(converge_point.source_branches)

    # Most convergences apply to the current leaf branch.  Nested/disruptive
    # lineages can carry a more specific leaf while the convergence belongs to
    # an ancestor branch, e.g. loop_entered::use_value converging through the
    # loop_entered/loop_skipped loop convergence.
    return bool(active_branches & source_branches)


# =============================================================================
# Converge routing helpers
# =============================================================================


def trace_nearest_converge_point_for_case(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    converge_points: list[FeatureConvergePoint],
    branch_points: list[FeatureBranchPoint],
) -> tuple[FeatureConvergePoint, FeaturePathSegment] | None:
    best: tuple[FeatureConvergePoint, FeaturePathSegment] | None = None

    for converge_point in converge_points:
        if not converge_applies_to_case(
            converge_point,
            case,
        ):
            continue

        forbidden_nodes = forbidden_sibling_branch_region_nodes(
            graphs.cfg,
            branch_points=branch_points,
            case=case,
            converge_point=converge_point,
        )

        segment = trace_feature_path_segment(
            graphs,
            feature_name=case.feature_name,
            segment_branch_path=case.active_branch_path,
            start_eis=list(case.current_eis),
            end_eis=[converge_point.converge_ei_id],
            role=FeaturePathSegmentRole.BRANCH_TO_CONVERGE,
            disposition=FeaturePathSegmentDisposition.ACCEPTED,
            disposition_reason=FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH,
            forbidden_nodes=forbidden_nodes,
        )

        if segment is None:
            continue

        if best is None or len(segment.path) < len(best[1].path):
            best = (
                converge_point,
                segment,
            )

    return best


# =============================================================================
# Branch case expansion helpers
# =============================================================================


def selected_target_eis_for_branch_marker(
    branch_marker: FeatureMarkerRecord,
) -> list[str]:
    if branch_marker.branch_selection is None:
        return []

    return branch_marker.branch_selection.selected_target_eis


def end_branch_name(end: FeatureMarkerRecord) -> str:
    value = end.kwargs.get("branch")

    if value is None:
        return "main"

    text = str(value).strip()
    return text if text else "main"


def conditional_end_branch_names(
    feature: FeatureMarkerInventory,
) -> set[str]:
    result: set[str] = set()

    for conditional_end in feature.conditional_ends:
        branch_name = end_branch_name(conditional_end)
        if branch_name != "main":
            result.add(branch_name)

    return result


def branch_marker_applies_to_end(
    *,
    branch_marker: FeatureMarkerRecord,
    end: FeatureMarkerRecord,
    normal_end_excluded_branches: set[str],
) -> bool:
    branch_name = branch_name_for_marker(branch_marker)
    required_branch = end_branch_name(end)

    if required_branch != "main":
        return branch_name == required_branch

    if end.marker_name == "FeatureEnd":
        return branch_name not in normal_end_excluded_branches

    return True


def branch_point_marker_for_branch(
    branch_point: FeatureBranchPoint,
    branch_name: str,
) -> FeatureMarkerRecord | None:
    for marker in branch_point.branch_markers:
        if branch_name_for_marker(marker) == branch_name:
            return marker

    return None


def all_control_target_eis_for_branch_point(
    branch_point: FeatureBranchPoint,
) -> list[str]:
    result: list[str] = []

    for marker in branch_point.branch_markers:
        marked_statement = marker.marked_statement
        if marked_statement is None:
            continue

        for control_ei in marked_statement.control_eis:
            for target in control_ei.conditional_targets:
                target_ei = target.get("target_ei")
                if target_ei and target_ei not in result:
                    result.append(str(target_ei))

    return result


def selected_target_eis_for_branch_point(
    branch_point: FeatureBranchPoint,
) -> set[str]:
    result: set[str] = set()

    for marker in branch_point.branch_markers:
        result.update(selected_target_eis_for_branch_marker(marker))

    return result


def continuation_target_eis_for_branch_point(
    branch_point: FeatureBranchPoint,
) -> list[str]:
    all_targets = all_control_target_eis_for_branch_point(branch_point)
    selected_targets = selected_target_eis_for_branch_point(branch_point)

    return [target_ei for target_ei in all_targets if target_ei not in selected_targets]


def target_eis_for_end_at_branch_point(
    *,
    end: FeatureMarkerRecord,
    branch_point: FeatureBranchPoint,
) -> tuple[list[str], str]:
    required_branch = end_branch_name(end)

    if required_branch != "main":
        required_marker = branch_point_marker_for_branch(
            branch_point,
            required_branch,
        )

        if required_marker is not None:
            return (
                selected_target_eis_for_branch_marker(required_marker),
                required_branch,
            )

    return (
        continuation_target_eis_for_branch_point(branch_point),
        "main",
    )


def forbidden_sibling_branch_region_nodes(
    cfg: nx.MultiDiGraph,
    *,
    branch_points: list[FeatureBranchPoint],
    case: FeatureFlowCase,
    converge_point: FeatureConvergePoint,
) -> set[str]:
    active_leaf = branch_leaf_name(case.active_branch_path)
    active_callable = case_callable_name(cfg, case)

    if active_callable is None:
        return set()

    forbidden: set[str] = set()

    converge_boundary_candidates = [
        converge_point.marker_node_id,
        converge_point.converge_ei_id,
    ]

    converge_boundary_key = min(
        parse_ei_sort_key(node_id)
        for node_id in converge_boundary_candidates
        if node_id
    )

    for branch_point in branch_points:
        branch_targets: list[tuple[str, str]] = []

        for branch_marker in branch_point.branch_markers:
            branch_name = branch_name_for_marker(branch_marker)
            selected_targets = selected_target_eis_for_branch_marker(branch_marker)

            for selected_target in selected_targets:
                branch_targets.append(
                    (
                        branch_name,
                        selected_target,
                    )
                )

        branch_targets.sort(key=lambda item: parse_ei_sort_key(item[1]))

        branch_names = {branch_name for branch_name, _target in branch_targets}

        if active_leaf not in branch_names:
            continue

        for index, (branch_name, branch_target) in enumerate(branch_targets):
            if branch_name == active_leaf:
                continue

            branch_start_key = parse_ei_sort_key(branch_target)

            later_branch_targets = [
                parse_ei_sort_key(other_target)
                for _other_branch_name, other_target in branch_targets[index + 1 :]
            ]

            branch_end_key = min(
                [
                    *later_branch_targets,
                    converge_boundary_key,
                ]
            )

            for node_id in cfg.nodes:
                node_text = str(node_id)

                if node_text in {
                    converge_point.marker_node_id,
                    converge_point.converge_ei_id,
                }:
                    continue

                if is_statement_anchor_node(
                    cfg,
                    node_text,
                ):
                    continue

                node_data = cfg.nodes.get(node_text, {})

                if node_data.get("callable_name") != active_callable:
                    continue

                node_key = parse_ei_sort_key(node_text)

                if branch_start_key <= node_key < branch_end_key:
                    forbidden.add(node_text)

    return forbidden


def extend_branch_path(
    branch_path: tuple[str, ...],
    branch_name: str,
) -> tuple[str, ...]:
    if branch_name == "main":
        return branch_path

    return *branch_path, branch_name


def advance_case_through_branch_marker(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    branch_marker: FeatureMarkerRecord,
) -> FeatureFlowCase | None:
    branch_name = branch_name_for_marker(branch_marker)
    target_eis = selected_target_eis_for_branch_marker(branch_marker)

    if not target_eis:
        return None

    next_case_branch_path = extend_branch_path(
        case.case_branch_path,
        branch_name,
    )
    next_active_branch_path = extend_branch_path(
        case.active_branch_path,
        branch_name,
    )

    branch_target_segment = trace_feature_path_segment(
        graphs,
        feature_name=case.feature_name,
        segment_branch_path=next_active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=target_eis,
        role=FeaturePathSegmentRole.BRANCH_TARGET,
        disposition=FeaturePathSegmentDisposition.ACCEPTED,
        disposition_reason=FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH,
    )

    if branch_target_segment is None:
        return None

    return FeatureFlowCase(
        feature_name=case.feature_name,
        case_branch_path=next_case_branch_path,
        active_branch_path=next_active_branch_path,
        current_eis=(branch_target_segment.end_ei,),
        segments=(*case.segments, branch_target_segment),
        status=case.status,
        end_kind=case.end_kind,
        outcome_kind=case.outcome_kind,
        end_marker_node_id=case.end_marker_node_id,
    )


def expand_case_through_branch_point(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    branch_point: FeatureBranchPoint,
    end: FeatureMarkerRecord,
    normal_end_excluded_branches: set[str],
) -> list[FeatureFlowCase]:
    expanded_cases: list[FeatureFlowCase] = []

    for branch_marker in branch_point.branch_markers:
        if not branch_marker_applies_to_end(
            branch_marker=branch_marker,
            end=end,
            normal_end_excluded_branches=normal_end_excluded_branches,
        ):
            continue

        expanded_case = advance_case_through_branch_marker(
            graphs,
            case=case,
            branch_marker=branch_marker,
        )

        if expanded_case is None:
            continue

        expanded_cases.append(expanded_case)

    return expanded_cases


# =============================================================================
# Segment-based feature flow tracing
# =============================================================================


def debug_print_end_path_failure(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    end: FeatureMarkerRecord,
    max_candidates: int = 10,
) -> None:
    end_targets = marker_end_eis(end)

    print()
    print("=== DEBUG END PATH FAILURE ===")
    print(f"feature={case.feature_name}")
    print(f"case_branch_path={case.case_branch_path}")
    print(f"active_branch_path={case.active_branch_path}")
    print(f"current_eis={case.current_eis}")
    print(f"end_marker_node_id={end.node_id}")
    print(f"end_marker_name={end.marker_name}")
    print(f"end_targets={end_targets}")

    print()
    print("source nodes:")
    for source in case.current_eis:
        node_data = graphs.cfg.nodes.get(source, {})
        print(
            f"  {source} | "
            f"callable={node_data.get('callable_name')} | "
            f"description={node_data.get('description') or node_data.get('condition') or node_data.get('stmt_type')}"
        )

    print()
    print("target nodes:")
    for target in end_targets:
        node_data = graphs.cfg.nodes.get(target, {})
        print(
            f"  {target} | "
            f"callable={node_data.get('callable_name')} | "
            f"description={node_data.get('description') or node_data.get('condition') or node_data.get('stmt_type')}"
        )

    print()
    print("candidate paths:")

    printed = 0

    for source in case.current_eis:
        for target in end_targets:
            if source not in graphs.cfg:
                print(f"  source missing from graph: {source}")
                continue

            if target not in graphs.cfg:
                print(f"  target missing from graph: {target}")
                continue

            try:
                candidate_paths = nx.shortest_simple_paths(
                    graphs.cfg_nav,
                    source,
                    target,
                )

                for candidate_path in candidate_paths:
                    printed += 1

                    rejection_reason = graph_path_rejection_reason(
                        graphs.cfg,
                        candidate_path,
                    )

                    print()
                    print(
                        f"  path {printed}: "
                        f"source={source} target={target} "
                        f"length={len(candidate_path)} "
                        f"valid={rejection_reason is None} "
                        f"reason={rejection_reason}"
                    )

                    for node_id in candidate_path:
                        node_data = graphs.cfg.nodes.get(node_id, {})
                        print(
                            f"    {node_id} | "
                            f"callable={node_data.get('callable_name')} | "
                            f"description={node_data.get('description') or node_data.get('condition') or node_data.get('stmt_type')}"
                        )

                    if printed >= max_candidates:
                        print("=== END DEBUG END PATH FAILURE ===")
                        print()
                        return

            except nx.NetworkXNoPath:
                print(f"  no path: {source} -> {target}")
            except nx.NodeNotFound as exc:
                print(f"  node not found: {exc}")

    if printed == 0:
        print("  <no candidate paths found>")

    print("=== END DEBUG END PATH FAILURE ===")
    print()


def trace_case_to_branch_point(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    branch_point: FeatureBranchPoint,
) -> FeatureFlowCase | None:
    segment = trace_feature_path_segment(
        graphs,
        feature_name=case.feature_name,
        segment_branch_path=case.active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=[branch_point.control_ei_id],
        role=FeaturePathSegmentRole.START_TO_BRANCH,
        disposition=FeaturePathSegmentDisposition.ACCEPTED,
        disposition_reason=FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH,
    )

    if segment is None:
        return None

    return append_segment_to_case(
        case,
        segment,
        current_eis=[segment.end_ei],
    )


def trace_case_to_converge_point(
    *,
    case: FeatureFlowCase,
    converge_point: FeatureConvergePoint,
    segment: FeaturePathSegment,
) -> FeatureFlowCase:
    return append_segment_to_case(
        case,
        segment,
        current_eis=[segment.end_ei],
        active_branch_path=branch_path_for_converge_target(converge_point),
    )


def trace_case_to_end(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    end: FeatureMarkerRecord,
    forbidden_nodes: set[str] | None = None,
) -> FeatureFlowCase | None:
    end_targets = marker_end_eis(end)

    role = (
        FeaturePathSegmentRole.TO_CONDITIONAL_END
        if end.marker_name == "FeatureEndConditional"
        else FeaturePathSegmentRole.TO_END
    )

    disposition = FeaturePathSegmentDisposition.ACCEPTED
    disposition_reason = FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH

    end_kind = (
        FeatureFlowEndKind.FEATURE_END_CONDITIONAL
        if end.marker_name == "FeatureEndConditional"
        else FeatureFlowEndKind.FEATURE_END
    )

    outcome_kind = (
        FeatureFlowOutcomeKind.CONDITIONAL
        if end.marker_name == "FeatureEndConditional"
        else FeatureFlowOutcomeKind.SUCCESS
    )

    segment = trace_feature_path_segment(
        graphs,
        feature_name=case.feature_name,
        segment_branch_path=case.active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=end_targets,
        role=role,
        disposition=disposition,
        disposition_reason=disposition_reason,
        forbidden_nodes=forbidden_nodes,
    )

    if segment is None:
        return None

    return complete_feature_flow_case(
        case,
        end_marker_node_id=end.node_id,
        end_kind=end_kind,
        outcome_kind=outcome_kind,
        current_eis=[segment.end_ei],
        segment=segment,
    )


def trace_case_to_end_before_branch_point(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    end: FeatureMarkerRecord,
    branch_point: FeatureBranchPoint,
) -> FeatureFlowCase | None:
    forbidden_nodes = {
        branch_point.control_ei_id,
        branch_point.marker_node_id,
    }

    return trace_case_to_end(
        graphs,
        case=case,
        end=end,
        forbidden_nodes=forbidden_nodes,
    )


# =============================================================================
# Branch-aware event traversal helpers
# =============================================================================


@dataclass(frozen=True)
class FeatureFlowTraversalEvent:
    event_kind: str
    event: FeatureBranchPoint | FeatureConvergePoint | FeatureMarkerRecord
    segment: FeaturePathSegment


def feature_event_endpoint_nodes(
    *,
    branch_points: list[FeatureBranchPoint],
    converge_points: list[FeatureConvergePoint],
    ends: list[FeatureMarkerRecord],
) -> set[str]:
    nodes: set[str] = set()

    for branch_point in branch_points:
        nodes.add(branch_point.control_ei_id)

    for converge_point in converge_points:
        nodes.add(converge_point.converge_ei_id)

    for end in ends:
        nodes.update(marker_end_eis(end))

    return nodes


def end_applies_to_case(
    end: FeatureMarkerRecord,
    case: FeatureFlowCase,
) -> bool:
    required_branch = end_branch_name(end)

    if required_branch == "main":
        return True

    return branch_leaf_name(case.active_branch_path) == required_branch


def branch_point_has_selectable_targets(
    branch_point: FeatureBranchPoint,
) -> bool:
    return any(
        selected_target_eis_for_branch_marker(branch_marker)
        for branch_marker in branch_point.branch_markers
    )


def trace_case_to_branch_point_ordered(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    branch_point: FeatureBranchPoint,
    event_endpoint_nodes: set[str],
) -> FeaturePathSegment | None:
    # A branch point is a semantic event, not an ordinary waypoint.  Once a
    # case has consumed it, do not allow graph search to route back to it and
    # expand the same branch again.  This is especially important for try/loop
    # decompositions where graph edges may make the control EI reachable from
    # branch-target regions.
    consumed_nodes = set(assemble_case_path(case))
    current_nodes = set(case.current_eis)
    if (
        branch_point.control_ei_id in consumed_nodes
        or branch_point.marker_node_id in consumed_nodes
    ) and not (
        branch_point.control_ei_id in current_nodes
        or branch_point.marker_node_id in current_nodes
    ):
        return None

    forbidden_nodes = event_endpoint_nodes - {branch_point.control_ei_id}

    return trace_feature_path_segment(
        graphs,
        feature_name=case.feature_name,
        segment_branch_path=case.active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=[branch_point.control_ei_id],
        role=FeaturePathSegmentRole.START_TO_BRANCH,
        disposition=FeaturePathSegmentDisposition.ACCEPTED,
        disposition_reason=FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH,
        forbidden_nodes=forbidden_nodes,
    )


def trace_case_to_converge_point_ordered(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    converge_point: FeatureConvergePoint,
    event_endpoint_nodes: set[str],
    branch_points: list[FeatureBranchPoint],
) -> FeaturePathSegment | None:
    consumed_nodes = set(assemble_case_path(case))
    current_nodes = set(case.current_eis)

    if not converge_applies_to_case(converge_point, case):
        return None

    # A branch target is often the convergence anchor itself, especially for
    # loop-skipped / branch-completion routes. That is a valid next event and
    # must be consumed as a zero-hop/single-node segment. Only suppress a
    # convergence point that was consumed earlier and is not the current
    # location.
    if (
        converge_point.converge_ei_id in consumed_nodes
        or converge_point.marker_node_id in consumed_nodes
    ) and not (
        converge_point.converge_ei_id in current_nodes
        or converge_point.marker_node_id in current_nodes
    ):
        return None

    forbidden_nodes = event_endpoint_nodes - {converge_point.converge_ei_id}
    sibling_forbidden_nodes = forbidden_sibling_branch_region_nodes(
        graphs.cfg,
        branch_points=branch_points,
        case=case,
        converge_point=converge_point,
    )

    return trace_feature_path_segment(
        graphs,
        feature_name=case.feature_name,
        segment_branch_path=case.active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=[converge_point.converge_ei_id],
        role=FeaturePathSegmentRole.BRANCH_TO_CONVERGE,
        disposition=FeaturePathSegmentDisposition.ACCEPTED,
        disposition_reason=FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH,
        forbidden_nodes=forbidden_nodes | sibling_forbidden_nodes,
    )


def trace_case_to_end_ordered(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    end: FeatureMarkerRecord,
    event_endpoint_nodes: set[str],
) -> FeaturePathSegment | None:
    if not end_applies_to_case(end, case):
        return None

    end_targets = marker_end_eis(end)
    forbidden_nodes = event_endpoint_nodes - set(end_targets)

    role = (
        FeaturePathSegmentRole.TO_CONDITIONAL_END
        if end.marker_name == "FeatureEndConditional"
        else FeaturePathSegmentRole.TO_END
    )

    return trace_feature_path_segment(
        graphs,
        feature_name=case.feature_name,
        segment_branch_path=case.active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=end_targets,
        role=role,
        disposition=FeaturePathSegmentDisposition.ACCEPTED,
        disposition_reason=FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH,
        forbidden_nodes=forbidden_nodes,
    )


def choose_next_traversal_event(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    branch_points: list[FeatureBranchPoint],
    converge_points: list[FeatureConvergePoint],
    ends: list[FeatureMarkerRecord],
    event_endpoint_nodes: set[str],
) -> FeatureFlowTraversalEvent | None:
    candidates: list[FeatureFlowTraversalEvent] = []

    for branch_point in branch_points:
        if not branch_point_has_selectable_targets(branch_point):
            continue

        segment = trace_case_to_branch_point_ordered(
            graphs,
            case=case,
            branch_point=branch_point,
            event_endpoint_nodes=event_endpoint_nodes,
        )

        if segment is not None:
            candidates.append(
                FeatureFlowTraversalEvent(
                    event_kind="branch",
                    event=branch_point,
                    segment=segment,
                )
            )

    for converge_point in converge_points:
        segment = trace_case_to_converge_point_ordered(
            graphs,
            case=case,
            converge_point=converge_point,
            event_endpoint_nodes=event_endpoint_nodes,
            branch_points=branch_points,
        )

        if segment is not None:
            candidates.append(
                FeatureFlowTraversalEvent(
                    event_kind="converge",
                    event=converge_point,
                    segment=segment,
                )
            )

    for end in ends:
        segment = trace_case_to_end_ordered(
            graphs,
            case=case,
            end=end,
            event_endpoint_nodes=event_endpoint_nodes,
        )

        if segment is not None:
            candidates.append(
                FeatureFlowTraversalEvent(
                    event_kind="end",
                    event=end,
                    segment=segment,
                )
            )

    if not candidates:
        return None

    event_priority = {
        "branch": 0,
        "end": 1,
        "converge": 2,
    }

    candidates.sort(
        key=lambda candidate: (
            len(candidate.segment.path),
            event_priority.get(candidate.event_kind, 99),
            parse_ei_sort_key(candidate.segment.end_ei),
        )
    )

    return candidates[0]


def advance_case_through_branch_marker_ordered(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    branch_marker: FeatureMarkerRecord,
    event_endpoint_nodes: set[str],
) -> FeatureFlowCase | None:
    branch_name = branch_name_for_marker(branch_marker)
    target_eis = selected_target_eis_for_branch_marker(branch_marker)

    if not target_eis:
        return None

    next_case_branch_path = extend_branch_path(
        case.case_branch_path,
        branch_name,
    )
    next_active_branch_path = extend_branch_path(
        case.active_branch_path,
        branch_name,
    )

    branch_target_segment = trace_feature_path_segment(
        graphs,
        feature_name=case.feature_name,
        segment_branch_path=next_active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=target_eis,
        role=FeaturePathSegmentRole.BRANCH_TARGET,
        disposition=FeaturePathSegmentDisposition.ACCEPTED,
        disposition_reason=FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH,
        forbidden_nodes=event_endpoint_nodes - set(target_eis),
    )

    if branch_target_segment is None:
        return None

    return FeatureFlowCase(
        feature_name=case.feature_name,
        case_branch_path=next_case_branch_path,
        active_branch_path=next_active_branch_path,
        current_eis=(branch_target_segment.end_ei,),
        segments=(*case.segments, branch_target_segment),
        status=case.status,
        end_kind=case.end_kind,
        outcome_kind=case.outcome_kind,
        end_marker_node_id=case.end_marker_node_id,
    )


def expand_case_through_branch_point_ordered(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
    branch_point: FeatureBranchPoint,
    event_endpoint_nodes: set[str],
) -> list[FeatureFlowCase]:
    expanded_cases: list[FeatureFlowCase] = []

    for branch_marker in branch_point.branch_markers:
        expanded_case = advance_case_through_branch_marker_ordered(
            graphs,
            case=case,
            branch_marker=branch_marker,
            event_endpoint_nodes=event_endpoint_nodes,
        )

        if expanded_case is None:
            continue

        expanded_cases.append(expanded_case)

    return expanded_cases


def complete_case_from_end_event(
    case: FeatureFlowCase,
    *,
    end: FeatureMarkerRecord,
    segment: FeaturePathSegment,
) -> FeatureFlowCase:
    end_kind = (
        FeatureFlowEndKind.FEATURE_END_CONDITIONAL
        if end.marker_name == "FeatureEndConditional"
        else FeatureFlowEndKind.FEATURE_END
    )

    outcome_kind = (
        FeatureFlowOutcomeKind.CONDITIONAL
        if end.marker_name == "FeatureEndConditional"
        else FeatureFlowOutcomeKind.SUCCESS
    )

    return complete_feature_flow_case(
        case,
        end_marker_node_id=end.node_id,
        end_kind=end_kind,
        outcome_kind=outcome_kind,
        current_eis=[segment.end_ei],
        segment=segment,
    )


def terminal_representative_segment_for_case(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
) -> FeaturePathSegment | None:
    candidates: list[tuple[str, str]] = []

    for source_ei in case.current_eis:
        if source_ei not in graphs.cfg:
            continue

        for _source, target, _key, edge_data in graphs.cfg.out_edges(
            source_ei,
            keys=True,
            data=True,
        ):
            if (
                edge_data.get("edge_type")
                != "derived_control_route_execution_item_terminal"
            ):
                continue

            if edge_data.get("resolved_target_kind") != "terminal_placeholder":
                continue

            if edge_data.get("route_kind") not in TERMINAL_REPRESENTATIVE_ROUTE_KINDS:
                continue

            if edge_data.get("exit_kind") not in TERMINAL_REPRESENTATIVE_EXIT_KINDS:
                continue

            candidates.append(
                (
                    str(source_ei),
                    str(target),
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            parse_ei_sort_key(item[0]),
            parse_ei_sort_key(item[1]),
        )
    )

    source_ei, terminal_ei = candidates[0]

    return FeaturePathSegment(
        feature_name=case.feature_name,
        segment_branch_path=case.active_branch_path,
        start_ei=source_ei,
        end_ei=terminal_ei,
        path=[source_ei, terminal_ei],
        role=FeaturePathSegmentRole.TO_CONDITIONAL_END,
        disposition=FeaturePathSegmentDisposition.ACCEPTED,
        disposition_reason=FeaturePathSegmentDispositionReason.EXACT_GRAPH_PATH,
    )


def complete_terminal_representative_case(
    graphs: FeatureFlowGraphs,
    *,
    case: FeatureFlowCase,
) -> FeatureFlowCase | None:
    segment = terminal_representative_segment_for_case(
        graphs,
        case=case,
    )

    if segment is None:
        return None

    return complete_feature_flow_case(
        case,
        end_marker_node_id=segment.end_ei,
        end_kind=FeatureFlowEndKind.FEATURE_END_CONDITIONAL,
        outcome_kind=FeatureFlowOutcomeKind.CONDITIONAL,
        current_eis=[segment.end_ei],
        segment=segment,
    )


def case_state_key(case: FeatureFlowCase) -> tuple[Any, ...]:
    return (
        case.feature_name,
        case.case_branch_path,
        case.active_branch_path,
        case.current_eis,
        tuple(feature_path_segment_id(segment) for segment in case.segments),
    )


def fallback_expected_end(
    ends: list[FeatureMarkerRecord],
    case: FeatureFlowCase,
) -> FeatureMarkerRecord | None:
    for end in ends:
        if end_applies_to_case(end, case):
            return end

    return ends[0] if ends else None


def trace_feature_flow_cases_for_feature(
    graphs: FeatureFlowGraphs,
    *,
    feature: FeatureMarkerInventory,
    start: FeatureMarkerRecord,
    verbose: bool = False,
) -> FeatureFlowTraceResult:
    ends = [
        *feature.ends,
        *feature.conditional_ends,
    ]

    if not ends:
        return FeatureFlowTraceResult(
            completed_cases=[],
            unresolved_cases=[],
        )

    branch_points = build_feature_branch_points(feature)
    converge_points = build_feature_converge_points(feature)
    event_endpoint_nodes = feature_event_endpoint_nodes(
        branch_points=branch_points,
        converge_points=converge_points,
        ends=ends,
    )

    active_cases: list[FeatureFlowCase] = [
        initial_feature_flow_case(
            feature_name=feature.feature_name,
            start_eis=marker_exit_eis(start),
        )
    ]

    completed_cases: list[FeatureFlowCase] = []
    unresolved_cases: list[UnresolvedFeatureFlowCase] = []
    seen_case_states: set[tuple[Any, ...]] = set()

    max_iterations = 250
    max_active_cases = 500
    iterations = 0

    while active_cases:
        iterations += 1
        if iterations > max_iterations:
            for case in active_cases:
                expected_end = fallback_expected_end(ends, case)
                if expected_end is not None:
                    unresolved_cases.append(
                        unresolved_case_for_end_failure(
                            case,
                            end=expected_end,
                        )
                    )
            break

        next_active_cases: list[FeatureFlowCase] = []

        for case in active_cases:
            state_key = case_state_key(case)
            if state_key in seen_case_states:
                continue
            seen_case_states.add(state_key)

            terminal_case = complete_terminal_representative_case(
                graphs,
                case=case,
            )
            if terminal_case is not None:
                completed_cases.append(terminal_case)
                continue

            next_event = choose_next_traversal_event(
                graphs,
                case=case,
                branch_points=branch_points,
                converge_points=converge_points,
                ends=ends,
                event_endpoint_nodes=event_endpoint_nodes,
            )

            if next_event is None:
                expected_end = fallback_expected_end(ends, case)
                if expected_end is not None:
                    unresolved_cases.append(
                        unresolved_case_for_end_failure(
                            case,
                            end=expected_end,
                        )
                    )
                continue

            if next_event.event_kind == "branch":
                branch_point = next_event.event
                if not isinstance(branch_point, FeatureBranchPoint):
                    continue

                case_at_branch_point = append_segment_to_case(
                    case,
                    next_event.segment,
                    current_eis=[next_event.segment.end_ei],
                )

                expanded_cases = expand_case_through_branch_point_ordered(
                    graphs,
                    case=case_at_branch_point,
                    branch_point=branch_point,
                    event_endpoint_nodes=event_endpoint_nodes,
                )

                if not expanded_cases:
                    expected_end = fallback_expected_end(ends, case_at_branch_point)
                    if expected_end is not None:
                        unresolved_cases.append(
                            unresolved_case_for_end_failure(
                                case_at_branch_point,
                                end=expected_end,
                            )
                        )
                    continue

                next_active_cases.extend(expanded_cases)
                continue

            if next_event.event_kind == "converge":
                converge_point = next_event.event
                if not isinstance(converge_point, FeatureConvergePoint):
                    continue

                converged_case = trace_case_to_converge_point(
                    case=case,
                    converge_point=converge_point,
                    segment=next_event.segment,
                )
                next_active_cases.append(converged_case)
                continue

            if next_event.event_kind == "end":
                end = next_event.event
                if not isinstance(end, FeatureMarkerRecord):
                    continue

                completed_cases.append(
                    complete_case_from_end_event(
                        case,
                        end=end,
                        segment=next_event.segment,
                    )
                )
                continue

        if len(next_active_cases) > max_active_cases:
            for case in next_active_cases[max_active_cases:]:
                expected_end = fallback_expected_end(ends, case)
                if expected_end is not None:
                    unresolved_cases.append(
                        unresolved_case_for_end_failure(
                            case,
                            end=expected_end,
                        )
                    )
            next_active_cases = next_active_cases[:max_active_cases]

        active_cases = next_active_cases

    return FeatureFlowTraceResult(
        completed_cases=completed_cases,
        unresolved_cases=unresolved_cases,
    )


def marker_start_eis(record: FeatureMarkerRecord) -> list[str]:
    return [record.node_id]


def trace_feature_flow_cases_to_end(
    graphs: FeatureFlowGraphs,
    *,
    feature: FeatureMarkerInventory,
    start: FeatureMarkerRecord,
    end: FeatureMarkerRecord,
    verbose: bool = False,
) -> FeatureFlowTraceResult:
    # Compatibility wrapper for older callers. The branch-aware engine traces all
    # feature ends together so conditional ends can act as ordered events instead
    # of being treated as separate global destinations.
    del end
    return trace_feature_flow_cases_for_feature(
        graphs,
        feature=feature,
        start=start,
        verbose=verbose,
    )


def trace_feature_flow_cases(
    graphs: FeatureFlowGraphs,
    inventories: dict[str, FeatureMarkerInventory],
    *,
    verbose: bool = False,
) -> FeatureFlowTraceResult:
    completed_cases: list[FeatureFlowCase] = []
    unresolved_cases: list[UnresolvedFeatureFlowCase] = []

    for feature in inventories.values():
        if not feature.starts:
            continue

        start = feature.starts[0]
        ends = [
            *feature.ends,
            *feature.conditional_ends,
        ]

        if verbose:
            branch_points = build_feature_branch_points(feature)
            converge_points = build_feature_converge_points(feature)

            print(
                f"Tracing feature {feature.feature_name}: "
                f"starts={len(feature.starts)} "
                f"branches={len(feature.branches)} "
                f"branch_points={len(branch_points)} "
                f"converges={len(feature.converges)} "
                f"converge_points={len(converge_points)} "
                f"ends={len(ends)}"
            )

        result = trace_feature_flow_cases_for_feature(
            graphs,
            feature=feature,
            start=start,
            verbose=verbose,
        )

        completed_cases.extend(result.completed_cases)
        unresolved_cases.extend(result.unresolved_cases)

    return FeatureFlowTraceResult(
        completed_cases=completed_cases,
        unresolved_cases=unresolved_cases,
    )


# =============================================================================
# Feature flow case inspection helpers
# =============================================================================


def summarize_feature_flow_cases(
    graphs: FeatureFlowGraphs,
    inventories: dict[str, FeatureMarkerInventory],
) -> None:
    trace_result = trace_feature_flow_cases(
        graphs,
        inventories,
    )

    cases = trace_result.completed_cases
    cases_by_feature: dict[str, list[FeatureFlowCase]] = {}

    for case in cases:
        cases_by_feature.setdefault(
            case.feature_name,
            [],
        ).append(case)

    for feature_name, feature_cases in sorted(cases_by_feature.items()):
        print(f"Feature cases for {feature_name}: {len(feature_cases)}")

        for case in feature_cases:
            print(
                f"  branch_path={case.case_branch_path} current_eis={case.current_eis}"
            )

    if trace_result.unresolved_cases:
        print(f"Unresolved feature cases: {len(trace_result.unresolved_cases)}")

        for unresolved_case in trace_result.unresolved_cases:
            print(
                f"  feature={unresolved_case.feature_name} "
                f"branch_path={unresolved_case.case_branch_path} "
                f"active_branch_path={unresolved_case.active_branch_path} "
                f"reason={unresolved_case.reason.value} "
                f"current_eis={unresolved_case.current_eis}"
            )


# =============================================================================
# Path output formatting
# =============================================================================


def compact_special_node_id(node_text: str, prefix: str) -> str:
    marker = f"{prefix}::"
    if node_text.startswith(marker):
        return node_text[len(marker) :]
    return node_text


def describe_path_node(cfg: nx.MultiDiGraph, node_id: str) -> dict[str, Any]:
    node_text = str(node_id)
    node_data = cfg.nodes.get(node_id, {})
    category: str = node_data.get("category", "")

    if node_text.startswith("collapsed::"):
        callable_name = node_data.get("callable_name")
        description = node_data.get("description") or "collapsed call"

        label_parts: list[str] = [
            str(item)
            for item in [
                node_data.get("unit"),
                callable_name,
                description,
            ]
            if item
        ]

        return {
            "id": compact_special_node_id(node_text, "collapsed"),
            "kind": "collapsed",
            "label": "::".join(label_parts) if label_parts else node_text,
        }

    if node_text.startswith("placeholder::"):
        operation_target = node_data.get("operation_target")
        description = node_data.get("description")

        label_parts: list[str] = [
            str(item)
            for item in [
                description,
                operation_target,
            ]
            if item
        ]

        return {
            "id": compact_special_node_id(node_text, "placeholder"),
            "kind": "placeholder",
            "label": "::".join(label_parts) if label_parts else node_text,
        }

    if node_text.startswith("external::"):
        external_type = node_data.get("type")
        operation_target = node_data.get("operation_target")
        description = node_data.get("description")

        label_parts: list[str] = [
            str(item)
            for item in [
                "external",
                external_type,
                operation_target,
                description,
            ]
            if item
        ]

        return {
            "id": compact_special_node_id(node_text, "external"),
            "kind": "external",
            "label": "::".join(label_parts) if label_parts else node_text,
        }

    description = (
        node_data.get("description")
        or node_data.get("condition")
        or node_data.get("stmt_type")
    )

    label_parts: list[str] = [
        str(item)
        for item in [
            node_data.get("unit"),
            node_data.get("callable_name"),
            description,
        ]
        if item
    ]

    result = {
        "id": node_text,
        "label": "::".join(label_parts) if label_parts else node_text,
    }

    if category and category != "execution_instance":
        result["kind"] = category

    return result


def describe_path(cfg: nx.MultiDiGraph, path: list[str]) -> list[dict[str, Any]]:
    return [describe_path_node(cfg, node_id) for node_id in path]


def collect_unique_segments(
    cases: list[FeatureFlowCase],
) -> dict[str, FeaturePathSegment]:
    segments: dict[str, FeaturePathSegment] = {}

    for case in cases:
        for segment in case.segments:
            segment_id = feature_path_segment_id(segment)

            existing = segments.get(segment_id)
            if existing is not None:
                continue

            segments[segment_id] = segment

    return segments


def collect_segment_references(
    cases: list[FeatureFlowCase],
) -> dict[str, list[str]]:
    references: dict[str, list[str]] = {}

    for case in cases:
        case_id = feature_flow_case_id(case)

        for segment in case.segments:
            segment_id = feature_path_segment_id(segment)
            case_refs = references.setdefault(segment_id, [])

            if case_id not in case_refs:
                case_refs.append(case_id)

    return references


def segment_reuse_summary(
    segment_references: dict[str, list[str]],
) -> dict[str, int]:
    reference_counts = [len(references) for references in segment_references.values()]

    shared_segment_count = sum(1 for count in reference_counts if count > 1)

    total_segment_references = sum(reference_counts)

    return {
        "shared_segments": shared_segment_count,
        "total_segment_references": total_segment_references,
        "max_segment_reference_count": max(reference_counts, default=0),
    }


def segment_role_summary(
    segments: dict[str, FeaturePathSegment],
) -> dict[str, int]:
    summary: dict[str, int] = {}

    for segment in segments.values():
        role = segment.role.value
        summary[role] = summary.get(role, 0) + 1

    return dict(
        sorted(
            summary.items(),
            key=lambda item: item[0],
        )
    )


def feature_path_segment_to_dict(
    cfg: nx.MultiDiGraph,
    segment: FeaturePathSegment,
    *,
    referenced_by: list[str] | None = None,
) -> dict[str, Any]:
    references = referenced_by or []

    return {
        "feature_name": segment.feature_name,
        "segment_branch_path": branch_path_to_key(segment.segment_branch_path),
        "start_ei": segment.start_ei,
        "end_ei": segment.end_ei,
        "role": segment.role.value,
        "disposition": segment.disposition.value,
        "disposition_reason": segment.disposition_reason.value,
        "path_length": len(segment.path),
        "reference_count": len(references),
        "referenced_by": references,
        "path": describe_path(cfg, segment.path),
    }


def feature_flow_case_to_dict(
    case: FeatureFlowCase,
) -> dict[str, Any]:
    assembled_path = assemble_case_path(case)

    return {
        "case_id": feature_flow_case_id(case),
        "feature_name": case.feature_name,
        "case_branch_path": branch_path_to_key(case.case_branch_path),
        "active_branch_path": branch_path_to_key(case.active_branch_path),
        "status": case.status.value,
        "end_kind": case.end_kind.value if case.end_kind is not None else None,
        "outcome_kind": (
            case.outcome_kind.value if case.outcome_kind is not None else None
        ),
        "end_marker_node_id": case.end_marker_node_id,
        "segment_count": len(case.segments),
        "path_length": len(assembled_path),
        "segment_ids": [feature_path_segment_id(segment) for segment in case.segments],
    }


def feature_converge_point_ref_to_dict(
    converge_point: FeatureConvergePoint,
) -> dict[str, Any]:
    return {
        "converge_point_id": converge_point.converge_point_id,
        "marker_node_id": converge_point.marker_node_id,
        "converge_ei_id": converge_point.converge_ei_id,
        "source_branches": list(converge_point.source_branches),
        "into_branch": converge_point.into_branch,
    }


def feature_end_marker_ref_to_dict(
    end: FeatureMarkerRecord,
) -> dict[str, Any]:
    return {
        "marker_name": end.marker_name,
        "node_id": end.node_id,
        "end_eis": marker_end_eis(end),
    }


def unresolved_feature_flow_case_to_dict(
    case: UnresolvedFeatureFlowCase,
) -> dict[str, Any]:
    return {
        "case_id": unresolved_feature_flow_case_id(case),
        "feature_name": case.feature_name,
        "case_branch_path": branch_path_to_key(case.case_branch_path),
        "active_branch_path": branch_path_to_key(case.active_branch_path),
        "status": FeatureFlowCaseStatus.UNRESOLVED.value,
        "reason": case.reason.value,
        "current_eis": list(case.current_eis),
        "segment_count": len(case.segments),
        "segment_ids": [feature_path_segment_id(segment) for segment in case.segments],
        "expected_converge_point": (
            feature_converge_point_ref_to_dict(case.expected_converge_point)
            if case.expected_converge_point is not None
            else None
        ),
        "expected_end_marker": (
            feature_end_marker_ref_to_dict(case.expected_end_marker)
            if case.expected_end_marker is not None
            else None
        ),
    }


# =============================================================================
# YAML output writers
# =============================================================================


def feature_branch_points_to_dict(
    inventories: dict[str, FeatureMarkerInventory],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []

    for feature_name, feature in sorted(inventories.items()):
        branch_points = build_feature_branch_points(feature)

        features.append(
            {
                "feature_name": feature_name,
                "branch_point_count": len(branch_points),
                "branch_points": [
                    {
                        "branch_point_id": branch_point.branch_point_id,
                        "marker_node_id": branch_point.marker_node_id,
                        "control_ei_id": branch_point.control_ei_id,
                        "branches": [
                            {
                                "branch": branch_name_for_marker(marker),
                                "marker_node_id": marker.node_id,
                                "control_polarity": (
                                    marker.branch_selection.control_polarity
                                    if marker.branch_selection is not None
                                    else None
                                ),
                                "selected_target_eis": (
                                    marker.branch_selection.selected_target_eis
                                    if marker.branch_selection is not None
                                    else []
                                ),
                                "selected_conditions": (
                                    marker.branch_selection.selected_conditions
                                    if marker.branch_selection is not None
                                    else []
                                ),
                            }
                            for marker in branch_point.branch_markers
                        ],
                    }
                    for branch_point in branch_points
                ],
            }
        )

    return {
        "feature_count": len(features),
        "features": features,
    }


def feature_converge_points_to_dict(
    inventories: dict[str, FeatureMarkerInventory],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []

    for feature_name, feature in sorted(inventories.items()):
        converge_points = build_feature_converge_points(feature)

        features.append(
            {
                "feature_name": feature_name,
                "converge_point_count": len(converge_points),
                "converge_points": [
                    {
                        "converge_point_id": converge_point.converge_point_id,
                        "marker_node_id": converge_point.marker_node_id,
                        "converge_ei_id": converge_point.converge_ei_id,
                        "source_branches": list(converge_point.source_branches),
                        "into_branch": converge_point.into_branch,
                    }
                    for converge_point in converge_points
                ],
            }
        )

    return {
        "feature_count": len(features),
        "features": features,
    }


def write_feature_converge_points(
    *,
    inventories: dict[str, FeatureMarkerInventory],
    output_path: Path,
) -> dict[str, Any]:
    output = feature_converge_points_to_dict(inventories)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        ),
        encoding="utf-8",
    )

    return output


def write_feature_branch_points(
    *,
    inventories: dict[str, FeatureMarkerInventory],
    output_path: Path,
) -> dict[str, Any]:
    output = feature_branch_points_to_dict(inventories)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        ),
        encoding="utf-8",
    )

    return output


def feature_flow_cases_to_dict(
    cfg: nx.MultiDiGraph,
    trace_result: FeatureFlowTraceResult,
) -> dict[str, Any]:
    segments = collect_unique_segments(trace_result.completed_cases)
    segment_references = collect_segment_references(trace_result.completed_cases)
    reuse_summary = segment_reuse_summary(segment_references)
    role_summary = segment_role_summary(segments)

    return {
        "summary": {
            "completed_cases": len(trace_result.completed_cases),
            "unresolved_cases": len(trace_result.unresolved_cases),
            "unique_segments": len(segments),
            **reuse_summary,
            "segments_by_role": role_summary,
        },
        "case_count": len(trace_result.completed_cases),
        "unresolved_case_count": len(trace_result.unresolved_cases),
        "segment_count": len(segments),
        "segments": {
            segment_id: feature_path_segment_to_dict(
                cfg,
                segment,
                referenced_by=segment_references.get(segment_id, []),
            )
            for segment_id, segment in segments.items()
        },
        "cases": [
            feature_flow_case_to_dict(case) for case in trace_result.completed_cases
        ],
        "unresolved_cases": [
            unresolved_feature_flow_case_to_dict(case)
            for case in trace_result.unresolved_cases
        ],
    }


def write_feature_flow_cases(
    *,
    graphs: FeatureFlowGraphs,
    inventories: dict[str, FeatureMarkerInventory],
    output_path: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    trace_result = trace_feature_flow_cases(
        graphs,
        inventories,
        verbose=verbose,
    )
    output = feature_flow_cases_to_dict(graphs.cfg, trace_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        ),
        encoding="utf-8",
    )

    return output


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trace completed feature-flow cases from PyBastion inventory YAML "
            "files and the Stage 1 EI CFG."
        )
    )
    parser.add_argument(
        "--inventory-root",
        type=Path,
        required=True,
        help="Root directory containing *.inventory.yaml files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Primary Stage 2 feature-flow cases YAML output path.",
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        required=True,
        help="Stage 1 EI CFG pickle or YAML used for feature-flow path tracing.",
    )
    parser.add_argument(
        "--cfg-format",
        choices=["pickle", "yaml"],
        default=None,
        help="CFG format. Inferred from --cfg extension when omitted.",
    )
    parser.add_argument(
        "--marker-inventory-output",
        type=Path,
        help="Optional marker inventory YAML output path.",
    )
    parser.add_argument(
        "--branch-points-output",
        type=Path,
        help="Optional feature branch point topology YAML output path.",
    )
    parser.add_argument(
        "--converge-points-output",
        type=Path,
        help="Optional feature converge point topology YAML output path.",
    )
    parser.add_argument(
        "--emit-all-output",
        action="store_true",
        help=(
            "Emit optional Stage 2 diagnostic/support outputs. Each optional "
            "output is written only when its explicit output path is also provided."
        ),
    )
    parser.add_argument(
        "--summarize-flow-cases",
        action="store_true",
        help="Print a console summary of expanded feature flow cases.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )

    args = parser.parse_args()

    inventory_paths = find_inventory_files(args.inventory_root)
    inventories = build_feature_marker_inventory_from_paths(inventory_paths)

    marker_inventory_output = feature_marker_inventory_to_dict(inventories)
    marker_count = sum(
        feature["summary"]["total_markers"]
        for feature in marker_inventory_output["features"]
    )

    print(f"Features: {marker_inventory_output['feature_count']}")
    print(f"Markers: {marker_count}")

    if args.emit_all_output and args.marker_inventory_output:
        args.marker_inventory_output.parent.mkdir(parents=True, exist_ok=True)
        args.marker_inventory_output.write_text(
            yaml.dump(
                marker_inventory_output,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            ),
            encoding="utf-8",
        )

        print(f"Feature marker inventory written to {args.marker_inventory_output}")

    if args.emit_all_output and args.branch_points_output:
        branch_points_output = write_feature_branch_points(
            inventories=inventories,
            output_path=args.branch_points_output,
        )

        branch_point_count = sum(
            feature["branch_point_count"]
            for feature in branch_points_output["features"]
        )

        print(f"Feature branch points written to {args.branch_points_output}")
        print(f"Branch points: {branch_point_count}")

    if args.emit_all_output and args.converge_points_output:
        converge_points_output = write_feature_converge_points(
            inventories=inventories,
            output_path=args.converge_points_output,
        )

        converge_point_count = sum(
            feature["converge_point_count"]
            for feature in converge_points_output["features"]
        )

        print(f"Feature converge points written to {args.converge_points_output}")
        print(f"Converge points: {converge_point_count}")

    graphs = load_graph(args.cfg, args.cfg_format)

    if graphs is None:
        raise ValueError(f"Could not load Stage 1 EI CFG from {args.cfg}")

    if args.summarize_flow_cases:
        summarize_feature_flow_cases(
            graphs,
            inventories,
        )

    case_output = write_feature_flow_cases(
        graphs=graphs,
        inventories=inventories,
        output_path=args.output,
        verbose=args.verbose,
    )

    print(f"Feature flow cases written to {args.output}")
    print(f"Feature flow cases: {case_output['case_count']}")
    print(f"Unresolved feature flow cases: {case_output['unresolved_case_count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
