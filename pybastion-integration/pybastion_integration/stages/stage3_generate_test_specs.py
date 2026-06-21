#!/usr/bin/env python3
"""
Stage 3: Generate Integration Test Specifications

Inputs:
  - Stage 1 inventory-backed EI call graph
  - Unit inventory files
  - Stage 2 completed feature-flow cases

Outputs:
  - Seam-scoped integration test specifications
  - Feature-flow-scoped integration test specifications

Seam-scoped specs describe integration obligations at unit boundaries. They
provide enough inventory-backed path, source, target, fixture, and uncertainty
context to write tests proving that adjacent units work together across a seam.

Feature-flow-scoped specs describe end-to-end feature obligations discovered by
Stage 2. They consume completed feature-flow cases directly and preserve the
case identity, branch path, end condition, path evidence, integration-relevant
operations, fixture constraints, and remaining uncertainty needed to write
integration tests for the described feature behavior.

Stage 3 must not recompute feature-flow paths. Feature-flow spec generation uses
the completed Stage 2 feature-flow cases as its path authority.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from pybastion_integration import config
from pybastion_integration.utils.inventory_index import (
    CallableContext,
    InventoryIndex,
    execution_items_constraint,
    execution_items_description,
    execution_items_statement_outcome,
    build_ei_details_index,
    discover_inventory_files,
    load_all_inventories,
    signature_info,
)

DEFAULT_SEAM_TYPES: tuple[str, ...] = ("interunit", "boundary")


# =============================================================================
# Graph loading and indexing
# =============================================================================


def infer_graph_format(path: Path, explicit_format: str | None) -> str:
    if explicit_format:
        return explicit_format

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    if suffix in {".yaml", ".yml"}:
        return "yaml"

    raise ValueError(
        f"Cannot infer graph format from extension for {path}. "
        "Pass --graph-format explicitly."
    )


def load_graph(path: Path, graph_format: str) -> nx.MultiDiGraph:
    if graph_format == "pickle":
        with open(path, "rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError(
                f"Expected NetworkX MultiDiGraph in {path}, got {type(graph)!r}"
            )
        return graph

    if graph_format == "yaml":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return nx.node_link_graph(data, edges="edges")

    raise ValueError(f"Unsupported graph format: {graph_format}")


def graph_nodes_by_id(cfg: nx.MultiDiGraph) -> dict[str, dict[str, Any]]:
    return {
        str(node_id): {"id": str(node_id), **dict(data)}
        for node_id, data in cfg.nodes(data=True)
    }


def graph_edges_by_source(cfg: nx.MultiDiGraph) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source, target, data in cfg.edges(data=True):
        edge = {
            "from": str(source),
            "to": str(target),
            **dict(data),
        }
        edge.setdefault("id", make_edge_id(str(source), str(target), edge))
        result[str(source)].append(edge)

    return dict(result)


def graph_edges(cfg: nx.MultiDiGraph) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []

    for source, target, data in cfg.edges(data=True):
        edge = {
            "from": str(source),
            "to": str(target),
            **dict(data),
        }
        edge.setdefault("id", make_edge_id(str(source), str(target), edge))
        edges.append(edge)

    return edges


def make_edge_id(source: str, target: str, edge: dict[str, Any]) -> str:
    basis = "|".join(
        [
            source,
            target,
            str(edge.get("edge_type", "")),
            str(edge.get("call_kind", "")),
            str(edge.get("target_callable_id", "")),
            str(edge.get("target_fqn", "")),
            str(edge.get("resolved_target", "")),
            str(edge.get("operation_target", "")),
        ]
    )
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12].upper()
    return f"EDGE_{digest}"


# =============================================================================
# Classification helpers
# =============================================================================


def classification_kind(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "unknown"

    classification = candidate.get("classification")

    if isinstance(classification, dict):
        if not classification.get("is_integration", False):
            return "unknown"

        return (
            classification.get("seam_kind")
            or classification.get("seam_type")
            or classification.get("integration_kind")
            or classification.get("integration_type")
            or classification.get("kind")
            or classification.get("type")
            or candidate.get("seam_kind")
            or candidate.get("seam_type")
            or candidate.get("integration_kind")
            or candidate.get("integration_type")
            or candidate.get("kind")
            or "unknown"
        )

    if isinstance(classification, str):
        return classification

    return (
        candidate.get("seam_kind")
        or candidate.get("seam_type")
        or candidate.get("integration_kind")
        or candidate.get("integration_type")
        or candidate.get("kind")
        or "unknown"
    )


def normalized_seam_kind(kind: str | None) -> str:
    value = (kind or "unknown").strip().lower()

    match value:
        case "project" | "project_callable" | "inter_unit" | "inter-unit":
            return "interunit"
        case "external" | "external_library" | "third_party" | "third-party":
            return "extlib"
        case "builtin" | "builtins":
            return "stdlib"
        case _:
            return value


def candidate_target(candidate: dict[str, Any] | None) -> str | None:
    if not candidate:
        return None
    return candidate.get("resolved_target") or candidate.get("target")


def candidate_signature(candidate: dict[str, Any] | None) -> str | None:
    if not candidate:
        return None
    return candidate.get("signature")


def candidate_execution_paths(
    candidate: dict[str, Any] | None, source_ei: str
) -> list[list[str]]:
    if not candidate:
        return [[source_ei]]

    paths = candidate.get("execution_paths") or candidate.get("executionPaths") or []
    normalized: list[list[str]] = []

    for path in paths:
        if isinstance(path, list) and path:
            normalized.append([str(ei_id) for ei_id in path])

    return normalized or [[source_ei]]


def outgoing_call_edges_by_source_ei(
    cfg: nx.MultiDiGraph,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for edge in graph_edges(cfg):
        if edge.get("edge_type") == "call":
            result[edge["from"]].append(edge)

    return dict(result)


def choose_matching_graph_edge(
    *,
    source_ei: str,
    candidate: dict[str, Any],
    outgoing_edges_by_source_ei: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    edges = outgoing_edges_by_source_ei.get(source_ei, [])
    if not edges:
        return None

    candidate_targets = {
        candidate.get("resolved_target"),
        candidate.get("target"),
        candidate.get("operation_target"),
    }
    candidate_targets = {str(target) for target in candidate_targets if target}

    for edge in edges:
        edge_targets = {
            edge.get("resolved_target"),
            edge.get("target_fqn"),
            edge.get("operation_target"),
        }
        edge_targets = {str(target) for target in edge_targets if target}

        if candidate_targets and edge_targets and candidate_targets & edge_targets:
            return edge

    if len(edges) == 1:
        return edges[0]

    return None


def target_context_for_edge(
    *,
    edge: dict[str, Any] | None,
    inventory_index: InventoryIndex,
    nodes_by_id: dict[str, dict[str, Any]],
) -> tuple[CallableContext | None, dict[str, Any]]:
    if edge is None:
        return None, {}

    target_node = nodes_by_id.get(edge["to"], {})
    target_callable_id = edge.get("target_callable_id") or target_node.get(
        "callable_id"
    )

    if not target_callable_id:
        return None, target_node

    return inventory_index.callable_contexts.get(target_callable_id), target_node


def should_include_inventory_candidate(
    *,
    candidate: dict[str, Any],
    seam_types: set[str],
) -> tuple[bool, str]:
    seam_kind = normalized_seam_kind(classification_kind(candidate))
    return seam_kind in seam_types, seam_kind


def discover_inventory_seams(
    *,
    cfg: nx.MultiDiGraph,
    inventory_index: InventoryIndex,
    nodes_by_id: dict[str, dict[str, Any]],
    seam_types: set[str],
    include_same_unit: bool,
    include_collapsed: bool,
) -> list[SeamSource]:
    outgoing_edges_by_source_ei = outgoing_call_edges_by_source_ei(cfg)
    seams: list[SeamSource] = []

    for source_context in inventory_index.callable_contexts.values():
        for source_ei, candidates in source_context.integration_by_ei.items():
            for candidate in candidates:
                include, seam_kind = should_include_inventory_candidate(
                    candidate=candidate,
                    seam_types=seam_types,
                )

                if not include:
                    continue

                edge = choose_matching_graph_edge(
                    source_ei=source_ei,
                    candidate=candidate,
                    outgoing_edges_by_source_ei=outgoing_edges_by_source_ei,
                )

                target_context, target_node = target_context_for_edge(
                    edge=edge,
                    inventory_index=inventory_index,
                    nodes_by_id=nodes_by_id,
                )

                if (
                    edge is not None
                    and target_node.get("category") == "collapsed_internal_operation"
                    and not include_collapsed
                ):
                    continue

                if (
                    not include_same_unit
                    and target_context is not None
                    and target_context.unit_fqn == source_context.unit_fqn
                    and target_context.callable_id != source_context.callable_id
                ):
                    continue

                seams.append(
                    SeamSource(
                        source_ei=source_ei,
                        target_node_id=edge["to"] if edge else None,
                        edge=edge,
                        candidate=candidate,
                        seam_kind=seam_kind,
                        source_context=source_context,
                        target_context=target_context,
                        target_node=target_node,
                    )
                )

    return seams


# =============================================================================
# Path categorization
# =============================================================================


def analyze_target_outcomes(target_context: CallableContext | None) -> dict[str, Any]:
    if target_context is None:
        return {
            "has_exceptions": False,
            "exception_execution_items": [],
            "success_execution_items": [],
            "total_execution_items": 0,
        }

    exception_execution_items: list[dict[str, Any]] = []
    success_execution_items: list[dict[str, Any]] = []

    for ei in target_context.execution_items:
        description = execution_items_description(ei).lower()
        statement_outcome = execution_items_statement_outcome(ei)
        terminates_via = statement_outcome.get("terminates_via") or ei.get(
            "terminates_via"
        )
        is_terminal = bool(
            statement_outcome.get("is_terminal", ei.get("is_terminal", False))
        )

        is_exception = (
            "exception propagates" in description
            or "raises" in description
            or terminates_via in {"raise", "exception"}
        )

        if is_exception:
            exception_execution_items.append(ei)
        elif "returns" in description or not is_terminal:
            success_execution_items.append(ei)

    return {
        "has_exceptions": len(exception_execution_items) > 0,
        "exception_execution_items": exception_execution_items,
        "success_execution_items": success_execution_items,
        "total_execution_items": len(target_context.execution_items),
    }


def categorize_path(
    path_eis: list[str],
    ei_details: dict[str, dict[str, Any]],
    target_outcomes: dict[str, Any],
) -> dict[str, Any]:
    has_validation_failure = False
    has_empty_iteration = False
    has_boundary_condition = False
    has_alternative_execution_item = False
    has_exception_path = False

    for ei_id in path_eis:
        ei = ei_details.get(ei_id, {})
        description = execution_items_description(ei).lower()
        condition = str(ei.get("condition") or "").lower()
        constraint = execution_items_constraint(ei)
        constraint_type = constraint.get("constraint_type")
        statement_outcome = execution_items_statement_outcome(ei)
        terminates_via = statement_outcome.get("terminates_via") or ei.get(
            "terminates_via"
        )

        if "validation" in description or "invalid" in description:
            has_validation_failure = True

        if constraint_type == "iteration" and "0 iterations" in description:
            has_empty_iteration = True
            has_boundary_condition = True

        if " is none" in condition or " is not none" in condition:
            has_boundary_condition = True

        if constraint_type == "condition":
            has_alternative_execution_item = True

        if terminates_via in {"raise", "exception"} or "raises" in description:
            has_exception_path = True

    if has_exception_path or has_validation_failure:
        category = "error_handling"
        subcategory = (
            "validation_failure" if has_validation_failure else "exception_path"
        )
    elif has_empty_iteration or has_boundary_condition:
        category = "edge_cases"
        subcategory = (
            "empty_collection" if has_empty_iteration else "boundary_condition"
        )
    elif has_alternative_execution_item:
        category = "alternative_flows"
        subcategory = "conditional_execution_items"
    else:
        category = "happy_path"
        subcategory = "success"

    return {
        "category": category,
        "subcategory": subcategory,
        "has_validation_failure": has_validation_failure,
        "has_boundary_condition": has_boundary_condition,
        "target_can_raise": target_outcomes.get("has_exceptions", False),
        "path_length": len(path_eis),
    }


def find_representative_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for path in paths:
        key = (path["category"], path["subcategory"])
        groups.setdefault(key, []).append(path)

    representatives: list[dict[str, Any]] = []

    for (category, subcategory), group_paths in groups.items():
        if category == "happy_path":
            shortest = min(group_paths, key=lambda p: p["path_length"])
            representatives.append(
                {
                    **shortest,
                    "represents_count": len(group_paths),
                    "representative_of": f"{category}/{subcategory}",
                }
            )

        elif category == "error_handling":
            representatives.append(
                {
                    **group_paths[0],
                    "represents_count": len(group_paths),
                    "representative_of": f"{category}/{subcategory}",
                }
            )

        elif category == "edge_cases":
            for path in group_paths:
                representatives.append(
                    {
                        **path,
                        "represents_count": 1,
                        "representative_of": f"{category}/{subcategory}",
                    }
                )

        else:
            for path in group_paths[:3]:
                representatives.append(
                    {
                        **path,
                        "represents_count": (
                            len(group_paths) // 3 if len(group_paths) > 3 else 1
                        ),
                        "representative_of": f"{category}/{subcategory}",
                    }
                )

    return representatives


def build_synthetic_error_representative(
    target_outcomes: dict[str, Any],
) -> dict[str, Any] | None:
    exception_execution_items = target_outcomes.get("exception_execution_items", [])
    if not exception_execution_items:
        return None

    primary = next(
        (
            ei
            for ei in exception_execution_items
            if (
                (
                    execution_items_statement_outcome(ei).get("terminates_via")
                    or ei.get("terminates_via")
                )
                == "raise"
                and ei.get("condition")
            )
        ),
        exception_execution_items[0],
    )

    constraint = execution_items_constraint(primary)

    return {
        "path_id": "PATH_SYNTHETIC_ERROR",
        "eis": [],
        "eis_original": [],
        "category": "error_handling",
        "subcategory": "triggers_target_exception",
        "synthetic": True,
        "has_validation_failure": False,
        "has_boundary_condition": False,
        "target_can_raise": True,
        "path_length": 0,
        "represents_count": 1,
        "representative_of": "error_handling/triggers_target_exception",
        "precondition": {
            "description": (
                "Arrange target object state to satisfy the exception condition "
                "before invoking the source callable"
            ),
            "condition": primary.get("condition"),
            "outcome": primary.get("description") or primary.get("outcome"),
            "constraint_expr": constraint.get("expr"),
            "constraint_type": constraint.get("constraint_type"),
            "variables_read": constraint.get("variables_read", []),
        },
    }


# =============================================================================
# Fixture identification
# =============================================================================


def outgoing_call_edges_by_source(
    cfg: nx.MultiDiGraph,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for edge in graph_edges(cfg):
        if edge.get("edge_type") == "call":
            result[edge["from"]].append(edge)

    return dict(result)


def fixture_from_candidate(
    candidate: dict[str, Any],
    ei_id: str,
    reason: str,
) -> dict[str, Any]:
    kind = normalized_seam_kind(classification_kind(candidate))
    target = (
        candidate_target(candidate)
        or candidate.get("target")
        or candidate.get("operation_target")
    )

    return {
        "type": kind,
        "ei_id": ei_id,
        "integration_id": candidate.get("id"),
        "mock_target": candidate.get("operation_target") or target,
        "mock_target_fqn": candidate.get("resolved_target") or target,
        "signature": candidate_signature(candidate),
        "reason": reason,
    }


def identify_fixtures_for_path(
    *,
    path_eis: list[str],
    seam_ei_id: str,
    seam_candidate_id: str | None,
    source_context: CallableContext,
    outgoing_call_edges: dict[str, list[dict[str, Any]]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    for ei_id in path_eis:
        if ei_id == seam_ei_id:
            continue

        for candidate in source_context.integration_by_ei.get(ei_id, []) or []:
            candidate_id = candidate.get("id")
            if candidate_id and candidate_id == seam_candidate_id:
                continue

            kind = normalized_seam_kind(classification_kind(candidate))
            if kind not in {"interunit", "boundary", "extlib", "stdlib"}:
                continue

            target = candidate_target(candidate)
            key = (candidate_id, ei_id, target)
            if key in seen:
                continue

            fixtures.append(
                fixture_from_candidate(
                    candidate,
                    ei_id,
                    reason="Earlier integration candidate on the selected path; mock to isolate the seam under test",
                )
            )
            seen.add(key)

        for edge in outgoing_call_edges.get(ei_id, []) or []:
            target_node = nodes_by_id.get(edge["to"], {})
            if target_node.get("category") != "collapsed_internal_operation":
                continue

            key = (
                edge.get("target_callable_id"),
                ei_id,
                target_node.get("callable_fqn"),
            )
            if key in seen:
                continue

            fixtures.append(
                {
                    "type": "mechanical_operation",
                    "ei_id": ei_id,
                    "integration_id": None,
                    "mock_target": edge.get("operation_target")
                    or target_node.get("callable_name"),
                    "mock_target_fqn": target_node.get("callable_fqn"),
                    "callable_id": target_node.get("callable_id"),
                    "signature": edge.get("signature"),
                    "reason": "Collapsed mechanical or utility operation; fixture to avoid expanding unrelated internal behavior",
                }
            )
            seen.add(key)

    return fixtures


def merge_fixture_lists(
    fixtures_by_path: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    for fixtures in fixtures_by_path:
        for fixture in fixtures:
            key = (
                fixture.get("type"),
                fixture.get("integration_id") or fixture.get("callable_id"),
                fixture.get("mock_target_fqn") or fixture.get("mock_target"),
            )
            if key in seen:
                continue
            merged.append(fixture)
            seen.add(key)

    return merged


def mark_fixture_eis(
    path_eis: list[str], fixtures: list[dict[str, Any]], seam_ei_id: str
) -> list[str]:
    fixture_eis = {fixture.get("ei_id") for fixture in fixtures if fixture.get("ei_id")}

    marked: list[str] = []
    for ei_id in path_eis:
        if ei_id == seam_ei_id:
            marked.append(ei_id)
        elif ei_id in fixture_eis:
            marked.append(f"{ei_id}_FIXTURE")
        else:
            marked.append(ei_id)

    return marked


# =============================================================================
# Seam discovery
# =============================================================================


@dataclass(slots=True)
class SeamSource:
    source_ei: str
    target_node_id: str | None
    edge: dict[str, Any] | None
    candidate: dict[str, Any] | None
    seam_kind: str
    source_context: CallableContext
    target_context: CallableContext | None
    target_node: dict[str, Any]


def seam_identity(seam: SeamSource) -> str:
    candidate_id = seam.candidate.get("id") if seam.candidate else None
    if candidate_id:
        return str(candidate_id)

    basis = "|".join(
        [
            seam.source_ei,
            seam.target_node_id or "",
            seam.seam_kind,
            str(seam.edge.get("id") if seam.edge else ""),
            str(seam.edge.get("target_fqn") if seam.edge else ""),
            str(seam.edge.get("resolved_target") if seam.edge else ""),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16].upper()


# =============================================================================
# Spec generation
# =============================================================================


def make_spec_id(seam_id: str, counter: int) -> str:
    h = hashlib.sha256(seam_id.encode("utf-8")).hexdigest()[:8].upper()
    return f"ITEST_{counter:04d}_{h}"


def build_spec(
    *,
    seam: SeamSource,
    nodes_by_id: dict[str, dict[str, Any]],
    outgoing_call_edges: dict[str, list[dict[str, Any]]],
    ei_details: dict[str, dict[str, Any]],
    counter: int,
) -> dict[str, Any]:
    seam_id = seam_identity(seam)
    spec_id = make_spec_id(seam_id, counter)
    source_context = seam.source_context
    target_context = seam.target_context
    target_outcomes = analyze_target_outcomes(target_context)
    raw_paths = candidate_execution_paths(seam.candidate, seam.source_ei)

    categorized: list[dict[str, Any]] = []
    for index, path_eis in enumerate(raw_paths):
        cat_info = categorize_path(path_eis, ei_details, target_outcomes)
        categorized.append(
            {
                "path_id": f"PATH_{index + 1:03d}",
                "eis": path_eis,
                **cat_info,
            }
        )

    representative_paths = find_representative_paths(categorized) if categorized else []

    if target_outcomes.get("has_exceptions"):
        has_error_representative = any(
            path["category"] == "error_handling" for path in representative_paths
        )
        if not has_error_representative:
            synthetic = build_synthetic_error_representative(target_outcomes)
            if synthetic:
                representative_paths.append(synthetic)

    path_summary = {
        "happy_path": sum(1 for p in categorized if p["category"] == "happy_path"),
        "error_handling": sum(
            1 for p in categorized if p["category"] == "error_handling"
        ),
        "edge_cases": sum(1 for p in categorized if p["category"] == "edge_cases"),
        "alternative_flows": sum(
            1 for p in categorized if p["category"] == "alternative_flows"
        ),
    }

    fixtures_by_path: list[list[dict[str, Any]]] = []
    marked_paths: list[dict[str, Any]] = []
    seam_candidate_id = seam.candidate.get("id") if seam.candidate else None

    for path in representative_paths:
        original_eis = path.get("eis", []) or []

        fixtures = identify_fixtures_for_path(
            path_eis=original_eis,
            seam_ei_id=seam.source_ei,
            seam_candidate_id=seam_candidate_id,
            source_context=source_context,
            outgoing_call_edges=outgoing_call_edges,
            nodes_by_id=nodes_by_id,
        )

        fixtures_by_path.append(fixtures)

        marked_paths.append(
            {
                **path,
                "eis": mark_fixture_eis(original_eis, fixtures, seam.source_ei),
                "eis_original": original_eis,
            }
        )

    fixture_requirements = merge_fixture_lists(fixtures_by_path)
    edge = seam.edge or {}
    candidate = seam.candidate or {}
    source_node = nodes_by_id.get(seam.source_ei, {})
    target_node = seam.target_node or {}

    target_unit = (
        target_context.unit_name
        if target_context is not None
        else target_node.get("unit")
    )

    target_fqn = (
        target_context.callable_fqn
        if target_context is not None
        else (
            target_node.get("fully_qualified")
            or target_node.get("qualified_name")
            or candidate.get("resolved_target")
            or candidate.get("target")
            or candidate.get("operation_target")
        )
    )

    target_callable_id = (
        target_context.callable_id
        if target_context is not None
        else target_node.get("callable_id")
    )

    target_name = (
        target_context.callable_name
        if target_context is not None
        else (
            target_node.get("name")
            or candidate.get("target")
            or candidate.get("operation_target")
            or candidate.get("resolved_target")
        )
    )

    return {
        "spec_id": spec_id,
        "description": f"Test integration seam: {source_context.callable_fqn} → {target_fqn}",
        "test_type": "integration",
        "spec_kind": "seam",
        "integration_point": {
            "id": seam_candidate_id,
            "source_kind": (
                "graph_backed" if seam.edge is not None else "inventory_only"
            ),
            "edge_id": edge.get("id"),
            "seam_kind": seam.seam_kind,
            "classification": candidate.get("classification"),
            "target_raw": candidate.get("target") or edge.get("operation_target"),
            "resolved_target": candidate.get("resolved_target")
            or edge.get("resolved_target"),
            "operation_target": candidate.get("operation_target")
            or edge.get("operation_target"),
            "ei_id": candidate.get("ei_id") or seam.source_ei,
            "call_signature": candidate.get("signature") or edge.get("signature"),
        },
        "feasibility": {
            "representative_paths": len(representative_paths),
            "original_paths": len(raw_paths),
            "path_summary": path_summary,
        },
        "source": {
            "unit": source_context.unit_name,
            "unit_fqn": source_context.unit_fqn,
            "callable_id": source_context.callable_id,
            "name": source_context.callable_name,
            "fully_qualified": source_context.callable_fqn,
            "kind": source_context.callable_kind,
            "signature": signature_info(source_context.entry).get("signature"),
            "seam_ei": seam.source_ei,
            "source_line": source_node.get("line"),
            "source_condition": source_node.get("condition"),
            "source_description": source_node.get("description"),
            "representative_paths": marked_paths,
        },
        "target": {
            "unit": target_unit,
            "unit_fqn": target_context.unit_fqn if target_context is not None else None,
            "callable_id": target_callable_id,
            "name": target_name,
            "fully_qualified": target_fqn,
            "kind": (
                target_context.callable_kind
                if target_context is not None
                else target_node.get("category")
            ),
            "signature": (
                signature_info(target_context.entry).get("signature")
                if target_context is not None
                else edge.get("signature")
            ),
            "unknown": target_context is None,
            "outcome_analysis": target_outcomes,
        },
        "fixture_requirements": fixture_requirements,
    }


# =============================================================================
# Main
# =============================================================================


def typed_stage_output_path(base_output: Path, spec_type: str) -> Path:
    return base_output.with_name(f"{spec_type}-{base_output.name}")


def config_stage_output(stage: int) -> Path | None:
    getter = getattr(config, "get_stage_output", None)
    if getter is None:
        return None
    return getter(stage)


def config_inventories_root() -> Path | None:
    getter = getattr(config, "get_inventories_root", None)
    if getter is None:
        return None
    return getter()


def yaml_dump_options() -> dict[str, Any]:
    if config is None:
        return {
            "default_flow_style": False,
            "sort_keys": False,
            "width": 120,
            "indent": 2,
        }

    return {
        "default_flow_style": False,
        "sort_keys": config.get_yaml_sort_keys(),
        "width": config.get_yaml_width(),
        "indent": config.get_yaml_indent(),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Stage 1 graph output. Defaults to config.get_stage_output(1) when available.",
    )
    parser.add_argument(
        "--graph-format",
        choices=["pickle", "yaml"],
        default=None,
        help="Graph format. Inferred from file extension when omitted.",
    )
    parser.add_argument(
        "--inventories-root",
        type=Path,
        default=None,
        help="Root directory for unit inventory discovery.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file. Defaults to config.get_stage_output(3) when available.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Target project root. Passed to config.set_target_root when config is available.",
    )
    parser.add_argument(
        "--seam-types",
        nargs="+",
        default=list(DEFAULT_SEAM_TYPES),
        metavar="TYPE",
        help="Seam types to generate specs for. Default: interunit boundary",
    )
    parser.add_argument(
        "--include-same-unit",
        action="store_true",
        help="Include same-unit call edges as seams. Default: false.",
    )
    parser.add_argument(
        "--include-collapsed",
        action="store_true",
        help="Include collapsed mechanical/utility operation nodes as primary seams. Default: false.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.target_root and config is not None:
        config.set_target_root(args.target_root)
        if args.verbose:
            print(f"Target root: {args.target_root}")

    input_path = args.input or config_stage_output(1)
    if input_path is None:
        print(
            "ERROR: --input is required when config.get_stage_output(1) is unavailable",
            file=sys.stderr,
        )
        return 1

    inventories_root = args.inventories_root or config_inventories_root()
    if inventories_root is None:
        print(
            "ERROR: --inventories-root is required when config.get_inventories_root() is unavailable",
            file=sys.stderr,
        )
        return 1

    base_output_path = args.output or config_stage_output(3)
    if base_output_path is None:
        print(
            "ERROR: --output is required when config.get_stage_output(3) is unavailable",
            file=sys.stderr,
        )
        return 1

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: Stage 1 graph not found: {input_path}", file=sys.stderr)
        return 1

    inventories_root = Path(inventories_root)
    if not inventories_root.exists():
        print(f"ERROR: Inventories root not found: {inventories_root}", file=sys.stderr)
        return 1

    seam_output_path = typed_stage_output_path(base_output_path, "seam")
    feature_output_path = typed_stage_output_path(base_output_path, "feature")

    graph_format = infer_graph_format(input_path, args.graph_format)

    if args.verbose:
        print(f"Loading graph: {input_path} ({graph_format})")

    cfg = load_graph(input_path, graph_format)

    inventory_paths = discover_inventory_files(inventories_root)
    if not inventory_paths:
        print(
            f"ERROR: No inventory files found under {inventories_root}", file=sys.stderr
        )
        return 1

    if args.verbose:
        print(f"Loading {len(inventory_paths)} inventory file(s)")

    inventory_index = load_all_inventories(inventory_paths)
    ei_details = build_ei_details_index(inventory_index)
    nodes_by_id = graph_nodes_by_id(cfg)
    outgoing_call_edges = outgoing_call_edges_by_source(cfg)

    seam_types = {normalized_seam_kind(kind) for kind in args.seam_types}

    seams = discover_inventory_seams(
        cfg=cfg,
        inventory_index=inventory_index,
        nodes_by_id=nodes_by_id,
        seam_types=seam_types,
        include_same_unit=args.include_same_unit,
        include_collapsed=args.include_collapsed,
    )

    graph_backed_seams = sum(1 for seam in seams if seam.edge is not None)
    inventory_only_seams = len(seams) - graph_backed_seams

    if args.verbose:
        print(f"Callables indexed: {len(inventory_index.callable_contexts)}")
        print(f"EIs indexed: {len(ei_details)}")
        print(f"Graph nodes: {cfg.number_of_nodes()}")
        print(f"Graph edges: {cfg.number_of_edges()}")
        print(f"Inventory seam sources: {len(seams)}")
        print(f"Graph-backed seams: {graph_backed_seams}")
        print(f"Inventory-only seams: {inventory_only_seams}")

    specs: list[dict[str, Any]] = []
    total_original_paths = 0
    total_representative_paths = 0

    for index, seam in enumerate(seams, start=1):
        spec = build_spec(
            seam=seam,
            nodes_by_id=nodes_by_id,
            outgoing_call_edges=outgoing_call_edges,
            ei_details=ei_details,
            counter=index,
        )
        specs.append(spec)
        total_original_paths += spec["feasibility"]["original_paths"]
        total_representative_paths += spec["feasibility"]["representative_paths"]

    reduction_pct = (
        100 * (1 - total_representative_paths / total_original_paths)
        if total_original_paths
        else 0
    )

    total_fixtures = sum(len(spec.get("fixture_requirements", [])) for spec in specs)
    fixture_type_counts: dict[str, int] = defaultdict(int)
    category_counts: dict[str, int] = {
        "happy_path": 0,
        "error_handling": 0,
        "edge_cases": 0,
        "alternative_flows": 0,
    }
    seam_kind_counts: dict[str, int] = defaultdict(int)

    for spec in specs:
        seam_kind_counts[spec["integration_point"]["seam_kind"]] += 1

        for fixture in spec.get("fixture_requirements", []):
            fixture_type_counts[fixture.get("type", "unknown")] += 1

        for category, count in spec["feasibility"]["path_summary"].items():
            category_counts[category] = category_counts.get(category, 0) + count

    output_data = {
        "stage": "integration-seam-test-specs",
        "metadata": {
            "spec_count": len(specs),
            "seam_types": sorted(seam_types),
            "input_file": str(input_path),
            "graph_format": graph_format,
            "inventories_root": str(inventories_root),
            "inventory_count": len(inventory_paths),
            "inventory_seam_sources": len(seams),
            "graph_backed_seam_sources": graph_backed_seams,
            "inventory_only_seam_sources": inventory_only_seams,
            "total_original_paths": total_original_paths,
            "total_representative_paths": total_representative_paths,
            "reduction_percentage": round(reduction_pct, 1),
            "total_fixture_requirements": total_fixtures,
            "fixture_type_counts": dict(sorted(fixture_type_counts.items())),
            "category_breakdown": category_counts,
            "seam_kind_counts": dict(sorted(seam_kind_counts.items())),
        },
        "test_specs": specs,
    }

    base_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(base_output_path, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, **yaml_dump_options())

    print(
        f"\n✓ Generated {len(specs)} integration seam test specification(s) → {seam_output_path}"
    )
    print(
        f"\n✓ Generated {len(specs)} integration feature test specification(s) → {feature_output_path}"
    )
    print(f"  Inventory seam sources:    {len(seams)}")
    print(f"  Graph-backed seams:        {graph_backed_seams}")
    print(f"  Inventory-only seams:      {inventory_only_seams}")
    print(f"  Original paths:            {total_original_paths}")
    print(f"  Representative paths:      {total_representative_paths}")
    print(f"  Reduction:                 {reduction_pct:.1f}%")
    print(f"  Fixture requirements:      {total_fixtures}")

    if args.verbose:
        print("\n  Seam kinds:")
        for seam_kind, count in sorted(seam_kind_counts.items()):
            print(f"    {seam_kind}: {count}")

        print("\n  Path categories:")
        for category, count in sorted(
            category_counts.items(), key=lambda item: -item[1]
        ):
            print(f"    {category}: {count}")

        print("\n  Fixture types:")
        for fixture_type, count in sorted(fixture_type_counts.items()):
            print(f"    {fixture_type}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
