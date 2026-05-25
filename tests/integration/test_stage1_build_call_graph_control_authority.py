from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pytest
from pybastion_integration.stages import stage1_build_call_graph as graph_builder

pytestmark = pytest.mark.integration


@dataclass
class FakeCallableContext:
    callable_id: str = "callable:fixture"
    callable_name: str = "fixture"
    callable_kind: str = "function"
    callable_fqn: str = "synthetic.fixture"
    unit_name: str = "synthetic"
    unit_fqn: str = "synthetic"
    entry: dict[str, Any] = field(default_factory=dict)
    integration_by_ei: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def make_context(
    execution_items: list[dict[str, Any]],
    *,
    control_flow: dict[str, Any] | None = None,
) -> FakeCallableContext:
    return FakeCallableContext(
        entry={
            "analysis_info": {
                "execution_items": execution_items,
                "control_flow": control_flow or {},
            }
        }
    )


def edge_types(cfg: nx.DiGraph) -> set[str]:
    return {data.get("edge_type") for _, _, data in cfg.edges(data=True)}


def edges_by_type(
    cfg: nx.DiGraph, edge_type: str
) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (source, target, data)
        for source, target, data in cfg.edges(data=True)
        if data.get("edge_type") == edge_type
    ]


def test_context_execution_items_reads_current_stage3_field_only() -> None:
    context = make_context(
        [
            {"id": "fixture_E0001", "stmt_type": "assign"},
            "not-an-execution-item",
            {"id": "fixture_E0002", "stmt_type": "return"},
        ]
    )

    assert graph_builder.context_execution_items(context) == [
        {"id": "fixture_E0001", "stmt_type": "assign"},
        {"id": "fixture_E0002", "stmt_type": "return"},
    ]


def test_context_execution_items_does_not_fall_back_to_branches() -> None:
    context = FakeCallableContext(
        entry={"analysis_info": {}},
    )
    context.branches = [{"id": "fixture_E9999"}]  # type: ignore[attr-defined]

    assert graph_builder.context_execution_items(context) == []


def test_add_explicit_within_callable_edges_diagnostic_mode_keeps_inventory_fields_fixed() -> (
    None
):
    context = make_context(
        [
            {
                "id": "fixture_E0001",
                "statement_outcome": {
                    "target_ei": "fixture_E0002",
                    "is_terminal": False,
                },
                "conditional_targets": [
                    {"target_ei": "fixture_E0003", "is_terminal": False}
                ],
                "disruptive_outcomes": [
                    {"target_ei": "fixture_E0004", "is_terminal": False}
                ],
            }
        ]
    )
    cfg = nx.DiGraph()

    graph_builder.add_explicit_within_callable_edges(
        cfg,
        context,
        modeled_call_site_eis=set(),
    )

    assert edge_types(cfg) == {
        "diagnostic_statement_outcome",
        "diagnostic_conditional_target",
        "diagnostic_disruptive_outcome",
    }
    assert edges_by_type(cfg, "diagnostic_statement_outcome")[0][:2] == (
        "fixture_E0001",
        "fixture_E0002",
    )
    assert edges_by_type(cfg, "diagnostic_conditional_target")[0][:2] == (
        "fixture_E0001",
        "fixture_E0003",
    )
    assert edges_by_type(cfg, "diagnostic_disruptive_outcome")[0][:2] == (
        "fixture_E0001",
        "fixture_E0004",
    )
    assert "statement_outcome" not in edge_types(cfg)
    assert "conditional_target" not in edge_types(cfg)
    assert "disruptive_outcome" not in edge_types(cfg)


def test_add_explicit_within_callable_edges_skips_modeled_call_sites_in_both_modes() -> (
    None
):
    context = make_context(
        [
            {
                "id": "fixture_E0001",
                "statement_outcome": {
                    "target_ei": "fixture_E0002",
                    "is_terminal": False,
                },
                "conditional_targets": [
                    {"target_ei": "fixture_E0003", "is_terminal": False}
                ],
                "disruptive_outcomes": [
                    {"target_ei": "fixture_E0004", "is_terminal": False}
                ],
            }
        ]
    )

    for diagnostic_only in (False, True):
        cfg = nx.DiGraph()
        graph_builder.add_explicit_within_callable_edges(
            cfg,
            context,
            modeled_call_site_eis={"fixture_E0001"},
        )
        assert cfg.number_of_edges() == 0


def test_add_control_flow_nodes_and_edges_adds_region_policy_and_route_layer() -> None:
    context = make_context(
        [{"id": "fixture_E0001"}],
        control_flow={
            "regions": [
                {
                    "id": "if:10:true_body",
                    "owner_id": "if:10",
                    "kind": "body",
                    "source_construct": "if",
                    "start_line": 11,
                    "end_line": 12,
                },
                {
                    "id": "if:10:false_body",
                    "owner_id": "if:10",
                    "kind": "body",
                    "source_construct": "if",
                    "start_line": 13,
                    "end_line": 14,
                },
            ],
            "routes": [
                {
                    "id": "if:10:route:true",
                    "kind": "conditional_true",
                    "owner_id": "if:10",
                    "source_region_id": "if:10:true_body",
                    "target_region_id": "if:10:false_body",
                    "condition": "x > 0",
                    "condition_result": True,
                },
                {
                    "id": "if:10:route:completion",
                    "kind": "normal_completion",
                    "owner_id": "if:10",
                    "source_region_id": "if:10:false_body",
                    "target_line": 20,
                },
            ],
            "policies": [
                {
                    "owner_id": "with:30",
                    "mechanism_kind": "with_exit",
                    "region_id": "if:10:true_body",
                    "target_region_id": "if:10:false_body",
                    "applies_to": ["normal", "return", "raise"],
                    "preserves_prior_outcome": True,
                }
            ],
        },
    )
    cfg = nx.DiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)

    categories = {data.get("category") for _, data in cfg.nodes(data=True)}
    assert "callable" in categories
    assert "control_region" in categories
    assert "post_execution_policy" in categories
    assert "control_line_target" in categories

    assert "control_route" in edge_types(cfg)
    assert "owns_control_region" in edge_types(cfg)
    assert "owns_post_execution_policy" in edge_types(cfg)
    assert "policy_applies_to_region" in edge_types(cfg)
    assert "policy_targets_region" in edge_types(cfg)

    route_edges = edges_by_type(cfg, "control_route")
    assert len(route_edges) == 2
    assert any(data.get("route_id") == "if:10:route:true" for _, _, data in route_edges)
    assert any(
        data.get("route_id") == "if:10:route:completion"
        and data.get("target_line") == 20
        for _, _, data in route_edges
    )


def test_control_route_target_line_resolves_to_execution_item() -> None:
    context = make_context(
        execution_items=[
            {
                "id": "unit.fn_E0001",
                "line": 10,
            },
            {
                "id": "unit.fn_E0002",
                "line": 20,
            },
        ],
        control_flow={
            "regions": [
                {
                    "id": "try:10:post_execution",
                    "owner_id": "try:10",
                    "kind": "post_execution",
                    "source_construct": "finally",
                }
            ],
            "routes": [
                {
                    "id": "try:10:route:resume_after_finally",
                    "kind": "resume_prior_outcome",
                    "owner_id": "try:10",
                    "source_region_id": "try:10:post_execution",
                    "target_line": 20,
                    "synthetic": True,
                    "preserves_prior_outcome": True,
                }
            ],
            "policies": [],
        },
    )

    cfg = nx.DiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_callable_nodes(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)

    source_node = graph_builder.control_region_node_id(
        context,
        "try:10:post_execution",
    )

    assert cfg.has_edge(source_node, "unit.fn_E0002")

    edge_data = cfg.get_edge_data(source_node, "unit.fn_E0002")
    assert edge_data["edge_type"] == "control_route"
    assert edge_data["route_id"] == "try:10:route:resume_after_finally"
    assert edge_data["resolved_target_kind"] == "execution_item"
    assert edge_data["target_line"] == 20

    line_target_nodes = [
        node
        for node, data in cfg.nodes(data=True)
        if data.get("category") == "control_line_target"
    ]

    assert line_target_nodes == []


def test_control_route_target_line_falls_back_to_line_placeholder_when_no_execution_item_exists() -> (
    None
):
    context = make_context(
        execution_items=[
            {
                "id": "unit.fn_E0001",
                "line": 10,
            },
        ],
        control_flow={
            "regions": [
                {
                    "id": "try:10:post_execution",
                    "owner_id": "try:10",
                    "kind": "post_execution",
                    "source_construct": "finally",
                }
            ],
            "routes": [
                {
                    "id": "try:10:route:resume_after_finally",
                    "kind": "resume_prior_outcome",
                    "owner_id": "try:10",
                    "source_region_id": "try:10:post_execution",
                    "target_line": 20,
                    "synthetic": True,
                    "preserves_prior_outcome": True,
                }
            ],
            "policies": [],
        },
    )

    cfg = nx.DiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_callable_nodes(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)

    source_node = graph_builder.control_region_node_id(
        context,
        "try:10:post_execution",
    )

    line_target_nodes = [
        node
        for node, data in cfg.nodes(data=True)
        if data.get("category") == "control_line_target"
    ]

    assert len(line_target_nodes) == 1

    target_node = line_target_nodes[0]
    assert cfg.has_edge(source_node, target_node)

    edge_data = cfg.get_edge_data(source_node, target_node)
    assert edge_data["edge_type"] == "control_route"
    assert edge_data["route_id"] == "try:10:route:resume_after_finally"
    assert edge_data["resolved_target_kind"] == "line_placeholder"
    assert edge_data["target_line"] == 20

    target_data = cfg.nodes[target_node]
    assert target_data["category"] == "control_line_target"
    assert target_data["target_line"] == 20
    assert target_data["route_id"] == "try:10:route:resume_after_finally"


def test_control_region_links_to_execution_items_in_line_range() -> None:
    context = make_context(
        execution_items=[
            {"id": "unit.fn_E0001", "line": 10},
            {"id": "unit.fn_E0002", "line": 11},
            {"id": "unit.fn_E0003", "line": 20},
        ],
        control_flow={
            "regions": [
                {
                    "id": "if:10:true_body",
                    "owner_id": "if:10",
                    "kind": "body",
                    "start_line": 10,
                    "end_line": 11,
                }
            ],
            "routes": [],
            "policies": [],
        },
    )

    cfg = nx.DiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_callable_nodes(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)

    region_node = graph_builder.control_region_node_id(
        context,
        "if:10:true_body",
    )

    assert cfg.has_edge(region_node, "unit.fn_E0001")
    assert cfg.has_edge(region_node, "unit.fn_E0002")
    assert not cfg.has_edge(region_node, "unit.fn_E0003")

    assert (
        cfg.get_edge_data(region_node, "unit.fn_E0001")["edge_type"]
        == "control_region_contains_execution_item"
    )

    assert cfg.has_edge("unit.fn_E0001", region_node)
    assert (
        cfg.get_edge_data("unit.fn_E0001", region_node)["edge_type"]
        == "execution_item_in_control_region"
    )


def test_region_to_region_control_route_adds_derived_execution_item_edge() -> None:
    context = make_context(
        execution_items=[
            {"id": "unit.fn_E0001", "line": 10},
            {"id": "unit.fn_E0002", "line": 20},
        ],
        control_flow={
            "regions": [
                {
                    "id": "if:10:condition",
                    "owner_id": "if:10",
                    "kind": "condition",
                    "start_line": 10,
                    "end_line": 10,
                },
                {
                    "id": "if:10:true_body",
                    "owner_id": "if:10",
                    "kind": "body",
                    "start_line": 20,
                    "end_line": 20,
                },
            ],
            "routes": [
                {
                    "id": "if:10:route:true",
                    "kind": "conditional_true",
                    "owner_id": "if:10",
                    "source_region_id": "if:10:condition",
                    "target_region_id": "if:10:true_body",
                    "condition_result": True,
                }
            ],
            "policies": [],
        },
    )

    cfg = nx.DiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_callable_nodes(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)

    assert cfg.has_edge("unit.fn_E0001", "unit.fn_E0002")

    edge_data = cfg.get_edge_data("unit.fn_E0001", "unit.fn_E0002")
    assert edge_data["edge_type"] == "derived_control_route_execution_item"
    assert edge_data["route_id"] == "if:10:route:true"
    assert edge_data["route_kind"] == "conditional_true"
    assert edge_data["source_region_id"] == "if:10:condition"
    assert edge_data["target_region_id"] == "if:10:true_body"


def test_target_line_control_route_adds_derived_execution_item_edge() -> None:
    context = make_context(
        execution_items=[
            {"id": "unit.fn_E0001", "line": 10},
            {"id": "unit.fn_E0002", "line": 20},
        ],
        control_flow={
            "regions": [
                {
                    "id": "try:10:post_execution",
                    "owner_id": "try:10",
                    "kind": "post_execution",
                    "start_line": 10,
                    "end_line": 10,
                }
            ],
            "routes": [
                {
                    "id": "try:10:route:resume_after_finally",
                    "kind": "resume_prior_outcome",
                    "owner_id": "try:10",
                    "source_region_id": "try:10:post_execution",
                    "target_line": 20,
                    "synthetic": True,
                    "preserves_prior_outcome": True,
                }
            ],
            "policies": [],
        },
    )

    cfg = nx.DiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_callable_nodes(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)

    assert cfg.has_edge("unit.fn_E0001", "unit.fn_E0002")

    edge_data = cfg.get_edge_data("unit.fn_E0001", "unit.fn_E0002")
    assert edge_data["edge_type"] == "derived_control_route_execution_item"
    assert edge_data["route_id"] == "try:10:route:resume_after_finally"
    assert edge_data["route_kind"] == "resume_prior_outcome"
    assert edge_data["source_region_id"] == "try:10:post_execution"
    assert edge_data["target_line"] == 20
    assert edge_data["resolved_target_kind"] == "execution_item"


def test_terminal_control_route_adds_derived_execution_item_terminal_edge() -> None:
    context = make_context(
        execution_items=[
            {"id": "unit.fn_E0001", "line": 10},
        ],
        control_flow={
            "regions": [
                {
                    "id": "with:10:body",
                    "owner_id": "with:10",
                    "kind": "body",
                    "start_line": 10,
                    "end_line": 10,
                }
            ],
            "routes": [
                {
                    "id": "with:10:route:return",
                    "kind": "function_return",
                    "owner_id": "with:10",
                    "source_region_id": "with:10:body",
                    "exit_kind": "return",
                    "synthetic": True,
                }
            ],
            "policies": [],
        },
    )

    cfg = nx.DiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_callable_nodes(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)

    terminal_nodes = [
        node
        for node, data in cfg.nodes(data=True)
        if data.get("category") == "control_terminal_target"
    ]

    assert len(terminal_nodes) == 1

    terminal_node = terminal_nodes[0]

    assert cfg.has_edge("unit.fn_E0001", terminal_node)

    edge_data = cfg.get_edge_data("unit.fn_E0001", terminal_node)
    assert edge_data["edge_type"] == "derived_control_route_execution_item_terminal"
    assert edge_data["route_id"] == "with:10:route:return"
    assert edge_data["route_kind"] == "function_return"
    assert edge_data["source_region_id"] == "with:10:body"
    assert edge_data["exit_kind"] == "return"
    assert edge_data["resolved_target_kind"] == "terminal_placeholder"


def test_every_control_route_with_source_execution_items_gets_derived_execution_item_edge() -> (
    None
):
    context = make_context(
        execution_items=[
            {"id": "unit.fn_E0001", "line": 10},
            {"id": "unit.fn_E0002", "line": 20},
            {"id": "unit.fn_E0003", "line": 30},
        ],
        control_flow={
            "regions": [
                {
                    "id": "if:10:condition",
                    "owner_id": "if:10",
                    "kind": "condition",
                    "start_line": 10,
                    "end_line": 10,
                },
                {
                    "id": "if:10:true_body",
                    "owner_id": "if:10",
                    "kind": "body",
                    "start_line": 20,
                    "end_line": 20,
                },
                {
                    "id": "try:30:post_execution",
                    "owner_id": "try:30",
                    "kind": "post_execution",
                    "start_line": 30,
                    "end_line": 30,
                },
            ],
            "routes": [
                {
                    "id": "if:10:route:true",
                    "kind": "conditional_true",
                    "owner_id": "if:10",
                    "source_region_id": "if:10:condition",
                    "target_region_id": "if:10:true_body",
                },
                {
                    "id": "try:30:route:resume",
                    "kind": "resume_prior_outcome",
                    "owner_id": "try:30",
                    "source_region_id": "try:30:post_execution",
                    "target_line": 20,
                    "synthetic": True,
                    "preserves_prior_outcome": True,
                },
                {
                    "id": "try:30:route:return",
                    "kind": "function_return",
                    "owner_id": "try:30",
                    "source_region_id": "try:30:post_execution",
                    "exit_kind": "return",
                    "synthetic": True,
                },
            ],
            "policies": [],
        },
    )

    cfg = nx.DiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_callable_nodes(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)

    derived_route_ids = {
        data.get("route_id")
        for _, _, data in cfg.edges(data=True)
        if data.get("edge_type")
        in {
            "derived_control_route_execution_item",
            "derived_control_route_execution_item_terminal",
        }
    }

    assert derived_route_ids == {
        "if:10:route:true",
        "try:30:route:resume",
        "try:30:route:return",
    }


def test_parallel_edges_preserve_diagnostic_and_derived_control_routes() -> None:
    context = make_context(
        execution_items=[
            {
                "id": "unit.fn_E0001",
                "line": 10,
                "statement_outcome": {
                    "target_ei": "unit.fn_E0002",
                    "is_terminal": False,
                },
            },
            {
                "id": "unit.fn_E0002",
                "line": 20,
            },
        ],
        control_flow={
            "regions": [
                {
                    "id": "if:10:true_body",
                    "owner_id": "if:10",
                    "kind": "body",
                    "start_line": 10,
                    "end_line": 10,
                }
            ],
            "routes": [
                {
                    "id": "if:10:route:true_body_completion",
                    "kind": "normal_completion",
                    "owner_id": "if:10",
                    "source_region_id": "if:10:true_body",
                    "target_line": 20,
                }
            ],
            "policies": [],
        },
    )

    cfg = nx.MultiDiGraph()

    graph_builder.add_callable_container_node(cfg, context)
    graph_builder.add_callable_nodes(cfg, context)
    graph_builder.add_control_flow_nodes_and_edges(cfg, context)
    graph_builder.add_explicit_within_callable_edges(
        cfg,
        context,
        modeled_call_site_eis=set(),
    )

    edge_types = {
        data.get("edge_type")
        for data in cfg.get_edge_data("unit.fn_E0001", "unit.fn_E0002").values()
    }

    assert "derived_control_route_execution_item" in edge_types
    assert "diagnostic_statement_outcome" in edge_types
