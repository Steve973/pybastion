from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        runner = candidate / "pybastion-unit/pybastion_unit/run_unit_analysis.py"
        fixture = candidate / "tests/fixtures/feature-flow-fixture"

        if runner.exists() and fixture.exists():
            return candidate

    raise AssertionError("could not locate project root from test file location")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
FIXTURE_ROOT = PROJECT_ROOT / "tests/fixtures/feature-flow-fixture"
RUNNER = PROJECT_ROOT / "pybastion-unit/pybastion_unit/run_unit_analysis.py"
INVENTORY_NAME = "synthetic_feature_flows.inventory.yaml"


def _inventory_candidates() -> list[Path]:
    return sorted(
        PROJECT_ROOT.rglob(INVENTORY_NAME),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def _run_feature_flow_analysis() -> Path:
    before = {path: path.stat().st_mtime_ns for path in _inventory_candidates()}

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(FIXTURE_ROOT),
            "--readiness",
            "-v",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, (
        "feature-flow unit analysis failed\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    candidates = _inventory_candidates()
    changed = [
        path
        for path in candidates
        if path not in before or path.stat().st_mtime_ns != before[path]
    ]

    if changed:
        return changed[0]

    assert candidates, f"no {INVENTORY_NAME} found under {PROJECT_ROOT}"
    return candidates[0]


def _route_by_kind(entry: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [route for route in _routes(entry) if route.get("kind") == kind]


def _owner_for_region_kind(
    entry: dict[str, Any], kind: str, source_construct: str
) -> str:
    matches = [
        region["owner_id"]
        for region in _regions(entry)
        if region.get("kind") == kind
        and region.get("source_construct") == source_construct
    ]

    assert len(matches) == 1, (
        f"expected one {kind!r} region for {source_construct!r}, "
        f"found {len(matches)}"
    )

    return matches[0]


@pytest.fixture(scope="session")
def inventory() -> dict[str, Any]:
    inventory_path = _run_feature_flow_analysis()
    payload = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"inventory was not a mapping: {inventory_path}"
    return payload


def _walk_entries(entries: list[dict[str, Any]]):
    for entry in entries:
        yield entry
        yield from _walk_entries(entry.get("children", []) or [])


def _entry_by_name(inventory: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        entry
        for entry in _walk_entries(inventory.get("entries", []) or [])
        if entry.get("name") == name
    ]
    assert len(matches) == 1, f"expected one entry named {name}, found {len(matches)}"
    return matches[0]


def _control_flow(entry: dict[str, Any]) -> dict[str, Any]:
    ainfo = entry.get("analysis_info") or {}
    control_flow = ainfo.get("control_flow")
    assert isinstance(control_flow, dict), f"{entry.get('name')} has no control_flow"
    return control_flow


def _routes(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return _control_flow(entry).get("routes", []) or []


def _regions(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return _control_flow(entry).get("regions", []) or []


def _policies(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return _control_flow(entry).get("policies", []) or []


def _condition_owner(entry: dict[str, Any], condition: str) -> str:
    matches = [
        region["owner_id"]
        for region in _regions(entry)
        if region.get("kind") == "condition"
        and (region.get("metadata") or {}).get("condition") == condition
    ]
    assert len(matches) == 1, (
        f"expected one condition region for {condition!r} in {entry.get('name')}, "
        f"found {len(matches)}"
    )
    return matches[0]


def _routes_for_owner(entry: dict[str, Any], owner_id: str) -> list[dict[str, Any]]:
    return [route for route in _routes(entry) if route.get("owner_id") == owner_id]


def _assert_owner_has_route_suffix(
    entry: dict[str, Any],
    owner_id: str,
    suffix: str,
) -> dict[str, Any]:
    matches = [
        route
        for route in _routes_for_owner(entry, owner_id)
        if str(route.get("id", "")).endswith(suffix)
    ]
    assert len(matches) == 1, (
        f"expected one route ending {suffix!r} for owner {owner_id}, "
        f"found {len(matches)}"
    )
    return matches[0]


def _assert_owner_lacks_route_suffix(
    entry: dict[str, Any],
    owner_id: str,
    suffix: str,
) -> None:
    matches = [
        route
        for route in _routes_for_owner(entry, owner_id)
        if str(route.get("id", "")).endswith(suffix)
    ]
    assert not matches, (
        f"did not expect route ending {suffix!r} for owner {owner_id}, "
        f"found {[route.get('id') for route in matches]}"
    )


def test_full_control_flow_probe_does_not_emit_normal_completion_for_disruptive_if_bodies(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_full_control_flow_probe")

    for condition in ("item < 0", "item == 0", "total == 70"):
        owner_id = _condition_owner(entry, condition)
        _assert_owner_lacks_route_suffix(entry, owner_id, ":true_body_completion")

    route_kinds = {route.get("kind") for route in _routes(entry)}
    assert "post_execution_entry" in route_kinds
    assert "resume_prior_outcome" in route_kinds

    policy_mechanisms = {policy.get("mechanism_kind") for policy in _policies(entry)}
    assert "try_finally" in policy_mechanisms
    assert "context_manager_exit" in policy_mechanisms


def test_if_normal_completion_probe_keeps_both_body_completion_routes(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_if_normal_completion_probe")
    owner_id = _condition_owner(entry, "value > 0")

    true_completion = _assert_owner_has_route_suffix(
        entry,
        owner_id,
        ":true_body_completion",
    )
    false_completion = _assert_owner_has_route_suffix(
        entry,
        owner_id,
        ":false_body_completion",
    )

    assert true_completion.get("target_line") == entry["line_end"]
    assert false_completion.get("target_line") == entry["line_end"]


def test_if_partial_disruption_probe_keeps_outer_completion_but_drops_nested_completion(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_if_partial_disruption_probe")
    outer_owner_id = _condition_owner(entry, "value > 0")
    nested_owner_id = _condition_owner(entry, "value == 10")

    outer_completion = _assert_owner_has_route_suffix(
        entry,
        outer_owner_id,
        ":true_body_completion",
    )
    _assert_owner_lacks_route_suffix(
        entry,
        nested_owner_id,
        ":true_body_completion",
    )

    assert outer_completion.get("target_line") == entry["line_end"]
    _assert_owner_has_route_suffix(
        entry,
        nested_owner_id,
        ":condition_false_fallthrough",
    )


def test_loop_direct_disruptions_emit_continue_and_break_routes(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_loop_direct_disruptions")
    for_owner_id = _owner_for_region_kind(entry, "condition", "for")
    while_owner_id = _owner_for_region_kind(entry, "condition", "while")

    continue_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == for_owner_id
        and route.get("kind") == "loop_continue"
    ]
    break_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == while_owner_id and route.get("kind") == "loop_break"
    ]

    assert len(continue_routes) == 1
    assert continue_routes[0].get("exit_kind") == "continue"
    assert continue_routes[0].get("target_region_id") == f"{for_owner_id}:iterator"

    assert len(break_routes) == 1
    assert break_routes[0].get("exit_kind") == "break"
    assert break_routes[0].get("target_line") == entry["line_end"]


def test_nested_loop_disruptions_use_nested_if_region_as_source(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_full_control_flow_probe")
    item_lt_zero_owner = _condition_owner(entry, "item < 0")
    item_eq_zero_owner = _condition_owner(entry, "item == 0")
    total_eq_seventy_owner = _condition_owner(entry, "total == 70")

    continue_routes = [
        route
        for route in _routes(entry)
        if route.get("kind") == "loop_continue"
        and route.get("source_region_id") == f"{item_lt_zero_owner}:true_body"
    ]

    for_break_routes = [
        route
        for route in _routes(entry)
        if route.get("kind") == "loop_break"
        and route.get("source_region_id") == f"{item_eq_zero_owner}:true_body"
    ]

    while_break_routes = [
        route
        for route in _routes(entry)
        if route.get("kind") == "loop_break"
        and route.get("source_region_id") == f"{total_eq_seventy_owner}:true_body"
    ]

    assert len(continue_routes) == 1
    assert continue_routes[0].get("target_region_id").endswith(":iterator")
    assert continue_routes[0].get("exit_kind") == "continue"

    assert len(for_break_routes) == 1
    assert for_break_routes[0].get("exit_kind") == "break"

    assert len(while_break_routes) == 1
    assert while_break_routes[0].get("exit_kind") == "break"


def test_if_direct_return_raise_emit_disruptive_region_routes(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_if_direct_return_raise")
    value_gt_zero_owner = _condition_owner(entry, "value > 0")

    return_routes = [
        route
        for route in _route_by_kind(entry, "function_return")
        if route.get("source_region_id") == f"{value_gt_zero_owner}:true_body"
    ]
    raise_routes = [
        route
        for route in _route_by_kind(entry, "raise")
        if route.get("source_region_id") == f"{value_gt_zero_owner}:false_body"
    ]

    assert len(return_routes) == 1
    assert return_routes[0].get("exit_kind") == "return"

    assert len(raise_routes) == 1
    assert raise_routes[0].get("exit_kind") == "raise"


def test_match_direct_return_raise_emit_disruptive_case_routes(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_match_direct_return_raise")
    return_routes = _route_by_kind(entry, "function_return")
    raise_routes = _route_by_kind(entry, "raise")

    assert len(return_routes) == 1
    assert return_routes[0].get("source_region_id", "").startswith("match:")
    assert return_routes[0].get("source_region_id", "").endswith(":case_body:1")
    assert return_routes[0].get("exit_kind") == "return"

    assert len(raise_routes) == 1
    assert raise_routes[0].get("source_region_id", "").startswith("match:")
    assert raise_routes[0].get("source_region_id", "").endswith(":case_body:2")
    assert raise_routes[0].get("exit_kind") == "raise"


def test_loop_direct_return_raise_emit_disruptive_body_routes(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_loop_direct_return_raise")
    return_routes = _route_by_kind(entry, "function_return")
    raise_routes = _route_by_kind(entry, "raise")
    for_owner_id = _owner_for_region_kind(entry, "condition", "for")
    while_owner_id = _owner_for_region_kind(entry, "condition", "while")

    _assert_owner_lacks_route_suffix(
        entry,
        for_owner_id,
        ":body_next_iteration",
    )
    _assert_owner_lacks_route_suffix(
        entry,
        while_owner_id,
        ":body_next_iteration",
    )

    assert len(return_routes) == 1
    assert return_routes[0].get("source_region_id", "").startswith("for:")
    assert return_routes[0].get("source_region_id", "").endswith(":loop_body")
    assert return_routes[0].get("exit_kind") == "return"

    assert len(raise_routes) == 1
    assert raise_routes[0].get("source_region_id", "").startswith("while:")
    assert raise_routes[0].get("source_region_id", "").endswith(":loop_body")
    assert raise_routes[0].get("exit_kind") == "raise"


def test_nested_if_return_raise_uses_nested_if_region_as_source(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_nested_if_return_raise")

    outer_owner_id = _condition_owner(entry, "value > 0")
    nested_return_owner_id = _condition_owner(entry, "value == 10")
    nested_raise_owner_id = _condition_owner(entry, "value < -10")

    return_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == nested_return_owner_id
        and route.get("kind") == "function_return"
        and route.get("source_region_id") == f"{nested_return_owner_id}:true_body"
    ]

    raise_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == nested_raise_owner_id
        and route.get("kind") == "raise"
        and route.get("source_region_id") == f"{nested_raise_owner_id}:true_body"
    ]

    assert len(return_routes) == 1
    assert return_routes[0].get("exit_kind") == "return"

    assert len(raise_routes) == 1
    assert raise_routes[0].get("exit_kind") == "raise"

    outer_completion = _assert_owner_has_route_suffix(
        entry,
        outer_owner_id,
        ":true_body_completion",
    )
    _assert_owner_lacks_route_suffix(
        entry,
        nested_return_owner_id,
        ":true_body_completion",
    )
    _assert_owner_lacks_route_suffix(
        entry,
        nested_raise_owner_id,
        ":true_body_completion",
    )

    assert outer_completion.get("target_line") is not None


def test_nested_match_loop_disruptions_use_match_case_region_as_source(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_mixed_control")

    loop_owner_id = _owner_for_region_kind(entry, "condition", "for")

    continue_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == loop_owner_id
        and route.get("kind") == "loop_continue"
        and route.get("source_region_id", "").startswith("match:")
        and route.get("source_region_id", "").endswith(":case_body:1")
    ]

    break_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == loop_owner_id
        and route.get("kind") == "loop_break"
        and route.get("source_region_id", "").startswith("match:")
        and route.get("source_region_id", "").endswith(":case_body:2")
    ]

    assert len(continue_routes) == 1
    assert continue_routes[0].get("exit_kind") == "continue"
    assert continue_routes[0].get("target_region_id") == f"{loop_owner_id}:iterator"

    assert len(break_routes) == 1
    assert break_routes[0].get("exit_kind") == "break"


def test_nested_try_loop_disruptions_use_nested_and_handler_regions_as_source(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_try_loop_disruptions")

    loop_owner_id = _owner_for_region_kind(entry, "condition", "for")

    loop_continue_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == loop_owner_id
        and route.get("kind") == "loop_continue"
    ]
    loop_break_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == loop_owner_id and route.get("kind") == "loop_break"
    ]

    nested_if_continue_routes = [
        route
        for route in loop_continue_routes
        if str(route.get("source_region_id", "")).startswith("if:")
        and str(route.get("source_region_id", "")).endswith(":true_body")
    ]

    handler_continue_routes = [
        route
        for route in loop_continue_routes
        if ":exception_handler:" in str(route.get("source_region_id", ""))
    ]

    nested_if_break_routes = [
        route
        for route in loop_break_routes
        if str(route.get("source_region_id", "")).startswith("if:")
        and str(route.get("source_region_id", "")).endswith(":true_body")
    ]

    assert len(nested_if_continue_routes) == 1
    assert nested_if_continue_routes[0].get("exit_kind") == "continue"
    assert nested_if_continue_routes[0].get("target_region_id") == (
        f"{loop_owner_id}:iterator"
    )

    assert len(nested_if_break_routes) == 1
    assert nested_if_break_routes[0].get("exit_kind") == "break"

    assert len(handler_continue_routes) == 1
    assert handler_continue_routes[0].get("exit_kind") == "continue"
    assert handler_continue_routes[0].get("target_region_id") == (
        f"{loop_owner_id}:iterator"
    )


def test_nested_with_loop_disruptions_use_nested_if_regions_as_source(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_with_loop_disruptions")

    loop_owner_id = _owner_for_region_kind(entry, "condition", "for")

    loop_continue_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == loop_owner_id
        and route.get("kind") == "loop_continue"
    ]
    loop_break_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == loop_owner_id and route.get("kind") == "loop_break"
    ]

    nested_if_continue_routes = [
        route
        for route in loop_continue_routes
        if str(route.get("source_region_id", "")).startswith("if:")
        and str(route.get("source_region_id", "")).endswith(":true_body")
    ]

    nested_if_break_routes = [
        route
        for route in loop_break_routes
        if str(route.get("source_region_id", "")).startswith("if:")
        and str(route.get("source_region_id", "")).endswith(":true_body")
    ]

    with_policies = [
        policy
        for policy in _policies(entry)
        if policy.get("mechanism_kind") == "context_manager_exit"
    ]

    assert len(nested_if_continue_routes) == 1
    assert nested_if_continue_routes[0].get("exit_kind") == "continue"
    assert nested_if_continue_routes[0].get("target_region_id") == (
        f"{loop_owner_id}:iterator"
    )

    assert len(nested_if_break_routes) == 1
    assert nested_if_break_routes[0].get("exit_kind") == "break"

    assert len(with_policies) == 1
    assert "continue" in with_policies[0].get("applies_to", [])
    assert "break" in with_policies[0].get("applies_to", [])


def test_nested_loop_transfers_bind_to_inner_loop_not_outer_loop(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_nested_loop_transfer_binding")

    loop_owner_ids = [
        region["owner_id"]
        for region in _regions(entry)
        if region.get("kind") == "condition" and region.get("source_construct") == "for"
    ]

    assert len(loop_owner_ids) == 2

    outer_loop_owner_id = loop_owner_ids[0]
    inner_loop_owner_id = loop_owner_ids[1]

    outer_transfer_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == outer_loop_owner_id
        and route.get("kind") in {"loop_continue", "loop_break"}
    ]

    inner_continue_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == inner_loop_owner_id
        and route.get("kind") == "loop_continue"
    ]

    inner_break_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == inner_loop_owner_id
        and route.get("kind") == "loop_break"
    ]

    assert outer_transfer_routes == []

    assert len(inner_continue_routes) == 1
    assert inner_continue_routes[0].get("exit_kind") == "continue"
    assert inner_continue_routes[0].get("target_region_id") == (
        f"{inner_loop_owner_id}:iterator"
    )

    assert len(inner_break_routes) == 1
    assert inner_break_routes[0].get("exit_kind") == "break"


def test_try_direct_return_raise_emits_region_disruptive_routes(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_try_direct_return_raise")

    try_owner_ids = [
        region["owner_id"]
        for region in _regions(entry)
        if region.get("kind") == "protected_body"
        and region.get("source_construct") == "try"
    ]

    assert len(try_owner_ids) == 1
    try_owner_id = try_owner_ids[0]

    protected_return_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and route.get("kind") == "function_return"
        and route.get("source_region_id") == f"{try_owner_id}:protected_body"
    ]

    handler_raise_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and route.get("kind") == "raise"
        and route.get("source_region_id", "").startswith(
            f"{try_owner_id}:exception_handler:"
        )
    ]

    post_execution_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and route.get("kind") == "post_execution_entry"
    ]

    try_finally_policies = [
        policy
        for policy in _policies(entry)
        if policy.get("owner_id") == try_owner_id
        and policy.get("mechanism_kind") == "try_finally"
    ]

    assert len(protected_return_routes) == 1
    assert protected_return_routes[0].get("exit_kind") == "return"

    assert len(handler_raise_routes) == 1
    assert handler_raise_routes[0].get("exit_kind") == "raise"

    assert len(post_execution_routes) == 1
    assert len(try_finally_policies) == 1
    assert "return" in try_finally_policies[0].get("applies_to", [])
    assert "raise" in try_finally_policies[0].get("applies_to", [])


def test_with_direct_return_raise_emits_body_disruptive_routes(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_with_direct_return_raise")

    with_owner_ids = [
        region["owner_id"]
        for region in _regions(entry)
        if region.get("kind") == "body" and region.get("source_construct") == "with"
    ]

    assert len(with_owner_ids) == 2

    return_routes = [
        route
        for route in _routes(entry)
        if route.get("kind") == "function_return"
        and route.get("source_region_id", "").endswith(":body")
        and route.get("source_region_id", "").startswith("with:")
    ]

    raise_routes = [
        route
        for route in _routes(entry)
        if route.get("kind") == "raise"
        and route.get("source_region_id", "").endswith(":body")
        and route.get("source_region_id", "").startswith("with:")
    ]

    post_execution_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") in with_owner_ids
        and route.get("kind") == "post_execution_entry"
    ]

    resume_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") in with_owner_ids
        and route.get("kind") == "resume_prior_outcome"
    ]

    with_policies = [
        policy
        for policy in _policies(entry)
        if policy.get("owner_id") in with_owner_ids
        and policy.get("mechanism_kind") == "context_manager_exit"
    ]

    assert len(return_routes) == 1
    assert return_routes[0].get("exit_kind") == "return"

    assert len(raise_routes) == 1
    assert raise_routes[0].get("exit_kind") == "raise"

    assert len(post_execution_routes) == 2
    assert all(
        route.get("preserves_prior_outcome") is True for route in post_execution_routes
    )

    assert len(resume_routes) == 2
    assert all(route.get("preserves_prior_outcome") is True for route in resume_routes)

    assert len(with_policies) == 2
    assert all("return" in policy.get("applies_to", []) for policy in with_policies)
    assert all("raise" in policy.get("applies_to", []) for policy in with_policies)


def test_try_direct_return_suppresses_protected_normal_to_else(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_try_direct_return_suppresses_else")

    try_owner_ids = [
        region["owner_id"]
        for region in _regions(entry)
        if region.get("kind") == "protected_body"
        and region.get("source_construct") == "try"
    ]

    assert len(try_owner_ids) == 1
    try_owner_id = try_owner_ids[0]

    protected_return_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and route.get("kind") == "function_return"
        and route.get("source_region_id") == f"{try_owner_id}:protected_body"
    ]

    protected_normal_to_else_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and str(route.get("id", "")).endswith(":protected_normal_to_else")
    ]

    else_return_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and route.get("kind") == "function_return"
        and route.get("source_region_id") == f"{try_owner_id}:success_continuation"
    ]

    assert len(protected_return_routes) == 1
    assert protected_return_routes[0].get("exit_kind") == "return"

    assert protected_normal_to_else_routes == []

    assert len(else_return_routes) == 1
    assert else_return_routes[0].get("exit_kind") == "return"


def test_try_finally_emits_resume_prior_outcome_route(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_try_direct_return_raise")

    try_owner_ids = [
        region["owner_id"]
        for region in _regions(entry)
        if region.get("kind") == "post_execution"
        and region.get("source_construct") == "finally"
    ]

    assert len(try_owner_ids) == 1
    try_owner_id = try_owner_ids[0]

    resume_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and route.get("kind") == "resume_prior_outcome"
        and route.get("source_region_id") == f"{try_owner_id}:post_execution"
    ]

    assert len(resume_routes) == 1
    assert resume_routes[0].get("preserves_prior_outcome") is True


def test_try_handler_and_else_emit_normal_completion_routes(
    inventory: dict[str, Any],
) -> None:
    entry = _entry_by_name(inventory, "fixture_try_handler_else_normal_completion")

    try_owner_ids = [
        region["owner_id"]
        for region in _regions(entry)
        if region.get("kind") == "protected_body"
        and region.get("source_construct") == "try"
    ]

    assert len(try_owner_ids) == 1
    try_owner_id = try_owner_ids[0]

    handler_completion_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and route.get("kind") == "normal_completion"
        and str(route.get("source_region_id", "")).startswith(
            f"{try_owner_id}:exception_handler:"
        )
    ]

    else_completion_routes = [
        route
        for route in _routes(entry)
        if route.get("owner_id") == try_owner_id
        and route.get("kind") == "normal_completion"
        and route.get("source_region_id") == f"{try_owner_id}:success_continuation"
    ]

    assert len(handler_completion_routes) == 1
    assert handler_completion_routes[0].get("exit_kind") == "normal"
    assert handler_completion_routes[0].get("target_line") == entry["line_end"]

    assert len(else_completion_routes) == 1
    assert else_completion_routes[0].get("exit_kind") == "normal"
    assert else_completion_routes[0].get("target_line") == entry["line_end"]


def test_control_flow_route_ids_are_unique_per_entry(
    inventory: dict[str, Any],
) -> None:
    for entry in _walk_entries(inventory.get("entries", []) or []):
        control_flow = (entry.get("analysis_info") or {}).get("control_flow")
        if not control_flow:
            continue

        route_ids = [
            route.get("id")
            for route in control_flow.get("routes", []) or []
            if route.get("id")
        ]

        assert len(route_ids) == len(set(route_ids)), (
            f"duplicate route IDs in {entry.get('name')}: "
            f"{sorted(route_id for route_id in set(route_ids) if route_ids.count(route_id) > 1)}"
        )


def test_loop_transfer_routes_are_not_duplicated_per_source(
    inventory: dict[str, Any],
) -> None:
    for entry in _walk_entries(inventory.get("entries", []) or []):
        control_flow = (entry.get("analysis_info") or {}).get("control_flow")
        if not control_flow:
            continue

        seen: set[tuple[str, str, str, int | None]] = set()

        for route in control_flow.get("routes", []) or []:
            if route.get("kind") not in {"loop_continue", "loop_break"}:
                continue

            key = (
                route.get("owner_id"),
                route.get("kind"),
                route.get("source_region_id"),
                route.get("target_line"),
            )

            assert (
                key not in seen
            ), f"duplicate loop transfer route in {entry.get('name')}: {route}"
            seen.add(key)
