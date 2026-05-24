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

    assert len(return_routes) == 1
    assert return_routes[0].get("source_region_id", "").startswith("for:")
    assert return_routes[0].get("source_region_id", "").endswith(":loop_body")
    assert return_routes[0].get("exit_kind") == "return"

    assert len(raise_routes) == 1
    assert raise_routes[0].get("source_region_id", "").startswith("while:")
    assert raise_routes[0].get("source_region_id", "").endswith(":loop_body")
    assert raise_routes[0].get("exit_kind") == "raise"
