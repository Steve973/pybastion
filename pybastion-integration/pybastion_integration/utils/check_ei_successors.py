#!/usr/bin/env python3
"""
Preflight check for EI successor completeness.

This checks stage 3 inventory files before the integration graph builder runs.

The goal is to verify that non-terminal execution items already carry explicit
successor information in the inventory, so the graph builder does not need to
infer continuation from previously-created graph edges.

Rules:

- Terminal EIs do not need a next EI.
- Non-terminal EIs must have at least one explicit target EI.
- Target EIs must exist in the same callable.
- Statement, conditional, and disruptive targets are all checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def discover_inventory_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.inventory.yaml"))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Inventory file is not a mapping: {path}")

    return payload


def iter_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(entry: dict[str, Any]) -> None:
        result.append(entry)

        children = entry.get("entries", []) or []
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    visit(child)

    for entry in entries:
        if isinstance(entry, dict):
            visit(entry)

    return result


def is_callable_entry(entry: dict[str, Any]) -> bool:
    analysis_info = entry.get("analysis_info")
    if not isinstance(analysis_info, dict):
        return False

    branches = analysis_info.get("branches")
    return isinstance(branches, list)


def target_eis_for_branch(branch: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []

    statement_outcome = branch.get("statement_outcome") or {}
    if isinstance(statement_outcome, dict):
        target_ei = statement_outcome.get("target_ei")
        if target_ei:
            targets.append(
                {
                    "source": "statement_outcome",
                    "target_ei": target_ei,
                    "is_terminal": bool(statement_outcome.get("is_terminal", False)),
                }
            )

    for conditional in branch.get("conditional_targets", []) or []:
        if not isinstance(conditional, dict):
            continue

        target_ei = conditional.get("target_ei")
        if target_ei:
            targets.append(
                {
                    "source": "conditional_targets",
                    "target_ei": target_ei,
                    "is_terminal": bool(conditional.get("is_terminal", False)),
                    "condition_result": conditional.get("condition_result"),
                    "target_condition": conditional.get("target_condition"),
                }
            )

    for disruptive in branch.get("disruptive_outcomes", []) or []:
        if not isinstance(disruptive, dict):
            continue

        target_ei = disruptive.get("target_ei")
        if target_ei:
            targets.append(
                {
                    "source": "disruptive_outcomes",
                    "target_ei": target_ei,
                    "is_terminal": bool(disruptive.get("is_terminal", False)),
                    "outcome": disruptive.get("outcome"),
                }
            )

    return targets


def branch_is_terminal(branch: dict[str, Any]) -> bool:
    outcome = branch.get("statement_outcome") or {}
    if isinstance(outcome, dict) and outcome.get("is_terminal"):
        return True

    terminates_via = outcome.get("terminates_via") if isinstance(outcome, dict) else None
    if terminates_via in {"return", "implicit-return", "yield", "raise", "exception"}:
        return True

    stmt_type = branch.get("stmt_type")
    description = str(branch.get("description") or "")

    if stmt_type == "Raise":
        return True

    if "raises exception" in description or description.startswith("raises "):
        return True

    return False


def check_callable(
    *,
    inventory_path: Path,
    unit_name: str,
    unit_fqn: str,
    entry: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    callable_id = str(entry.get("id", "unknown"))
    callable_name = str(entry.get("name", "unknown"))
    callable_fqn = str(entry.get("fully_qualified_name", callable_name))

    analysis_info = entry.get("analysis_info") or {}
    branches = analysis_info.get("branches") or []

    if not isinstance(branches, list):
        errors.append(
            {
                "kind": "invalid_branches",
                "inventory": str(inventory_path),
                "unit": unit_name,
                "unit_fqn": unit_fqn,
                "callable_id": callable_id,
                "callable_fqn": callable_fqn,
                "message": "analysis_info.branches is not a list",
            }
        )
        return errors

    branch_ids = {
        str(branch.get("id"))
        for branch in branches
        if isinstance(branch, dict) and branch.get("id")
    }

    for branch in branches:
        if not isinstance(branch, dict):
            continue

        ei_id = str(branch.get("id", "unknown"))
        terminal = branch_is_terminal(branch)
        targets = target_eis_for_branch(branch)

        if not terminal and not targets:
            errors.append(
                {
                    "kind": "missing_next_ei",
                    "inventory": str(inventory_path),
                    "unit": unit_name,
                    "unit_fqn": unit_fqn,
                    "callable_id": callable_id,
                    "callable_fqn": callable_fqn,
                    "ei_id": ei_id,
                    "line": branch.get("line"),
                    "stmt_type": branch.get("stmt_type"),
                    "description": branch.get("description"),
                    "message": (
                        "Non-terminal EI has no explicit target_ei in "
                        "statement_outcome, conditional_targets, or disruptive_outcomes"
                    ),
                }
            )
            continue

        for target in targets:
            target_ei = str(target["target_ei"])

            if target_ei not in branch_ids:
                errors.append(
                    {
                        "kind": "target_ei_not_in_callable",
                        "inventory": str(inventory_path),
                        "unit": unit_name,
                        "unit_fqn": unit_fqn,
                        "callable_id": callable_id,
                        "callable_fqn": callable_fqn,
                        "ei_id": ei_id,
                        "target_ei": target_ei,
                        "target_source": target["source"],
                        "line": branch.get("line"),
                        "stmt_type": branch.get("stmt_type"),
                        "description": branch.get("description"),
                        "message": (
                            "EI target_ei does not exist in the same callable's "
                            "branch set"
                        ),
                    }
                )

    return errors


def check_inventory(path: Path) -> list[dict[str, Any]]:
    payload = load_yaml(path)

    unit_name = str(payload.get("unit", path.stem))
    unit_fqn = str(payload.get("fully_qualified_name", unit_name))

    entries = payload.get("entries", []) or []
    if not isinstance(entries, list):
        return [
            {
                "kind": "invalid_entries",
                "inventory": str(path),
                "unit": unit_name,
                "unit_fqn": unit_fqn,
                "message": "entries is not a list",
            }
        ]

    errors: list[dict[str, Any]] = []

    for entry in iter_entries(entries):
        if not is_callable_entry(entry):
            continue

        errors.extend(
            check_callable(
                inventory_path=path,
                unit_name=unit_name,
                unit_fqn=unit_fqn,
                entry=entry,
            )
        )

    return errors


def run_preflight(
    *,
    inventories_root: Path,
    report_path: Path | None = None,
    fail_on_error: bool = False,
    verbose: bool = False,
) -> bool:
    if not inventories_root.exists():
        print(
            f"ERROR: inventories root does not exist: {inventories_root}",
            file=sys.stderr,
        )
        return False

    inventory_paths = discover_inventory_files(inventories_root)
    if not inventory_paths:
        print(
            f"ERROR: no inventory files found under {inventories_root}",
            file=sys.stderr,
        )
        return False

    all_errors: list[dict[str, Any]] = []

    for inventory_path in inventory_paths:
        if verbose:
            print(f"Checking {inventory_path}")

        try:
            all_errors.extend(check_inventory(inventory_path))
        except Exception as exc:
            all_errors.append(
                {
                    "kind": "inventory_read_error",
                    "inventory": str(inventory_path),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "inventories_checked": len(inventory_paths),
        "error_count": len(all_errors),
        "errors": all_errors,
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                report,
                f,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            )

    if all_errors:
        print(f"EI successor preflight failed: {len(all_errors)} problem(s)")
        if report_path is None:
            yaml.dump(
                report,
                sys.stdout,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            )
        else:
            print(f"Wrote report to {report_path}")

        return not fail_on_error

    print(f"EI successor preflight passed: {len(inventory_paths)} inventory file(s)")
    if report_path is not None:
        print(f"Wrote report to {report_path}")

    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that non-terminal EIs have explicit inventory successors."
    )
    parser.add_argument(
        "--inventories-root",
        type=Path,
        required=True,
        help="Root containing *.inventory.yaml files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional YAML report path",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit nonzero when successor problems are found",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    ok = run_preflight(
        inventories_root=args.inventories_root,
        report_path=args.report,
        fail_on_error=args.fail_on_error,
        verbose=args.verbose,
    )

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())