#!/usr/bin/env python3
"""
Execution Item Enumerator - stage 2 using structured unit index.

This keeps the current EI decomposition and resolution behavior, but changes how
callables/scopes are discovered. Instead of rediscovering callables from a flat
inventory, it consumes the structured stage 1 unit index and analyzes entries in
that deterministic order.

Key changes:
- structured unit index is the authoritative callable/scope inventory
- AST is parsed once per file
- a lightweight locator maps indexed entries to AST nodes
- EI generation/resolution logic remains intentionally close to the current stage 2
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pybastion_common.models import Branch, StatementOutcome, ConditionalTarget, DisruptiveOutcome, OwnerInfo, \
    ProjectIndex, UnitIndex, UnitIndexEntry
from pybastion_unit.helpers.constraint_metadata_helper import (
    enrich_outcome_with_constraint,
    populate_constraint_relationships,
)
from pybastion_unit.helpers.decorator_processing import extract_statement_decorators
from pybastion_unit.semantic_decomposition import decompose_statement
from pybastion_unit.semantic_decomposition.decomp_types import DecompositionContext, ControlOwner, OwnerKind, DecomposerResult
from pybastion_unit.shared.callable_id_generation import (
    FUNC_ID_EXPR,
    generate_ei_id,
    generate_function_entry_ei_id,
)

ENUM_BASES: set[str] = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
ANALYZABLE_ENTRY_KINDS: set[str] = {
    "unit_function",
    "method",
    "nested_function",
    "class",
    "nested_class",
    "module_assignment",
}
IMPLICIT_RETURN_EI_NUM = 9999


# ============================================================================
# Unit index model
# ============================================================================


def load_project_index(filepath: Path) -> ProjectIndex:
    payload = json.loads(filepath.read_text(encoding="utf-8"))
    units: list[UnitIndex] = []
    for unit_payload in payload.get("units", []):
        entries = [UnitIndexEntry(**entry) for entry in unit_payload.get("entries", [])]
        units.append(
            UnitIndex(
                unit_id=unit_payload["unit_id"],
                fully_qualified_name=unit_payload["fully_qualified_name"],
                filepath=unit_payload["filepath"],
                language=unit_payload["language"],
                source_hash=unit_payload["source_hash"],
                entries=entries,
            )
        )
    return ProjectIndex(source_root=payload["source_root"], units=units)


# ============================================================================
# Statement context model
# ============================================================================


@dataclass(frozen=True)
class StatementContext:
    stmt: ast.stmt
    next_stmt_lines: list[int] | None
    owners: tuple[ControlOwner, ...] = field(default_factory=tuple)


def _prepend_next_line(
        local_next_line: int | None,
        inherited_next_lines: list[int] | None,
) -> list[int] | None:
    if local_next_line is None:
        return inherited_next_lines
    if inherited_next_lines is None:
        return [local_next_line]
    return [local_next_line, *inherited_next_lines]


def get_statement_contexts(node: ast.AST) -> list[StatementContext]:
    result: list[StatementContext] = []

    def visit_block(
            statements: list[ast.stmt],
            inherited_next_lines: list[int] | None = None,
            owners: tuple[ControlOwner, ...] = (),
    ) -> None:
        for i, stmt in enumerate(statements):
            local_next_line = statements[i + 1].lineno if i + 1 < len(statements) else None
            next_lines = _prepend_next_line(local_next_line, inherited_next_lines)

            result.append(
                StatementContext(
                    stmt=stmt,
                    next_stmt_lines=next_lines,
                    owners=owners,
                )
            )

            if isinstance(stmt, ast.If):
                visit_block(
                    stmt.body,
                    next_lines,
                    (*owners, ControlOwner(
                        kind=OwnerKind.IF,
                        node=stmt,
                        region="body",
                        next_stmt_lines=next_lines
                    )),
                )
                visit_block(
                    stmt.orelse,
                    next_lines,
                    (*owners, ControlOwner(
                        kind=OwnerKind.IF,
                        node=stmt,
                        region="orelse",
                        next_stmt_lines=next_lines
                    )),
                )
                continue

            if isinstance(stmt, ast.Match):
                for case_index, case in enumerate(stmt.cases, start=1):
                    visit_block(
                        case.body,
                        next_lines,
                        (*owners, ControlOwner(
                            kind=OwnerKind.MATCH,
                            node=stmt,
                            region=f"case[{case_index}]",
                            next_stmt_lines=next_lines,
                        )),
                    )
                continue

            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                visit_block(
                    stmt.body,
                    next_lines,
                    (*owners, ControlOwner(
                        kind=OwnerKind.LOOP,
                        node=stmt,
                        region="body",
                        next_stmt_lines=next_lines
                    ))
                )
                visit_block(
                    stmt.orelse,
                    next_lines,
                    (*owners, ControlOwner(
                        kind=OwnerKind.LOOP,
                        node=stmt,
                        region="orelse",
                        next_stmt_lines=next_lines
                    ))
                )
                continue

            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                visit_block(
                    stmt.body,
                    next_lines,
                    (*owners, ControlOwner(
                        kind=OwnerKind.WITH,
                        node=stmt,
                        region="body",
                        next_stmt_lines=next_lines
                    )),
                )
                continue

            if isinstance(stmt, ast.Try):
                visit_block(
                    stmt.body,
                    next_lines,
                    (*owners, ControlOwner(
                        kind=OwnerKind.TRY,
                        node=stmt,
                        region="body",
                        next_stmt_lines=next_lines,
                    )),
                )

                visit_block(
                    stmt.orelse,
                    next_lines,
                    (*owners, ControlOwner(
                        kind=OwnerKind.TRY,
                        node=stmt,
                        region="else",
                        next_stmt_lines=next_lines,
                    )),
                )

                for handler in stmt.handlers:
                    visit_block(
                        handler.body,
                        next_lines,
                        (*owners, ControlOwner(
                            kind=OwnerKind.TRY,
                            node=stmt,
                            region="except",
                            next_stmt_lines=next_lines,
                        )),
                    )

                visit_block(
                    stmt.finalbody,
                    next_lines,
                    (*owners, ControlOwner(
                        kind=OwnerKind.TRY,
                        node=stmt,
                        region="finally",
                        next_stmt_lines=next_lines,
                    )),
                )

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        visit_block(node.body)

    return result


def _build_decomposition_context(context: StatementContext) -> DecompositionContext:
    return DecompositionContext(
        next_stmt_lines=context.next_stmt_lines,
        owners=context.owners,
    )


# ============================================================================
# Result structures
# ============================================================================


class FunctionResult:
    def __init__(self, name: str, line_start: int, line_end: int, branches: list[Branch]) -> None:
        self.name = name
        self.line_start = line_start
        self.line_end = line_end
        self.branches = branches
        self.total_eis = len(branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "total_eis": self.total_eis,
            "branches": [branch.to_dict() for branch in self.branches],
        }


def create_function_entry_branch(
        callable_id: str,
        line_num: int,
        target_line: int | None = None,
) -> Branch:
    return Branch(
        id=generate_function_entry_ei_id(callable_id),
        line=line_num,
        condition=f"enters function {callable_id}",
        description="function start",
        stmt_type="FunctionInvocation",
        statement_outcome=StatementOutcome(
            outcome="function start",
            target_line=target_line,
            synthetic=True,
        ),
    )


def get_first_ei_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    if not getattr(node, "body", None):
        return node.lineno

    first_stmt = node.body[0]
    if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
    ):
        if len(node.body) > 1:
            return node.body[1].lineno
        return node.lineno

    return first_stmt.lineno


# ============================================================================
# Resolution helpers
# ============================================================================


def _implicit_return_ei_id(callable_id: str) -> str:
    return generate_ei_id(callable_id, IMPLICIT_RETURN_EI_NUM)


def _redirect_implicit_return_outcome(
        outcome: StatementOutcome | ConditionalTarget | DisruptiveOutcome,
        sink_id: str,
) -> bool:
    if not outcome.is_terminal:
        return False
    if outcome.terminates_via != "implicit-return":
        return False

    outcome.is_terminal = False
    outcome.terminates_via = None
    outcome.target_line = None
    outcome.target_ei = sink_id
    return True


def _redirect_explicit_implicit_returns(branches: list[Branch], callable_id: str) -> bool:
    sink_id = _implicit_return_ei_id(callable_id)
    referenced = False

    for branch in branches:
        for outcome in _iter_resolvable_outcomes(branch):
            if _redirect_implicit_return_outcome(outcome, sink_id):
                referenced = True

    return referenced


def _append_implicit_return_sink_if_referenced(
        branches: list[Branch],
        callable_id: str,
        sink_line: int,
) -> None:
    sink_id = _implicit_return_ei_id(callable_id)

    if any(branch.id == sink_id for branch in branches):
        return

    referenced = False
    for branch in branches:
        for outcome in _iter_resolvable_outcomes(branch):
            if outcome.target_ei == sink_id:
                referenced = True
                break
        if referenced:
            break

    if not referenced:
        return

    branches.append(
        Branch(
            id=sink_id,
            line=sink_line,
            condition="implicit return",
            description="implicit return",
            stmt_type="ImplicitReturn",
            statement_outcome=StatementOutcome(
                outcome="implicit return",
                is_terminal=True,
                terminates_via="implicit-return",
                synthetic=True,
            ),
        )
    )


def _iter_resolvable_outcomes(branch: Branch) -> list[StatementOutcome | ConditionalTarget | DisruptiveOutcome]:
    outcomes: list[StatementOutcome | ConditionalTarget | DisruptiveOutcome] = []

    if branch.statement_outcome is not None:
        outcomes.append(branch.statement_outcome)

    if branch.conditional_targets:
        outcomes.extend(branch.conditional_targets)

    if branch.disruptive_outcomes:
        outcomes.extend(branch.disruptive_outcomes)

    return outcomes


def _iter_skippable_outcomes(branch: Branch) -> list[StatementOutcome | ConditionalTarget | DisruptiveOutcome]:
    return _iter_resolvable_outcomes(branch)


def _resolve_branch_target_line(branches: list[Branch], branch: Branch) -> None:
    for outcome in _iter_resolvable_outcomes(branch):
        if outcome.target_line is None:
            continue

        for candidate in branches:
            if candidate.line == outcome.target_line:
                outcome.target_ei = candidate.id
                break

        outcome.target_line = None


def _resolve_same_line_if_target(branches: list[Branch], branch: Branch, condition: bool) -> str | None:
    for candidate in branches:
        if candidate.line != branch.line:
            continue
        if candidate.id == branch.id:
            continue
        if candidate.stmt_type != "If":
            continue
        if candidate.constraint is None:
            continue
        if candidate.constraint.constraint_type != "condition":
            continue
        if candidate.constraint.polarity != condition:
            continue
        return candidate.id

    return None


def _resolve_conditional_targets(branches: list[Branch]) -> None:
    for branch in branches:
        if not branch.conditional_targets:
            continue

        for target in branch.conditional_targets:
            if target.is_terminal or target.target_line is None:
                continue

            if (
                    branch.stmt_type == "If"
                    and branch.description.startswith("evaluates ")
                    and target.target_line == branch.line
            ):
                sibling_target = _resolve_same_line_if_target(
                    branches,
                    branch,
                    target.condition_result,
                )
                if sibling_target:
                    target.target_ei = sibling_target
                    continue

            for candidate in branches:
                if candidate.line == target.target_line:
                    target.target_ei = candidate.id
                    break


def _resolve_skip_eis(branches: list[Branch]) -> None:
    line_to_eis: dict[int, list[str]] = {}
    for branch in branches:
        line_to_eis.setdefault(branch.line, []).append(branch.id)

    for branch in branches:
        for outcome in _iter_skippable_outcomes(branch):
            if not outcome.skips_lines:
                continue

            resolved_ids: list[str] = []
            for line_num in outcome.skips_lines:
                resolved_ids.extend(line_to_eis.get(line_num, []))

            seen: set[str] = set()
            outcome.skips_eis = [ei for ei in resolved_ids if not (ei in seen or seen.add(ei))]


def _is_forbidden_successor(current: Branch, candidate: Branch) -> bool:
    owner = candidate.owner_info
    current_owner = current.owner_info

    predicates = [
        owner is not None
        and owner.stmt_type == "Try"
        and owner.region == "except",

        current_owner is not None
        and owner is not None
        and current_owner.stmt_type == "If"
        and owner.stmt_type == "If"
        and current_owner.line == owner.line
        and current_owner.region == "body"
        and owner.region == "orelse",
    ]

    return any(predicates)


def _is_excluded_successor(current: Branch, candidate: Branch) -> bool:
    if current.constraint is None:
        return False
    return candidate.id in (current.constraint.excludes or [])


def _is_skipped_successor(
    candidate: Branch,
    outcome: StatementOutcome | ConditionalTarget | DisruptiveOutcome | None = None,
) -> bool:
    if outcome is None:
        return False
    return candidate.id in (outcome.skips_eis or [])


def _same_statement_successor(
    branches: list[Branch],
    index: int,
    branch: Branch,
) -> Branch | None:
    outcome = branch.statement_outcome
    if outcome is None:
        return None

    for candidate in branches[index + 1:]:
        if candidate.line != branch.line:
            break

        if _is_skipped_successor(candidate, outcome):
            continue
        if _is_excluded_successor(branch, candidate):
            continue
        if _is_forbidden_successor(branch, candidate):
            continue

        # Prefer same-line normal statement outcomes first.
        if candidate.statement_outcome is not None:
            return candidate

        # If there are no more same-line normal outcomes, allow a same-line
        # disruptive/terminal branch as the final step.
        if candidate.disruptive_outcomes:
            return candidate

    return None


def _assign_fallthrough_next_eis(branches: list[Branch], callable_id: str) -> None:
    """
    Final safe fallthrough for normal statement outcomes.

    Priority:
    1. explicit target_ei already assigned on statement_outcome
    2. same-statement sequential continuation for Raise/Return
    3. generic forward fallthrough respecting skips/excludes
    4. implicit return sink when no next EI exists
    """
    implicit_return_sink = _implicit_return_ei_id(callable_id)

    for index, branch in enumerate(branches):
        outcome = branch.statement_outcome
        if outcome is None:
            continue

        if outcome.target_ei or outcome.is_terminal:
            continue

        same_stmt = _same_statement_successor(branches, index, branch)
        if same_stmt is not None:
            outcome.target_ei = same_stmt.id
            continue

        for candidate in branches[index + 1:]:
            if _is_skipped_successor(candidate, outcome):
                continue
            if _is_excluded_successor(branch, candidate):
                continue
            if _is_forbidden_successor(branch, candidate):
                continue
            outcome.target_ei = candidate.id
            break

        if outcome.target_ei is None:
            outcome.target_ei = implicit_return_sink


# ============================================================================
# AST node locator
# ============================================================================


class IndexedNodeLocator(ast.NodeVisitor):
    def __init__(self, module_fqn: str):
        self.module_fqn = module_fqn
        self.scope_stack: list[str] = [module_fqn] if module_fqn else []
        self.nodes_by_fqn_and_line: dict[tuple[str, int], ast.AST] = {}
        self.assignment_nodes_by_fqn_and_line: dict[tuple[str, int], ast.AST] = {}

    def current_fqn_prefix(self) -> str:
        return ".".join(self.scope_stack)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        fqn = f"{self.current_fqn_prefix()}.{node.name}" if self.scope_stack else node.name
        self.nodes_by_fqn_and_line[(fqn, node.lineno)] = node
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        fqn = f"{self.current_fqn_prefix()}.{node.name}" if self.scope_stack else node.name
        self.nodes_by_fqn_and_line[(fqn, node.lineno)] = node
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        fqn = f"{self.current_fqn_prefix()}.{node.name}" if self.scope_stack else node.name
        self.nodes_by_fqn_and_line[(fqn, node.lineno)] = node
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                fqn = f"{self.module_fqn}.{target.id}" if self.module_fqn else target.id
                self.assignment_nodes_by_fqn_and_line[(fqn, node.lineno)] = node
                break
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            fqn = f"{self.module_fqn}.{node.target.id}" if self.module_fqn else node.target.id
            self.assignment_nodes_by_fqn_and_line[(fqn, node.lineno)] = node
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            fqn = f"{self.module_fqn}.{node.target.id}" if self.module_fqn else node.target.id
            self.assignment_nodes_by_fqn_and_line[(fqn, node.lineno)] = node
        self.generic_visit(node)


# ============================================================================
# Function enumeration
# ============================================================================


def _owner_info_from_context(context: StatementContext) -> OwnerInfo | None:
    if not context.owners:
        return None

    owner = context.owners[-1]
    return OwnerInfo(
        stmt_type=type(owner.node).__name__,
        region=owner.region,
        line=getattr(owner.node, "lineno", None),
    )


def enumerate_function_eis(
        func_node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        source_lines: list[str],
        callable_id: str,
) -> FunctionResult:
    branches: list[Branch] = []
    ei_counter = 1

    if re.compile(FUNC_ID_EXPR).match(callable_id):
        first_ei_line = get_first_ei_line(func_node)
        branches.append(create_function_entry_branch(callable_id, func_node.lineno, first_ei_line))

    statement_contexts = get_statement_contexts(func_node)

    for context in statement_contexts:
        stmt = context.stmt

        if stmt == func_node:
            continue
        if not (func_node.lineno <= stmt.lineno <= func_node.end_lineno):
            continue

        decomp_context = _build_decomposition_context(context)
        outcomes = decompose_statement(stmt, source_lines, decomp_context)
        stmt_decorators = extract_statement_decorators(stmt, source_lines)

        for decomposed in outcomes or []:
            ei_id = generate_ei_id(callable_id, ei_counter)
            skips_lines: list[int] = []
            if decomposed.statement_outcome is not None:
                skips_lines = decomposed.statement_outcome.skips_lines

            condition, result, constraint = enrich_outcome_with_constraint(
                decomposed.description,
                decomposed.call_node,
                stmt,
                ei_id,
                stmt.lineno,
                skips_lines,
            )

            owner_info = _owner_info_from_context(context)

            branches.append(
                Branch(
                    id=ei_id,
                    line=stmt.lineno,
                    condition=condition,
                    description=decomposed.description,
                    constraint=constraint,
                    stmt_type=type(stmt).__name__,
                    decorators=stmt_decorators,
                    statement_outcome=decomposed.statement_outcome,
                    conditional_targets=decomposed.conditional_targets,
                    disruptive_outcomes=decomposed.disruptive_outcomes,
                    owner_info=owner_info,
                )
            )
            ei_counter += 1

    for branch in branches:
        _resolve_branch_target_line(branches, branch)

    _resolve_conditional_targets(branches)
    _resolve_skip_eis(branches)

    _redirect_explicit_implicit_returns(branches, callable_id)
    _assign_fallthrough_next_eis(branches, callable_id)
    _append_implicit_return_sink_if_referenced(branches, callable_id, func_node.end_lineno)

    return FunctionResult(
        name=getattr(func_node, "name", "<class>"),
        line_start=func_node.lineno,
        line_end=func_node.end_lineno,
        branches=branches,
    )


def enumerate_assignment_eis(
        node: ast.Assign | ast.AnnAssign | ast.AugAssign,
        source_lines: list[str],
        callable_id: str,
        target_name: str,
) -> FunctionResult | None:
    value = getattr(node, "value", None)
    if not isinstance(value, ast.Call):
        return None

    branches: list[Branch] = []
    ei_counter = 0

    outcomes: list[DecomposerResult] = decompose_statement(
        node,
        source_lines,
        DecompositionContext(
            next_stmt_lines=None,
        ),
    )

    for decomposed in outcomes or []:
        ei_counter += 1
        ei_id = generate_ei_id(callable_id, ei_counter)
        skips_lines: list[int] = []
        if decomposed.statement_outcome is not None:
            skips_lines = decomposed.statement_outcome.skips_lines

        condition, result, constraint = enrich_outcome_with_constraint(
            decomposed.description,
            decomposed.call_node,
            node,
            ei_id,
            node.lineno,
            skips_lines,
        )
        branches.append(
            Branch(
                id=ei_id,
                line=node.lineno,
                condition=condition,
                description=decomposed.description,
                constraint=constraint,
                stmt_type=type(node).__name__,
                statement_outcome=decomposed.statement_outcome,
                conditional_targets=decomposed.conditional_targets,
                disruptive_outcomes=decomposed.disruptive_outcomes,
            )
        )

    _resolve_skip_eis(branches)
    for branch in branches:
        _resolve_branch_target_line(branches, branch)
    _resolve_conditional_targets(branches)

    _redirect_explicit_implicit_returns(branches, callable_id)
    _assign_fallthrough_next_eis(branches, callable_id)
    _append_implicit_return_sink_if_referenced(branches, callable_id, node.end_lineno)

    function_result = FunctionResult(
        name=target_name,
        line_start=node.lineno,
        line_end=node.end_lineno,
        branches=branches,
    )
    populate_constraint_relationships(branches)
    return function_result


def _is_enum_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in ENUM_BASES:
            return True
        if isinstance(base, ast.Attribute) and base.attr in ENUM_BASES:
            return True
    return False


def _matches_target(entry: UnitIndexEntry, target_name: str | None) -> bool:
    return target_name is None or entry.name == target_name


def enumerate_unit_from_index(
        filepath: Path,
        unit_index: UnitIndex,
        function_name: str | None = None,
) -> list[FunctionResult]:
    source = filepath.read_text(encoding="utf-8")
    source_lines = source.split("\n")
    tree = ast.parse(source)

    locator = IndexedNodeLocator(unit_index.fully_qualified_name)
    locator.visit(tree)

    results: list[FunctionResult] = []

    for entry in unit_index.entries:
        if entry.kind not in ANALYZABLE_ENTRY_KINDS:
            continue
        if not _matches_target(entry, function_name):
            continue

        if entry.kind == "module_assignment":
            node: ast.Assign | ast.AnnAssign | ast.AugAssign | None = locator.assignment_nodes_by_fqn_and_line.get(
                (entry.fully_qualified_name, entry.lineno)
            )
            if node is None:
                continue
            assignment_result = enumerate_assignment_eis(node, source_lines, entry.id, entry.name)
            if assignment_result is not None:
                results.append(assignment_result)
            continue

        node = locator.nodes_by_fqn_and_line.get((entry.fully_qualified_name, entry.lineno))
        if node is None:
            continue

        if isinstance(node, ast.ClassDef):
            if not _is_enum_class(node):
                continue
            result = enumerate_function_eis(node, source_lines, entry.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result = enumerate_function_eis(node, source_lines, entry.id)
        else:
            continue

        populate_constraint_relationships(result.branches)
        results.append(result)

    return results


# ============================================================================
# CLI / formatting
# ============================================================================


def format_for_yaml(results: list[FunctionResult]) -> dict[str, Any]:
    if not results:
        return {}
    return {
        "module": "unknown",
        "functions": [result.to_dict() for result in results],
    }


def format_outcome_map_text(result: FunctionResult) -> str:
    lines: list[str] = []
    lines.append(f"=== {result.name} (lines {result.line_start}-{result.line_end}) ===")
    lines.append(f"Total EIs: {result.total_eis}")
    lines.append("")
    lines.append("Execution Items:")

    for branch in result.branches:
        lines.append(f"\n{branch.id} (Line {branch.line}):")
        lines.append(f"  Condition: {branch.condition}")
        lines.append(f"  Description: {branch.description}")

    return "\n".join(lines)


def _select_unit(project_index: ProjectIndex, filepath: Path) -> UnitIndex | None:
    resolved_file = filepath.resolve()

    for unit in project_index.units:
        if Path(unit.filepath).resolve() == resolved_file:
            return unit

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate Execution Items (EIs) from Python source using unit index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("file", type=Path, help="Python source file")
    parser.add_argument("--unit-index", required=True, type=Path, help="Structured stage 1 project index JSON")
    parser.add_argument("--function", "-f", help="Specific function name to enumerate")
    parser.add_argument("--text", action="store_true", help="Output human readable text instead of YAML")
    parser.add_argument("--output", "-o", type=Path, help="Save output to file")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1
    if not args.unit_index.exists():
        print(f"Error: Unit index not found: {args.unit_index}")
        return 1

    project_index = load_project_index(args.unit_index)
    unit = _select_unit(project_index, args.file)
    if unit is None:
        print(f"Error: No unit entry found in index for {args.file}")
        return 1

    results = enumerate_unit_from_index(args.file, unit, args.function)

    if not results:
        if args.function:
            print(f"Error: Function '{args.function}' not found in indexed entries for {args.file}")
        else:
            print(f"Error: No analyzable indexed entries found in {args.file}")
        return 1

    if args.text:
        output = "\n\n".join(format_outcome_map_text(result) for result in results)
    else:
        data = format_for_yaml(results)
        data["module"] = unit.fully_qualified_name
        output = yaml.dump(data, sort_keys=False, allow_unicode=True, width=float("inf"))

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Saved to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
