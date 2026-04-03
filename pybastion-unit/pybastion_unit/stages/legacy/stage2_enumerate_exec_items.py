#!/usr/bin/env python3
"""
Execution Item Enumerator - Context aware stage 2 rewrite

This version keeps the useful parts of the current script, but changes the
statement collection model so stage 2 can reason about owning control contexts
before decomposition. That is especially important for break and continue.

Key design choices:
- singular statements still use the existing decomposition helpers
- stage 2, not semantic(), decides what continuation chain a nested statement sees
- break and continue are resolved in loop context, not by whatever local nested
  block happened to contain them
- explicit next_ei / target_ei resolution is preferred, then safe fall through
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pybastion_common.models import Branch
from pybastion_unit.helpers.constraint_metadata_helper import (
    enrich_outcome_with_constraint,
    populate_constraint_relationships,
)
from pybastion_unit.helpers.decorator_processing import extract_statement_decorators
from pybastion_unit.stages.legacy.statement_decomposition import (
    ControlOwner,
    DecomposerResult,
    DecompositionContext,
    OwnerKind,
    decompose_statement,
)
from pybastion_unit.shared.callable_id_generation import (
    FUNC_ID_EXPR,
    generate_assignment_id,
    generate_ei_id,
    generate_function_entry_ei_id,
    generate_function_id,
)

ENUM_BASES: set[str] = {
    "Enum",
    "IntEnum",
    "StrEnum",
    "Flag",
    "IntFlag"
}


# ============================================================================
# Inventory / FQN helpers
# ============================================================================


def load_callable_inventory(filepath: Path | None) -> dict[str, str]:
    inventory: dict[str, str] = {}
    if not filepath or not filepath.exists():
        return inventory

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            fqn, callable_id = line.split(":", 1)
            inventory[fqn] = callable_id

    return inventory


def derive_fqn_from_path(filepath: Path, source_root: Path | None) -> str:
    if not source_root:
        return filepath.stem

    try:
        relative = filepath.relative_to(source_root)
    except ValueError:
        return filepath.stem

    parts = list(relative.parts[:-1]) + [relative.stem]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


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
    """
    Retrieves a list of `StatementContext` objects representing the execution contexts for
    statements within an abstract syntax tree (AST) node.

    The function traverses the specified AST node and its descendants, capturing detailed
    information about the context of each statement, including ownership by control-flow
    structures like `if` statements, loops, `with` statements, and `try` blocks. Callable
    on function definitions, asynchronous function definitions, class definitions, and
    modules.

    Parameters:
        node (ast.AST): The root AST node to process. Must be one of `FunctionDef`,
            `AsyncFunctionDef`, `ClassDef`, or `Module`.

    Returns:
        list[StatementContext]: A list of `StatementContext` objects, each containing
            the statement, its next potential statement lines, and its ownership
            hierarchy.
    """
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
    """
    Builds and returns a new decomposition context based on the provided statement context.

    This function creates a new instance of DecompositionContext by extracting and
    utilizing specific attributes from the given StatementContext. It serves as a
    utility for preparing decomposition data within a contextual scope.

    Args:
        context (StatementContext): The source context containing attributes necessary
        to construct the decomposition context.

    Returns:
        DecompositionContext: A new decomposition context initialized with relevant
        attributes from the provided statement context.
    """
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
            "branches": [b.to_dict() for b in self.branches],
        }


def create_function_entry_branch(
        callable_id: str,
        line_num: int,
        target_line: int | None = None,
) -> Branch:
    return Branch(
        id=generate_function_entry_ei_id(callable_id),
        condition=f"enters function {callable_id}",
        outcome="function start",
        stmt_type="FunctionInvocation",
        synthetic=True,
        line=line_num,
        target_line=target_line,
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


def _resolve_branch_target_line(branches: list[Branch], branch: Branch) -> None:
    if branch.target_line is None:
        return

    for candidate in branches:
        if candidate.line == branch.target_line:
            branch.next_ei = candidate.id
            break

    branch.target_line = None


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
                    and branch.outcome.startswith("evaluates ")
                    and target.target_line == branch.line
            ):
                sibling_target = _resolve_same_line_if_target(branches, branch, target.condition)
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
        if not branch.constraint or not branch.constraint.skips_eis:
            continue

        resolved_ids: list[str] = []
        for raw in branch.constraint.skips_eis:
            try:
                line_num = int(raw)
            except (TypeError, ValueError):
                continue
            resolved_ids.extend(line_to_eis.get(line_num, []))

        # Preserve order while removing duplicates
        seen: set[str] = set()
        branch.constraint.skips_eis = [ei for ei in resolved_ids if not (ei in seen or seen.add(ei))]


def _is_excluded_successor(current: Branch, candidate: Branch) -> bool:
    if current.constraint is None:
        return False
    return candidate.id in (current.constraint.excludes or [])


def _is_skipped_successor(current: Branch, candidate: Branch) -> bool:
    if current.constraint is None:
        return False
    return candidate.id in (current.constraint.skips_eis or [])


def _same_statement_successor(
    branches: list[Branch],
    index: int,
    branch: Branch,
) -> Branch | None:
    if branch.stmt_type not in {"Raise", "Return"}:
        return None

    for candidate in branches[index + 1:]:
        # stop once we leave this statement
        if candidate.line != branch.line or candidate.stmt_type != branch.stmt_type:
            break

        if _is_skipped_successor(branch, candidate):
            continue
        if _is_excluded_successor(branch, candidate):
            continue

        # skip exception-alternative branches when we are following a success path
        if candidate.terminates_via == "exception":
            continue
        if "exception propagates" in candidate.outcome.lower():
            continue
        if " raises exception" in candidate.condition.lower():
            continue

        return candidate

    return None


def _assign_fallthrough_next_eis(branches: list[Branch]) -> None:
    """
    Final safe fallthrough.

    Priority:
    1. explicit next_ei already assigned
    2. same-statement sequential continuation for Raise/Return
    3. generic forward fallthrough respecting skips/excludes
    """
    for index, branch in enumerate(branches):
        if branch.next_ei or branch.is_terminal:
            continue

        same_stmt = _same_statement_successor(branches, index, branch)
        if same_stmt is not None:
            branch.next_ei = same_stmt.id
            continue

        for candidate in branches[index + 1:]:
            if candidate.is_terminal:
                continue
            if _is_skipped_successor(branch, candidate):
                continue
            if _is_excluded_successor(branch, candidate):
                continue
            branch.next_ei = candidate.id
            break


# ============================================================================
# Function enumeration
# ============================================================================


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
            condition, result, constraint = enrich_outcome_with_constraint(
                decomposed.outcome,
                decomposed.call_node,
                stmt,
                ei_id,
                stmt.lineno,
                decomposed.skips_lines,
            )

            branches.append(
                Branch(
                    id=ei_id,
                    line=stmt.lineno,
                    condition=condition,
                    outcome=result,
                    constraint=constraint,
                    stmt_type=type(stmt).__name__,
                    decorators=stmt_decorators,
                    target_line=decomposed.target_line,
                    conditional_targets=decomposed.conditional_targets,
                )
            )
            ei_counter += 1

    for branch in branches:
        _resolve_branch_target_line(branches, branch)

    _resolve_conditional_targets(branches)
    _resolve_skip_eis(branches)
    _assign_fallthrough_next_eis(branches)

    return FunctionResult(
        name=getattr(func_node, "name", "<class>"),
        line_start=func_node.lineno,
        line_end=func_node.end_lineno,
        branches=branches,
    )


# ============================================================================
# File processing
# ============================================================================


class CallableFinder(ast.NodeVisitor):
    def __init__(
            self,
            module_fqn: str,
            source_lines: list[str],
            inventory: dict[str, str],
            unit_id: str,
            target_name: str | None,
    ) -> None:
        self.module_fqn = module_fqn
        self.source_lines = source_lines
        self.inventory = inventory
        self.unit_id = unit_id
        self.target_name = target_name
        self.results: list[FunctionResult] = []
        self.fqn_stack = [module_fqn] if module_fqn else []
        self.func_counter = 1
        self.assignment_counter = 1
        self.function_depth = 0

    @staticmethod
    def _is_enum_class(node: ast.ClassDef) -> bool:
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ENUM_BASES:
                return True
            if isinstance(base, ast.Attribute) and base.attr in ENUM_BASES:
                return True
        return False

    def _enumerate_and_record(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
            fqn: str,
    ) -> None:
        callable_id = self.inventory.get(fqn)
        if not callable_id:
            callable_id = generate_function_id(self.unit_id, self.func_counter)

        result = enumerate_function_eis(node, self.source_lines, callable_id)
        populate_constraint_relationships(result.branches)
        self.results.append(result)
        self.func_counter += 1

    def visit_Assign(self, node: ast.Assign) -> None:
        self._process_assignment(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.function_depth == 0:
            self._process_assignment(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._process_assignment(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.fqn_stack.append(node.name)
        if self._is_enum_class(node):
            fqn = ".".join(self.fqn_stack)
            self._enumerate_and_record(node, fqn)
        else:
            self.generic_visit(node)
        self.fqn_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_depth += 1
        self._process_function(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_depth += 1
        self._process_function(node)
        self.function_depth -= 1

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.target_name and node.name != self.target_name:
            return

        fqn = f"{'.'.join(self.fqn_stack)}.{node.name}" if self.fqn_stack else node.name
        self._enumerate_and_record(node, fqn)

        self.fqn_stack.append(node.name)
        for item in ast.walk(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item is not node:
                self._process_function(item)
        self.fqn_stack.pop()

    def _process_assignment(self, node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> None:
        if not isinstance(node.value, ast.Call):
            return

        if isinstance(node, ast.Assign):
            if not node.targets:
                return
            first_target = node.targets[0]
            if not isinstance(first_target, ast.Name):
                return
            target_name = first_target.id
        else:
            if not isinstance(node.target, ast.Name):
                return
            target_name = node.target.id

        fqn = f"{self.module_fqn}.{target_name}" if self.module_fqn else target_name
        callable_id = self.inventory.get(fqn)
        if not callable_id:
            callable_id = generate_assignment_id(self.unit_id, self.assignment_counter)

        self.assignment_counter += 1
        branches: list[Branch] = []
        ei_counter = 0

        outcomes: list[DecomposerResult] = decompose_statement(
            node,
            self.source_lines,
            DecompositionContext(
                next_stmt_lines=None,
            ),
        )

        for decomposed in outcomes or []:
            ei_counter += 1
            ei_id = generate_ei_id(callable_id, ei_counter)
            condition, result, constraint = enrich_outcome_with_constraint(
                decomposed.outcome,
                decomposed.call_node,
                node,
                ei_id,
                node.lineno,
                decomposed.skips_lines,
            )
            branches.append(
                Branch(
                    id=ei_id,
                    line=node.lineno,
                    condition=condition,
                    outcome=result,
                    constraint=constraint,
                    stmt_type=type(node).__name__,
                    target_line=decomposed.target_line,
                    conditional_targets=decomposed.conditional_targets,
                )
            )

        _resolve_skip_eis(branches)
        for branch in branches:
            _resolve_branch_target_line(branches, branch)
        _resolve_conditional_targets(branches)
        _assign_fallthrough_next_eis(branches)

        function_result = FunctionResult(
            name=target_name,
            line_start=node.lineno,
            line_end=node.end_lineno,
            branches=branches,
        )
        populate_constraint_relationships(branches)
        self.results.append(function_result)


def enumerate_file(
        filepath: Path,
        unit_id: str,
        function_name: str | None = None,
        callable_inventory: dict[str, str] | None = None,
        module_fqn: str | None = None,
) -> list[FunctionResult]:
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    source_lines = source.split("\n")
    tree = ast.parse(source)
    inventory = callable_inventory or {}

    finder = CallableFinder(
        module_fqn or "",
        source_lines,
        inventory,
        unit_id,
        function_name,
    )
    finder.visit(tree)
    return finder.results


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
        lines.append(f"  Outcome: {branch.outcome}")

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate Execution Items (EIs) from Python source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mymodule.py --output mymodule_eis.yaml
  %(prog)s mymodule.py --function validate_typed_dict
  %(prog)s mymodule.py --text
        """,
    )

    parser.add_argument("file", type=Path, help="Python source file")
    parser.add_argument("--unit-id", "-u", required=True, help="Unit ID (required)")
    parser.add_argument("--function", "-f", help="Specific function name to enumerate")
    parser.add_argument("--callable-inventory", type=Path, help="Callable inventory file (FQN:ID pairs)")
    parser.add_argument("--source-root", type=Path, help="Source root for deriving FQN")
    parser.add_argument("--text", action="store_true", help="Output human readable text instead of YAML")
    parser.add_argument("--output", "-o", type=Path, help="Save output to file")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    inventory = load_callable_inventory(args.callable_inventory) if args.callable_inventory else {}
    module_fqn = derive_fqn_from_path(args.file, args.source_root) if args.source_root else None

    results = enumerate_file(
        args.file,
        args.unit_id,
        args.function,
        inventory,
        module_fqn,
    )

    if not results:
        if args.function:
            print(f"Error: Function '{args.function}' not found in {args.file}")
        else:
            print(f"Error: No functions found in {args.file}")
        return 1

    if args.text:
        output = "\n\n".join(format_outcome_map_text(result) for result in results)
    else:
        data = format_for_yaml(results)
        data["module"] = args.file.stem
        output = yaml.dump(data, sort_keys=False, allow_unicode=True, width=float("inf"))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
