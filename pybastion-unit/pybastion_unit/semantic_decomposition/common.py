from __future__ import annotations

import ast
from typing import Iterable

from pybastion_common.models import (
    DisruptiveOutcome,
    StatementOutcome,
    TargetHint,
)
from pybastion_unit.shared.knowledge_base import NO_OP_CALLS
from .decomp_types import DecomposerResult


def is_no_op_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return any(fqn.endswith(f".{func.id}") for fqn in NO_OP_CALLS)
    if isinstance(func, ast.Attribute):
        return ast.unparse(func) in NO_OP_CALLS
    return False


def extract_all_operations(node: ast.AST) -> list[ast.Call]:
    operations: list[tuple[ast.Call, int, int, int]] = []

    def collect(n: ast.AST, depth: int = 0) -> None:
        if isinstance(n, ast.Call):
            operations.append((n, depth, n.lineno, n.col_offset))
        for child in ast.iter_child_nodes(n):
            collect(child, depth + 1)

    collect(node)
    operations.sort(key=lambda item: (-item[1], item[2], item[3]))
    return [op[0] for op in operations if not is_no_op_call(op[0])]


def operation_eis(expressions: Iterable[ast.AST]) -> list[DecomposerResult]:
    results: list[DecomposerResult] = []
    for expr in expressions:
        for op in extract_all_operations(expr):
            text = ast.unparse(op)
            results.append(
                DecomposerResult(
                    description=f"{text} succeeds",
                    call_node=op,
                    statement_outcome=StatementOutcome(
                        outcome=f"{text} succeeds",
                    ),
                    disruptive_outcomes=[
                        DisruptiveOutcome(
                            outcome="exception propagates",
                            is_terminal=True,
                            terminates_via="exception",
                        )
                    ],
                )
            )
    return results


def semantic(
        description: str,
        next_stmt_lines: list[int] | None = None,
        *,
        skips_lines: list[int] | None = None,
        is_terminal: bool = False,
        terminates_via: str | None = None,
        target_hint: TargetHint | None = None,
) -> DecomposerResult:
    next_line = next_stmt_lines[0] if next_stmt_lines and len(next_stmt_lines) == 1 else None
    return DecomposerResult(
        description=description,
        statement_outcome=StatementOutcome(
            outcome=description,
            target_line=next_line,
            skips_lines=[] if skips_lines is None else list(skips_lines),
            is_terminal=is_terminal,
            terminates_via=terminates_via,
            target_hint=target_hint,
        ),
    )


def body_lines(body: list[ast.stmt]) -> list[int]:
    if not body:
        return []
    return list(range(body[0].lineno, body[-1].end_lineno + 1))


def select_target_from_chain(
        next_stmt_lines: list[int] | None,
        skips_lines: list[int] | None,
) -> int | None:
    if not next_stmt_lines:
        return None
    if not skips_lines:
        return next_stmt_lines[0]
    for line in next_stmt_lines:
        if line not in skips_lines:
            return line
    return None


def implicit_return_result() -> DecomposerResult:
    return DecomposerResult(
        description="implicit return",
        statement_outcome=StatementOutcome(
            outcome="implicit return",
            is_terminal=True,
            terminates_via="implicit-return",
        ),
    )


def expr_hint(node: ast.AST, polarity: bool | None = None) -> TargetHint:
    return TargetHint(
        line=getattr(node, "lineno", None),
        expr=ast.unparse(node),
        polarity=polarity,
    )


def _and_join(parts: list[str]) -> str:
    if not parts:
        return "True"
    if len(parts) == 1:
        return parts[0]
    return " and ".join(f"({part})" for part in parts)


def _negated_expr(node: ast.AST) -> str:
    return f"not ({ast.unparse(node)})"


def _enumerate_target_conditions(test: ast.AST) -> list[tuple[str, bool]]:
    if isinstance(test, ast.BoolOp):
        values = list(test.values)

        if isinstance(test.op, ast.And):
            outcomes: list[tuple[str, bool]] = []
            prefix: list[str] = []

            for value in values:
                outcomes.append((_and_join([*prefix, _negated_expr(value)]), False))
                prefix.append(ast.unparse(value))

            outcomes.append((_and_join(prefix), True))
            return outcomes

        if isinstance(test.op, ast.Or):
            outcomes: list[tuple[str, bool]] = []
            prefix: list[str] = []

            for value in values:
                outcomes.append((_and_join([*prefix, ast.unparse(value)]), True))
                prefix.append(_negated_expr(value))

            outcomes.append((_and_join(prefix), False))
            return outcomes

    expr = ast.unparse(test)
    return [
        (expr, True),
        (f"not ({expr})", False),
    ]
