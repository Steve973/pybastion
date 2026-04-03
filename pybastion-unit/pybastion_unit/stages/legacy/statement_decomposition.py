"""
Statement decomposition

Context-aware statement decomposition using explicit outcome containers:
- StatementOutcome for normal routed flow
- ConditionalTarget for boolean control outcomes
- DisruptiveOutcome for disruptive/non-routed flow
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Any

from typing_extensions import Self

from pybastion_common.models import ConditionalTarget, TargetHint, DisruptiveOutcome, StatementOutcome
from pybastion_unit.shared.knowledge_base import NO_OP_CALLS


@dataclass
class DecomposerResult:
    description: str
    call_node: ast.Call | None = None
    statement_outcome: StatementOutcome | None = None
    conditional_targets: list[ConditionalTarget] | None = None
    disruptive_outcomes: list[DisruptiveOutcome] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "description": self.description,
        }
        if self.call_node is not None:
            result["call_expr"] = ast.unparse(self.call_node)
        if self.statement_outcome is not None:
            result["statement_outcome"] = self.statement_outcome.to_dict()
        if self.conditional_targets:
            result["conditional_targets"] = [t.to_dict() for t in self.conditional_targets]
        if self.disruptive_outcomes:
            result["disruptive_outcomes"] = [o.to_dict() for o in self.disruptive_outcomes]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        call_expr = data.get("call_expr")
        call_node = ast.parse(call_expr, mode="eval").body if call_expr else None
        if call_node is not None and not isinstance(call_node, ast.Call):
            raise ValueError(f"Invalid call expression: {call_expr}")
        return cls(
            description=data["description"],
            call_node=call_node,
            statement_outcome=(
                StatementOutcome.from_dict(data["statement_outcome"])
                if data.get("statement_outcome") else None
            ),
            conditional_targets=(
                [ConditionalTarget.from_dict(t) for t in data["conditional_targets"]]
                if data.get("conditional_targets") else None
            ),
            disruptive_outcomes=(
                [DisruptiveOutcome.from_dict(o) for o in data["disruptive_outcomes"]]
                if data.get("disruptive_outcomes") else None
            ),
        )


class OwnerKind(str, Enum):
    IF = "if"
    MATCH = "match"
    LOOP = "loop"
    TRY = "try"
    WITH = "with"


@dataclass(frozen=True)
class ControlOwner:
    kind: OwnerKind
    node: ast.stmt
    region: str | None = None
    next_stmt_lines: list[int] | None = None


@dataclass(frozen=True)
class DecompositionContext:
    next_stmt_lines: list[int] | None = None
    owners: tuple[ControlOwner, ...] = field(default_factory=tuple)

    def push(self, owner: ControlOwner, next_stmt_lines: list[int] | None = None) -> Self:
        return DecompositionContext(
            next_stmt_lines=self.next_stmt_lines if next_stmt_lines is None else next_stmt_lines,
            owners=(*self.owners, owner),
        )

    def nearest_owner(self, kind: OwnerKind) -> ControlOwner | None:
        for owner in reversed(self.owners):
            if owner.kind == kind:
                return owner
        return None

    @property
    def nearest_loop(self) -> ControlOwner | None:
        return self.nearest_owner(OwnerKind.LOOP)


@dataclass(frozen=True)
class StatementPart:
    node: ast.AST


def _is_no_op_call(node: ast.Call) -> bool:
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
    return [op[0] for op in operations if not _is_no_op_call(op[0])]


def _statement_outcome(
    outcome: str,
    *,
    target_line: int | None = None,
    skips_lines: list[int] | None = None,
    is_terminal: bool = False,
    terminates_via: str | None = None,
    target_hint: TargetHint | None = None,
) -> StatementOutcome:
    return StatementOutcome(
        outcome=outcome,
        target_line=target_line,
        skips_lines=[] if skips_lines is None else list(skips_lines),
        is_terminal=is_terminal,
        terminates_via=terminates_via,
        target_hint=target_hint,
    )


def _statement_result(
    description: str,
    *,
    target_line: int | None = None,
    skips_lines: list[int] | None = None,
    is_terminal: bool = False,
    terminates_via: str | None = None,
    target_hint: TargetHint | None = None,
    call_node: ast.Call | None = None,
) -> DecomposerResult:
    return DecomposerResult(
        description=description,
        call_node=call_node,
        statement_outcome=_statement_outcome(
            description,
            target_line=target_line,
            skips_lines=skips_lines,
            is_terminal=is_terminal,
            terminates_via=terminates_via,
            target_hint=target_hint,
        ),
    )


def _disruptive_result(
    description: str,
    *,
    target_line: int | None = None,
    skips_lines: list[int] | None = None,
    is_terminal: bool = False,
    terminates_via: str | None = None,
    target_hint: TargetHint | None = None,
    call_node: ast.Call | None = None,
) -> DecomposerResult:
    return DecomposerResult(
        description=description,
        call_node=call_node,
        disruptive_outcomes=[
            DisruptiveOutcome(
                outcome=description,
                target_line=target_line,
                skips_lines=[] if skips_lines is None else list(skips_lines),
                is_terminal=is_terminal,
                terminates_via=terminates_via,
                target_hint=target_hint,
            )
        ],
    )


def operation_eis(expressions: Iterable[ast.AST]) -> list[DecomposerResult]:
    results: list[DecomposerResult] = []
    for expr in expressions:
        for op in extract_all_operations(expr):
            text = ast.unparse(op)
            results.append(
                DecomposerResult(
                    description=f"{text} succeeds",
                    call_node=op,
                    statement_outcome=StatementOutcome(outcome=f"{text} succeeds"),
                    disruptive_outcomes=[
                        DisruptiveOutcome(
                            outcome="exception propagates",
                            is_terminal=True,
                            terminates_via="exception",
                            target_hint=TargetHint(
                                line=getattr(op, "lineno", None),
                                expr=text,
                            ),
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
    return _statement_result(
        description,
        target_line=next_line,
        skips_lines=skips_lines,
        is_terminal=is_terminal,
        terminates_via=terminates_via,
        target_hint=target_hint,
    )


def _body_lines(body: list[ast.stmt]) -> list[int]:
    if not body:
        return []
    return list(range(body[0].lineno, body[-1].end_lineno + 1))


def _select_target_from_chain(
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


def _implicit_return_result() -> DecomposerResult:
    return _statement_result(
        "implicit return",
        is_terminal=True,
        terminates_via="implicit-return",
    )


def _expr_hint(node: ast.AST, polarity: bool | None = None) -> TargetHint:
    return TargetHint(
        line=getattr(node, "lineno", None),
        expr=ast.unparse(node),
        polarity=polarity,
    )


def _wrap(text: str) -> str:
    return f"({text})"


def _join_and(parts: list[str]) -> str:
    if not parts:
        return "True"
    if len(parts) == 1:
        return parts[0]
    return " and ".join(_wrap(part) for part in parts)


def _negate_text(text: str) -> str:
    return f"not ({text})"


def enumerate_truth_conditions(node: ast.AST) -> tuple[list[str], list[str]]:
    match node:
        case ast.BoolOp(op=ast.And(), values=values):
            return _enumerate_and(values)
        case ast.BoolOp(op=ast.Or(), values=values):
            return _enumerate_or(values)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            operand_true, operand_false = enumerate_truth_conditions(operand)
            return operand_false, operand_true
        case _:
            expr = ast.unparse(node)
            return [expr], [_negate_text(expr)]


def _enumerate_and(values: list[ast.AST]) -> tuple[list[str], list[str]]:
    false_conditions: list[str] = []
    prior_true_prefixes: list[str] = []

    for value in values:
        value_true, value_false = enumerate_truth_conditions(value)

        if prior_true_prefixes:
            for prior in prior_true_prefixes:
                for false_cond in value_false:
                    false_conditions.append(_join_and([prior, false_cond]))
        else:
            false_conditions.extend(value_false)

        if not prior_true_prefixes:
            prior_true_prefixes = list(value_true)
        else:
            new_prefixes: list[str] = []
            for prior in prior_true_prefixes:
                for true_cond in value_true:
                    new_prefixes.append(_join_and([prior, true_cond]))
            prior_true_prefixes = new_prefixes

    return prior_true_prefixes, false_conditions


def _enumerate_or(values: list[ast.AST]) -> tuple[list[str], list[str]]:
    true_conditions: list[str] = []
    prior_false_prefixes: list[str] = []

    for value in values:
        value_true, value_false = enumerate_truth_conditions(value)

        if prior_false_prefixes:
            for prior in prior_false_prefixes:
                for true_cond in value_true:
                    true_conditions.append(_join_and([prior, true_cond]))
        else:
            true_conditions.extend(value_true)

        if not prior_false_prefixes:
            prior_false_prefixes = list(value_false)
        else:
            new_prefixes: list[str] = []
            for prior in prior_false_prefixes:
                for false_cond in value_false:
                    new_prefixes.append(_join_and([prior, false_cond]))
            prior_false_prefixes = new_prefixes

    return true_conditions, prior_false_prefixes


def _build_control_conditional_targets(
    test: ast.AST,
    *,
    true_target: int | None,
    false_target: int | None,
    false_terminates_via: str | None,
) -> list[ConditionalTarget]:
    true_conditions, false_conditions = enumerate_truth_conditions(test)
    condition = ast.unparse(test)
    targets: list[ConditionalTarget] = []

    for target_condition in true_conditions:
        targets.append(
            ConditionalTarget(
                target_condition=target_condition,
                condition_result=True,
                target_line=true_target,
                target_hint=TargetHint(
                    line=true_target,
                    expr=condition,
                    polarity=True,
                ),
            )
        )

    for target_condition in false_conditions:
        if false_terminates_via is None:
            targets.append(
                ConditionalTarget(
                    target_condition=target_condition,
                    condition_result=False,
                    target_line=false_target,
                    target_hint=TargetHint(
                        line=false_target,
                        expr=condition,
                        polarity=False,
                    ),
                )
            )
        else:
            targets.append(
                ConditionalTarget(
                    target_condition=target_condition,
                    condition_result=False,
                    is_terminal=True,
                    terminates_via=false_terminates_via,
                    target_hint=TargetHint(
                        line=getattr(test, "lineno", None),
                        expr=condition,
                        polarity=False,
                    ),
                )
            )

    return targets


def _build_control_disruptive_outcomes(test: ast.AST) -> list[DisruptiveOutcome] | None:
    ops = extract_all_operations(test)
    if not ops:
        return None
    outcomes: list[DisruptiveOutcome] = []
    for op in ops:
        text = ast.unparse(op)
        outcomes.append(
            DisruptiveOutcome(
                outcome=f"exception propagates from {text}",
                is_terminal=True,
                terminates_via="exception",
                target_hint=TargetHint(
                    line=getattr(op, "lineno", None),
                    expr=text,
                ),
            )
        )
    return outcomes


class StatementDecomposer:
    @classmethod
    def statement_parts(cls, stmt: ast.stmt, context: DecompositionContext) -> list[StatementPart]:
        return []

    @classmethod
    def semantic_eis(
        cls,
        stmt: ast.stmt,
        source_lines: list[str],
        context: DecompositionContext,
    ) -> list[DecomposerResult]:
        return []

    @classmethod
    def decompose(
        cls,
        stmt: ast.stmt,
        source_lines: list[str],
        context: DecompositionContext,
    ) -> list[DecomposerResult]:
        results: list[DecomposerResult] = []
        for part in cls.statement_parts(stmt, context):
            results.extend(decompose_expression_node(part.node, context))
        results.extend(cls.semantic_eis(stmt, source_lines, context))
        return results


class ControlOwnerDecomposer(StatementDecomposer):
    pass


class BoolOpDecomposer:
    @classmethod
    def decompose_expression_boolop(
        cls,
        boolop: ast.BoolOp,
        context: DecompositionContext,
    ) -> list[DecomposerResult]:
        values = list(boolop.values)
        if not values:
            return []

        results: list[DecomposerResult] = []
        for value in values:
            results.extend(decompose_expression_node(value, context))
        return results


class IfDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(
        cls,
        stmt: ast.If,
        context: DecompositionContext,
    ) -> list[StatementPart]:
        return []

    @classmethod
    def semantic_eis(
        cls,
        stmt: ast.If,
        source_lines: list[str],
        context: DecompositionContext,
    ) -> list[DecomposerResult]:
        results: list[DecomposerResult] = []

        current: ast.If | None = stmt
        while current is not None:
            true_target = current.body[0].lineno if current.body else None

            if current.orelse:
                false_target = current.orelse[0].lineno
                false_terminates_via = None
            elif context.next_stmt_lines:
                false_target = _select_target_from_chain(
                    context.next_stmt_lines,
                    _body_lines(current.body),
                )
                false_terminates_via = None if false_target is not None else "implicit-return"
            else:
                false_target = None
                false_terminates_via = "implicit-return"

            condition = ast.unparse(current.test)
            results.append(
                DecomposerResult(
                    description=f"evaluates {condition}",
                    conditional_targets=_build_control_conditional_targets(
                        current.test,
                        true_target=true_target,
                        false_target=false_target,
                        false_terminates_via=false_terminates_via,
                    ),
                    disruptive_outcomes=_build_control_disruptive_outcomes(current.test),
                )
            )

            if current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
            else:
                current = None

        return results


def decompose_expression_node(node: ast.AST, context: DecompositionContext) -> list[DecomposerResult]:
    if isinstance(node, ast.BoolOp):
        return BoolOpDecomposer.decompose_expression_boolop(node, context)
    if isinstance(node, ast.IfExp):
        results: list[DecomposerResult] = []
        results.extend(decompose_expression_node(node.test, context))
        results.extend(decompose_expression_node(node.body, context))
        results.extend(decompose_expression_node(node.orelse, context))
        return results
    if isinstance(node, ast.NamedExpr):
        return decompose_expression_node(node.value, context)
    if isinstance(node, ast.Await):
        return decompose_expression_node(node.value, context)
    if isinstance(node, ast.BinOp):
        return decompose_expression_node(node.left, context) + decompose_expression_node(node.right, context)
    if isinstance(node, ast.UnaryOp):
        return decompose_expression_node(node.operand, context)
    if isinstance(node, ast.Call):
        return operation_eis([node])
    if isinstance(node, ast.Compare):
        return operation_eis([node])
    if isinstance(node, ast.keyword):
        return decompose_expression_node(node.value, context)
    if isinstance(node, ast.Subscript):
        return decompose_expression_node(node.value, context) + decompose_expression_node(node.slice, context)
    if isinstance(node, ast.Slice):
        results: list[DecomposerResult] = []
        if node.lower:
            results.extend(decompose_expression_node(node.lower, context))
        if node.upper:
            results.extend(decompose_expression_node(node.upper, context))
        if node.step:
            results.extend(decompose_expression_node(node.step, context))
        return results
    if isinstance(node, ast.Attribute):
        return []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        results: list[DecomposerResult] = []
        for elt in node.elts:
            results.extend(decompose_expression_node(elt, context))
        return results
    if isinstance(node, ast.Dict):
        results: list[DecomposerResult] = []
        for key in node.keys:
            if key is not None:
                results.extend(decompose_expression_node(key, context))
        for value in node.values:
            results.extend(decompose_expression_node(value, context))
        return results
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return operation_eis([node])
    if isinstance(node, (ast.Name, ast.Constant)):
        return []
    return operation_eis([node])


class SingularStatementDecomposer(StatementDecomposer):
    pass


class MatchDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Match, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.subject)]

    @classmethod
    def semantic_eis(cls, stmt: ast.Match, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        results: list[DecomposerResult] = []
        for case in stmt.cases:
            pattern = ast.unparse(case.pattern)
            guard = f" if {ast.unparse(case.guard)}" if case.guard else ""
            target = case.body[0].lineno if case.body else None
            results.append(_statement_result(f"case {pattern}{guard} matches", target_line=target))
        return results


class ForDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.For, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.iter)]

    @classmethod
    def semantic_eis(cls, stmt: ast.For, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        target = ast.unparse(stmt.target)
        iter_expr = ast.unparse(stmt.iter)
        body_lines = _body_lines(stmt.body)
        loop_body_target = stmt.body[0].lineno if stmt.body else None
        exit_target = stmt.orelse[0].lineno if stmt.orelse else _select_target_from_chain(context.next_stmt_lines, body_lines)
        return [
            _statement_result(
                f"for {target} in {iter_expr}: 0 iterations",
                target_line=exit_target,
                skips_lines=body_lines,
            ),
            _statement_result(
                f"for {target} in {iter_expr}: ≥1 iterations",
                target_line=loop_body_target,
            ),
        ]


class AsyncForDecomposer(ForDecomposer):
    pass


class WhileDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.While, context: DecompositionContext) -> list[StatementPart]:
        return []

    @classmethod
    def semantic_eis(cls, stmt: ast.While, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        body_lines = _body_lines(stmt.body)
        loop_body_target = stmt.body[0].lineno if stmt.body else None
        exit_target = stmt.orelse[0].lineno if stmt.orelse else _select_target_from_chain(context.next_stmt_lines, body_lines)
        condition = ast.unparse(stmt.test)
        return [
            DecomposerResult(
                description=f"evaluates {condition}",
                conditional_targets=_build_control_conditional_targets(
                    stmt.test,
                    true_target=loop_body_target,
                    false_target=exit_target,
                    false_terminates_via=None if exit_target is not None else "implicit-return",
                ),
                disruptive_outcomes=_build_control_disruptive_outcomes(stmt.test),
            )
        ]


class TryDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Try, context: DecompositionContext) -> list[StatementPart]:
        parts: list[StatementPart] = []
        for handler in stmt.handlers:
            if handler.type:
                parts.append(StatementPart(handler.type))
        return parts

    @classmethod
    def semantic_eis(cls, stmt: ast.Try, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        if stmt.orelse:
            success_target = stmt.orelse[0].lineno
        elif stmt.finalbody:
            success_target = stmt.finalbody[0].lineno
        elif context.next_stmt_lines:
            success_target = context.next_stmt_lines[0]
        else:
            success_target = None

        results = [_statement_result("try block executes successfully", target_line=success_target)]
        for handler in stmt.handlers:
            handler_target = handler.body[0].lineno if handler.body else None
            if handler.type:
                results.append(_statement_result(f"raises {ast.unparse(handler.type)} → enters except handler", target_line=handler_target))
            else:
                results.append(_statement_result("raises exception → enters except handler", target_line=handler_target))
        if stmt.orelse:
            results.append(_statement_result("try succeeds without exception → enters else block", target_line=stmt.orelse[0].lineno))
        if stmt.finalbody:
            results.append(_statement_result("protected flow completes → enters finally block", target_line=stmt.finalbody[0].lineno))
        return results


class WithDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.With, context: DecompositionContext) -> list[StatementPart]:
        return []

    @classmethod
    def semantic_eis(cls, stmt: ast.With, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        contexts = ", ".join(ast.unparse(item.context_expr) for item in stmt.items)
        body_target = stmt.body[0].lineno if stmt.body else None
        disruptive_outcomes = []
        for item in stmt.items:
            for op in extract_all_operations(item.context_expr):
                disruptive_outcomes.append(
                    DisruptiveOutcome(
                        outcome=f"exception propagates from {ast.unparse(op)}",
                        is_terminal=True,
                        terminates_via="exception",
                        target_hint=TargetHint(line=getattr(op, 'lineno', None), expr=ast.unparse(op)),
                    )
                )
        return [
            DecomposerResult(
                description=f"with {contexts}: enters successfully",
                statement_outcome=StatementOutcome(
                    outcome=f"with {contexts}: enters successfully",
                    target_line=body_target,
                ),
                disruptive_outcomes=disruptive_outcomes or None,
            )
        ]


class AsyncWithDecomposer(WithDecomposer):
    pass


class AssignDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Assign, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)]

    @classmethod
    def semantic_eis(cls, stmt: ast.Assign, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


class AugAssignDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.AugAssign, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)]

    @classmethod
    def semantic_eis(cls, stmt: ast.AugAssign, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


class AnnAssignDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.AnnAssign, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)] if stmt.value else []

    @classmethod
    def semantic_eis(cls, stmt: ast.AnnAssign, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


class ExprDecomposer(SingularStatementDecomposer):
    @classmethod
    def decompose(cls, stmt: ast.Expr, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            return []
        return super().decompose(stmt, source_lines, context)

    @classmethod
    def statement_parts(cls, stmt: ast.Expr, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)]

    @classmethod
    def semantic_eis(cls, stmt: ast.Expr, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        if extract_all_operations(stmt.value):
            return []
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


class ReturnDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Return, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)] if stmt.value else []

    @classmethod
    def semantic_eis(cls, stmt: ast.Return, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        ret_val = ast.unparse(stmt.value) if stmt.value else "None"
        return [_statement_result(f"returns {ret_val}", is_terminal=True, terminates_via="return")]


class RaiseDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Raise, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.exc)] if stmt.exc else []

    @classmethod
    def semantic_eis(cls, stmt: ast.Raise, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        exc = ast.unparse(stmt.exc) if stmt.exc else "exception"
        return [_disruptive_result(f"raises {exc}", is_terminal=True, terminates_via="raise")]


class BreakDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Break, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        loop_owner = context.nearest_loop
        next_lines = loop_owner.next_stmt_lines if loop_owner else context.next_stmt_lines
        return [semantic("break", next_lines)]


class ContinueDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Continue, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        loop_owner = context.nearest_loop
        next_lines = loop_owner.next_stmt_lines if loop_owner else context.next_stmt_lines
        return [semantic("continue", next_lines)]


class PassDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Pass, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        return [semantic("pass", context.next_stmt_lines)]


class DeleteDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Delete, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        return [semantic(f"del {', '.join(ast.unparse(t) for t in stmt.targets)}", context.next_stmt_lines)]


class ImportDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Import, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        return [semantic(f"import {', '.join(alias.name for alias in stmt.names)}", context.next_stmt_lines)]


class ImportFromDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.ImportFrom, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        module = stmt.module or ""
        names = ", ".join(alias.name for alias in stmt.names)
        return [semantic(f"from {module} import {names}", context.next_stmt_lines)]


class AssertDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Assert, context: DecompositionContext) -> list[StatementPart]:
        return []

    @classmethod
    def semantic_eis(cls, stmt: ast.Assert, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        condition = ast.unparse(stmt.test)
        false_message = f"{condition} is false → raises AssertionError"
        return [
            DecomposerResult(
                description=f"evaluates {condition}",
                conditional_targets=_build_control_conditional_targets(
                    stmt.test,
                    true_target=context.next_stmt_lines[0] if context.next_stmt_lines and len(context.next_stmt_lines) == 1 else None,
                    false_target=None,
                    false_terminates_via="raise",
                ),
                disruptive_outcomes=[
                    DisruptiveOutcome(
                        outcome=false_message,
                        is_terminal=True,
                        terminates_via="raise",
                        target_hint=TargetHint(line=getattr(stmt.test, 'lineno', None), expr=condition, polarity=False),
                    ),
                    *(_build_control_disruptive_outcomes(stmt.test) or []),
                ],
            )
        ]


class GlobalDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Global, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        return [semantic(f"global {', '.join(stmt.names)}", context.next_stmt_lines)]


class NonlocalDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Nonlocal, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        return [semantic(f"nonlocal {', '.join(stmt.names)}", context.next_stmt_lines)]


class DefaultDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.stmt, source_lines: list[str], context: DecompositionContext) -> list[DecomposerResult]:
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


_DECOMPOSERS: dict[type[ast.stmt], type[StatementDecomposer]] = {
    ast.If: IfDecomposer,
    ast.Match: MatchDecomposer,
    ast.For: ForDecomposer,
    ast.AsyncFor: AsyncForDecomposer,
    ast.While: WhileDecomposer,
    ast.Try: TryDecomposer,
    ast.With: WithDecomposer,
    ast.AsyncWith: AsyncWithDecomposer,
    ast.Assign: AssignDecomposer,
    ast.AugAssign: AugAssignDecomposer,
    ast.AnnAssign: AnnAssignDecomposer,
    ast.Expr: ExprDecomposer,
    ast.Return: ReturnDecomposer,
    ast.Raise: RaiseDecomposer,
    ast.Break: BreakDecomposer,
    ast.Continue: ContinueDecomposer,
    ast.Pass: PassDecomposer,
    ast.Delete: DeleteDecomposer,
    ast.Assert: AssertDecomposer,
    ast.Import: ImportDecomposer,
    ast.ImportFrom: ImportFromDecomposer,
    ast.Global: GlobalDecomposer,
    ast.Nonlocal: NonlocalDecomposer,
}

_DEFAULT_DECOMPOSER = DefaultDecomposer


def decompose_statement(
    stmt: ast.stmt,
    source_lines: list[str],
    context: DecompositionContext | None = None,
) -> list[DecomposerResult]:
    effective_context = context or DecompositionContext()
    decomposer = _DECOMPOSERS.get(type(stmt), _DEFAULT_DECOMPOSER)
    return decomposer.decompose(stmt, source_lines, effective_context)
