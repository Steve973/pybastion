from __future__ import annotations

import ast

from pybastion_common.models import (
    ConditionalTarget,
    DisruptiveOutcome,
    StatementOutcome,
    TargetHint,
)
from .base import ControlOwnerDecomposer
from .common import body_lines, select_target_from_chain
from .decomp_types import DecompositionContext, DecomposerResult, StatementPart
from .expressions import enumerate_truth_conditions


def _build_if_conditional_targets(
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


class IfDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.If, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.test)]

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
                false_target = select_target_from_chain(
                    context.next_stmt_lines,
                    body_lines(current.body),
                )
                false_terminates_via = None if false_target is not None else "implicit-return"
            else:
                false_target = None
                false_terminates_via = "implicit-return"

            condition = ast.unparse(current.test)

            results.append(
                DecomposerResult(
                    description=f"evaluates {condition}",
                    conditional_targets=_build_if_conditional_targets(
                        current.test,
                        true_target=true_target,
                        false_target=false_target,
                        false_terminates_via=false_terminates_via,
                    ),
                )
            )

            if current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
            else:
                current = None

        return results


class MatchDecomposer(ControlOwnerDecomposer):
    @staticmethod
    def _case_label(index: int, case: ast.match_case) -> str:
        pattern = ast.unparse(case.pattern)
        if case.guard is not None:
            return f"match case {index}: {pattern} if {ast.unparse(case.guard)}"
        return f"match case {index}: {pattern}"

    @staticmethod
    def _join_and(parts: list[str]) -> str:
        if not parts:
            return "True"
        if len(parts) == 1:
            return parts[0]
        return " and ".join(f"({part})" for part in parts)

    @staticmethod
    def _is_unconditional_wildcard(case: ast.match_case) -> bool:
        return (
            isinstance(case.pattern, ast.MatchAs)
            and case.pattern.name is None
            and case.guard is None
        )

    @classmethod
    def semantic_eis(
        cls,
        stmt: ast.Match,
        source_lines: list[str],
        context: DecompositionContext,
    ) -> list[DecomposerResult]:
        subject = ast.unparse(stmt.subject)

        conditional_targets: list[ConditionalTarget] = []
        prior_failures: list[str] = []

        all_case_lines: list[int] = []
        for case in stmt.cases:
            all_case_lines.extend(body_lines(case.body))

        for index, case in enumerate(stmt.cases, start=1):
            case_label = cls._case_label(index, case)
            case_lines = body_lines(case.body)

            case_target = case.body[0].lineno if case.body else select_target_from_chain(
                context.next_stmt_lines,
                case_lines,
            )

            success_condition = cls._join_and([*prior_failures, case_label])

            conditional_targets.append(
                ConditionalTarget(
                    target_condition=success_condition,
                    condition_result=True,
                    target_line=case_target,
                    target_hint=TargetHint(
                        line=case_target,
                        expr=case_label,
                        polarity=True,
                    ),
                )
            )

            if cls._is_unconditional_wildcard(case):
                break

            prior_failures.append(f"not ({case_label})")

        if not any(cls._is_unconditional_wildcard(case) for case in stmt.cases):
            fallthrough_target = select_target_from_chain(
                context.next_stmt_lines,
                all_case_lines,
            )

            no_match_condition = cls._join_and(prior_failures) if prior_failures else "no match cases"

            if fallthrough_target is not None:
                conditional_targets.append(
                    ConditionalTarget(
                        target_condition=no_match_condition,
                        condition_result=False,
                        target_line=fallthrough_target,
                        target_hint=TargetHint(
                            line=fallthrough_target,
                            expr=f"match {subject}",
                            polarity=False,
                        ),
                    )
                )
            else:
                conditional_targets.append(
                    ConditionalTarget(
                        target_condition=no_match_condition,
                        condition_result=False,
                        is_terminal=True,
                        terminates_via="implicit-return",
                        target_hint=TargetHint(
                            line=getattr(stmt, "lineno", None),
                            expr=f"match {subject}",
                            polarity=False,
                        ),
                    )
                )

        return [
            DecomposerResult(
                description=f"evaluates match {subject}",
                conditional_targets=conditional_targets,
            )
        ]


class ForDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.For, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.iter)]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.For,
            source_lines: list[str],
            context: DecompositionContext,
    ) -> list[DecomposerResult]:
        target = ast.unparse(stmt.target)
        iter_expr = ast.unparse(stmt.iter)
        loop_expr = f"for {target} in {iter_expr}"

        loop_body = body_lines(stmt.body)
        loop_body_target = stmt.body[0].lineno if stmt.body else None

        if stmt.orelse:
            false_target = stmt.orelse[0].lineno
            false_terminates_via = None
        elif context.next_stmt_lines:
            false_target = select_target_from_chain(
                context.next_stmt_lines,
                loop_body,
            )
            false_terminates_via = None if false_target is not None else "implicit-return"
        else:
            false_target = None
            false_terminates_via = "implicit-return"

        conditional_targets: list[ConditionalTarget] = [
            ConditionalTarget(
                target_condition=f"{loop_expr} has another iteration",
                condition_result=True,
                target_line=loop_body_target,
                target_hint=TargetHint(
                    line=loop_body_target,
                    expr=loop_expr,
                    polarity=True,
                ),
            )
        ]

        if false_terminates_via is None:
            conditional_targets.append(
                ConditionalTarget(
                    target_condition=f"{loop_expr} has no more iterations",
                    condition_result=False,
                    target_line=false_target,
                    target_hint=TargetHint(
                        line=false_target,
                        expr=loop_expr,
                        polarity=False,
                    ),
                )
            )
        else:
            conditional_targets.append(
                ConditionalTarget(
                    target_condition=f"{loop_expr} has no more iterations",
                    condition_result=False,
                    is_terminal=True,
                    terminates_via=false_terminates_via,
                    target_hint=TargetHint(
                        line=getattr(stmt, "lineno", None),
                        expr=loop_expr,
                        polarity=False,
                    ),
                )
            )

        return [
            DecomposerResult(
                description=f"evaluates whether {loop_expr} has another iteration",
                conditional_targets=conditional_targets,
            )
        ]


class AsyncForDecomposer(ForDecomposer):
    pass


class WhileDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.While, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.test)]

    @classmethod
    def semantic_eis(cls, stmt: ast.While, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        condition = ast.unparse(stmt.test)
        loop_body = body_lines(stmt.body)
        loop_body_target = stmt.body[0].lineno if stmt.body else None
        exit_target = stmt.orelse[0].lineno if stmt.orelse else select_target_from_chain(
            context.next_stmt_lines,
            loop_body,
        )
        return [
            DecomposerResult(
                description=f"{condition} initially false",
                statement_outcome=StatementOutcome(
                    outcome=f"{condition} initially false",
                    target_line=exit_target,
                    skips_lines=loop_body,
                ),
            ),
            DecomposerResult(
                description=f"{condition} initially true",
                statement_outcome=StatementOutcome(
                    outcome=f"{condition} initially true",
                    target_line=loop_body_target,
                ),
            ),
        ]


class TryDecomposer(ControlOwnerDecomposer):
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Try,
            source_lines: list[str],
            context: DecompositionContext
    ) -> list[DecomposerResult]:
        body_target = stmt.body[0].lineno if stmt.body else None

        except_lines: list[int] = []
        for handler in stmt.handlers:
            except_lines.extend(body_lines(handler.body))

        results = [
            DecomposerResult(
                description="enter the try body",
                statement_outcome=StatementOutcome(
                    outcome="enter the try body",
                    target_line=body_target,
                    skips_lines=except_lines,
                ),
            )
        ]

        if stmt.orelse:
            results.append(
                DecomposerResult(
                    description="try succeeds without exception → enters else block",
                    statement_outcome=StatementOutcome(
                        outcome="try succeeds without exception → enters else block",
                        target_line=stmt.orelse[0].lineno,
                        skips_lines=except_lines,
                    ),
                )
            )

        if stmt.finalbody:
            results.append(
                DecomposerResult(
                    description="protected flow completes → enters finally block",
                    statement_outcome=StatementOutcome(
                        outcome="protected flow completes → enters finally block",
                        target_line=stmt.finalbody[0].lineno,
                        skips_lines=except_lines,
                    ),
                )
            )

        return results


class WithDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.With, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(item.context_expr) for item in stmt.items]

    @classmethod
    def semantic_eis(cls, stmt: ast.With, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        contexts = ", ".join(ast.unparse(item.context_expr) for item in stmt.items)
        body_target = stmt.body[0].lineno if stmt.body else None
        return [
            DecomposerResult(
                description=f"with {contexts}: enters successfully",
                statement_outcome=StatementOutcome(
                    outcome=f"with {contexts}: enters successfully",
                    target_line=body_target,
                ),
            ),
            DecomposerResult(
                description=f"with {contexts}: raises exception on entry",
                disruptive_outcomes=[
                    DisruptiveOutcome(
                        outcome=f"with {contexts}: raises exception on entry",
                        is_terminal=True,
                        terminates_via="exception",
                    )
                ],
            ),
        ]


class AsyncWithDecomposer(WithDecomposer):
    pass
