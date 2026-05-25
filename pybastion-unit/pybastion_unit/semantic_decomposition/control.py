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
from .decomp_models import (
    ControlOwnerKind,
    ControlRegion,
    ControlRegionKind,
    ControlRoute,
    ControlRouteKind,
    ControlStatementDecomposition,
    DecomposerResult,
    ExecutionStatementDecomposition,
    ExitKind,
    PostExecutionBindingMode,
    PostExecutionBindingScope,
    PostExecutionMechanismKind,
    PostExecutionPolicy,
    PostExecutionTriggerEvent,
)
from .decomp_types import DecompositionContext, StatementPart
from .expressions import enumerate_truth_conditions


# return/raise are region-owned transfers. Do not recursively scan nested
# control owners here; nested owners emit their own direct return/raise routes.
def _direct_region_disruptive_exit_routes(
    *,
    owner_id: str,
    source_region_id: str,
    body: list[ast.stmt],
) -> list[ControlRoute]:
    routes: list[ControlRoute] = []

    for index, child in enumerate(body):
        match child:
            case ast.Return():
                routes.append(
                    ControlRoute(
                        id=_route_id(owner_id, f"return:{child.lineno}:{index}"),
                        owner_id=owner_id,
                        kind=ControlRouteKind.FUNCTION_RETURN,
                        source_region_id=source_region_id,
                        target_line=child.lineno,
                        exit_kind=ExitKind.RETURN,
                        synthetic=True,
                    )
                )

            case ast.Raise():
                routes.append(
                    ControlRoute(
                        id=_route_id(owner_id, f"raise:{child.lineno}:{index}"),
                        owner_id=owner_id,
                        kind=ControlRouteKind.RAISE,
                        source_region_id=source_region_id,
                        target_line=child.lineno,
                        exit_kind=ExitKind.RAISE,
                        synthetic=True,
                    )
                )

    return routes


# break/continue are loop-owned transfers. The route owner is the enclosing
# loop, but the source_region_id is the innermost modeled region containing the
# transfer statement. Nested loops are boundaries because break/continue bind to
# the nearest loop.
def _collect_loop_transfer_routes(
    *,
    owner_id: str,
    body: list[ast.stmt],
    source_region_id: str,
    decision_region_id: str,
    decision_line: int,
    loop_exit_line: int | None,
) -> list[ControlRoute]:
    routes: list[ControlRoute] = []

    def append_continue(stmt: ast.Continue, current_source_region_id: str) -> None:
        routes.append(
            ControlRoute(
                id=_route_id(owner_id, f"continue:{stmt.lineno}:{len(routes)}"),
                owner_id=owner_id,
                kind=ControlRouteKind.LOOP_CONTINUE,
                source_region_id=current_source_region_id,
                target_region_id=decision_region_id,
                target_line=decision_line,
                exit_kind=ExitKind.CONTINUE,
                synthetic=True,
            )
        )

    def append_break(stmt: ast.Break, current_source_region_id: str) -> None:
        routes.append(
            ControlRoute(
                id=_route_id(owner_id, f"break:{stmt.lineno}:{len(routes)}"),
                owner_id=owner_id,
                kind=ControlRouteKind.LOOP_BREAK,
                source_region_id=current_source_region_id,
                target_line=loop_exit_line,
                exit_kind=ExitKind.BREAK,
                synthetic=True,
            )
        )

    def visit_block(statements: list[ast.stmt], current_source_region_id: str) -> None:
        for child in statements:
            visit_stmt(child, current_source_region_id)

    def visit_stmt(stmt: ast.stmt, current_source_region_id: str) -> None:
        match stmt:
            case ast.Continue():
                append_continue(stmt, current_source_region_id)

            case ast.Break():
                append_break(stmt, current_source_region_id)

            case ast.For() | ast.AsyncFor() | ast.While():
                return

            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                return

            case ast.If():
                if_owner_id = _if_control_id(stmt)
                true_region_id = _region_id(if_owner_id, "true_body")
                false_region_id = _region_id(if_owner_id, "false_body")

                visit_block(stmt.body, true_region_id)

                if stmt.orelse:
                    visit_block(stmt.orelse, false_region_id)

            case ast.Match():
                match_owner_id = _match_control_id(stmt)

                for index, case in enumerate(stmt.cases, start=1):
                    case_region_id = _region_id(
                        match_owner_id,
                        f"case_body:{index}",
                    )
                    visit_block(case.body, case_region_id)

            case ast.Try():
                try_owner_id = _try_control_id(stmt)

                protected_region_id = _region_id(try_owner_id, "protected_body")
                visit_block(stmt.body, protected_region_id)

                for index, handler in enumerate(stmt.handlers):
                    handler_region_id = _region_id(
                        try_owner_id,
                        f"exception_handler:{index}",
                    )
                    visit_block(handler.body, handler_region_id)

                if stmt.orelse:
                    else_region_id = _region_id(
                        try_owner_id,
                        "success_continuation",
                    )
                    visit_block(stmt.orelse, else_region_id)

                if stmt.finalbody:
                    finally_region_id = _region_id(
                        try_owner_id,
                        "post_execution",
                    )
                    visit_block(stmt.finalbody, finally_region_id)

            case ast.With() | ast.AsyncWith():
                with_owner_id = _with_control_id(stmt)
                body_region_id = _region_id(with_owner_id, "body")
                visit_block(stmt.body, body_region_id)

            case _:
                return

    visit_block(body, source_region_id)

    return routes


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


def _control_id(kind: str, stmt: ast.AST) -> str:
    return f"{kind}:{getattr(stmt, 'lineno', 'unknown')}"


def _if_control_id(stmt: ast.If) -> str:
    return _control_id("if", stmt)


def _match_control_id(stmt: ast.Match) -> str:
    return _control_id("match", stmt)


def _for_control_id(stmt: ast.For | ast.AsyncFor) -> str:
    return _control_id("async_for" if isinstance(stmt, ast.AsyncFor) else "for", stmt)


def _while_control_id(stmt: ast.While) -> str:
    return _control_id("while", stmt)


def _with_control_id(stmt: ast.With | ast.AsyncWith) -> str:
    return _control_id(
        "async_with" if isinstance(stmt, ast.AsyncWith) else "with", stmt
    )


def _loop_owner_kind(stmt: ast.For | ast.AsyncFor) -> ControlOwnerKind:
    return (
        ControlOwnerKind.ASYNC_FOR
        if isinstance(stmt, ast.AsyncFor)
        else ControlOwnerKind.FOR
    )


def _with_owner_kind(stmt: ast.With | ast.AsyncWith) -> ControlOwnerKind:
    return (
        ControlOwnerKind.ASYNC_WITH
        if isinstance(stmt, ast.AsyncWith)
        else ControlOwnerKind.WITH
    )


def _with_mechanism_kind(stmt: ast.With | ast.AsyncWith) -> PostExecutionMechanismKind:
    return (
        PostExecutionMechanismKind.ASYNC_CONTEXT_MANAGER_EXIT
        if isinstance(stmt, ast.AsyncWith)
        else PostExecutionMechanismKind.CONTEXT_MANAGER_EXIT
    )


def _with_binding_scope(stmt: ast.With | ast.AsyncWith) -> PostExecutionBindingScope:
    return (
        PostExecutionBindingScope.ASYNC_WITH_BODY
        if isinstance(stmt, ast.AsyncWith)
        else PostExecutionBindingScope.WITH_BODY
    )


def _continuation_target(
    context: DecompositionContext,
    skipped_lines: list[int],
) -> int | None:
    return select_target_from_chain(context.next_stmt_lines, skipped_lines)


def _body_has_disruptive_terminal(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue))
        for stmt in body
    )


class IfDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(
        cls, stmt: ast.If, context: DecompositionContext
    ) -> list[StatementPart]:
        return [StatementPart(stmt.test)]

    @classmethod
    def control_decomposition(
        cls, stmt: ast.If, context: DecompositionContext
    ) -> ControlStatementDecomposition:
        owner_id = _if_control_id(stmt)

        condition_region_id = _region_id(owner_id, "condition")
        true_region_id = _region_id(owner_id, "true_body")
        false_region_id = _region_id(owner_id, "false_body")

        true_target = _stmt_list_start(stmt.body)
        false_body_target = _stmt_list_start(stmt.orelse)
        skipped_lines = body_lines(stmt.body) + body_lines(stmt.orelse)
        continuation_target = _continuation_target(context, skipped_lines)

        condition = ast.unparse(stmt.test)

        regions: list[ControlRegion] = [
            ControlRegion(
                id=condition_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.CONDITION,
                start_line=stmt.lineno,
                end_line=stmt.lineno,
                source_construct="if",
                metadata={"condition": condition},
            ),
            ControlRegion(
                id=true_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.TRUE_BODY,
                start_line=_stmt_list_start(stmt.body),
                end_line=_stmt_list_end(stmt.body),
                source_construct="if",
            ),
        ]

        routes: list[ControlRoute] = [
            ControlRoute(
                id=_route_id(owner_id, "enter_condition"),
                owner_id=owner_id,
                kind=ControlRouteKind.ENTER,
                target_region_id=condition_region_id,
                target_line=stmt.lineno,
            ),
            ControlRoute(
                id=_route_id(owner_id, "condition_true"),
                owner_id=owner_id,
                kind=ControlRouteKind.CONDITIONAL_TRUE,
                source_region_id=condition_region_id,
                target_region_id=true_region_id,
                target_line=true_target,
                condition=condition,
                condition_result=True,
            ),
        ]

        if stmt.orelse:
            regions.append(
                ControlRegion(
                    id=false_region_id,
                    owner_id=owner_id,
                    kind=ControlRegionKind.FALSE_BODY,
                    start_line=_stmt_list_start(stmt.orelse),
                    end_line=_stmt_list_end(stmt.orelse),
                    source_construct="else",
                )
            )
            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "condition_false"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.CONDITIONAL_FALSE,
                    source_region_id=condition_region_id,
                    target_region_id=false_region_id,
                    target_line=false_body_target,
                    condition=condition,
                    condition_result=False,
                )
            )

            if not _body_has_disruptive_terminal(stmt.orelse):
                routes.append(
                    ControlRoute(
                        id=_route_id(owner_id, "false_body_completion"),
                        owner_id=owner_id,
                        kind=ControlRouteKind.NORMAL_COMPLETION,
                        source_region_id=false_region_id,
                        target_line=continuation_target,
                        exit_kind=ExitKind.NORMAL,
                    )
                )
        else:
            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "condition_false_fallthrough"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.CONDITIONAL_FALSE,
                    source_region_id=condition_region_id,
                    target_line=continuation_target,
                    condition=condition,
                    condition_result=False,
                    exit_kind=(
                        ExitKind.NORMAL
                        if continuation_target is not None
                        else ExitKind.IMPLICIT_RETURN
                    ),
                    implicit=continuation_target is None,
                )
            )

        if not _body_has_disruptive_terminal(stmt.body):
            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "true_body_completion"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.NORMAL_COMPLETION,
                    source_region_id=true_region_id,
                    target_line=continuation_target,
                    exit_kind=ExitKind.NORMAL,
                )
            )

        routes.extend(
            _direct_region_disruptive_exit_routes(
                owner_id=owner_id,
                source_region_id=true_region_id,
                body=stmt.body,
            )
        )

        if stmt.orelse:
            routes.extend(
                _direct_region_disruptive_exit_routes(
                    owner_id=owner_id,
                    source_region_id=false_region_id,
                    body=stmt.orelse,
                )
            )

        return ControlStatementDecomposition(
            description=f"if statement at line {stmt.lineno}",
            owner_kind=ControlOwnerKind.IF,
            control_id=owner_id,
            line=stmt.lineno,
            end_line=stmt.end_lineno,
            regions=regions,
            routes=routes,
            policies=[],
            source_construct="if",
        )

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
                false_terminates_via = (
                    None if false_target is not None else "implicit-return"
                )
            else:
                false_target = None
                false_terminates_via = "implicit-return"

            condition = ast.unparse(current.test)

            results.append(
                DecomposerResult(
                    decomposition=ExecutionStatementDecomposition(
                        description=f"evaluates {condition}",
                        conditional_targets=_build_if_conditional_targets(
                            current.test,
                            true_target=true_target,
                            false_target=false_target,
                            false_terminates_via=false_terminates_via,
                        ),
                    )
                )
            )

            if (
                current.orelse
                and len(current.orelse) == 1
                and isinstance(current.orelse[0], ast.If)
            ):
                current = current.orelse[0]
            else:
                current = None

        results.append(
            DecomposerResult(decomposition=cls.control_decomposition(stmt, context))
        )
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
    def control_decomposition(
        cls, stmt: ast.Match, context: DecompositionContext
    ) -> ControlStatementDecomposition:
        owner_id = _match_control_id(stmt)
        condition_region_id = _region_id(owner_id, "subject")

        all_case_lines: list[int] = []
        for case in stmt.cases:
            all_case_lines.extend(body_lines(case.body))

        fallthrough_target = _continuation_target(context, all_case_lines)

        regions: list[ControlRegion] = [
            ControlRegion(
                id=condition_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.CONDITION,
                start_line=stmt.lineno,
                end_line=stmt.lineno,
                source_construct="match",
                metadata={"subject": ast.unparse(stmt.subject)},
            )
        ]

        routes: list[ControlRoute] = [
            ControlRoute(
                id=_route_id(owner_id, "enter_subject"),
                owner_id=owner_id,
                kind=ControlRouteKind.ENTER,
                target_region_id=condition_region_id,
                target_line=stmt.lineno,
            )
        ]

        prior_failures: list[str] = []
        has_unconditional_case = False

        for index, case in enumerate(stmt.cases, start=1):
            case_region_id = _region_id(owner_id, f"case_body:{index}")
            case_label = cls._case_label(index, case)
            case_target = _stmt_list_start(case.body) or fallthrough_target
            success_condition = cls._join_and([*prior_failures, case_label])

            regions.append(
                ControlRegion(
                    id=case_region_id,
                    owner_id=owner_id,
                    kind=ControlRegionKind.CASE_BODY,
                    start_line=_stmt_list_start(case.body),
                    end_line=_stmt_list_end(case.body),
                    ordinal=index,
                    source_construct="case",
                    metadata={"case_label": case_label},
                )
            )

            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, f"match_case:{index}"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.MATCH_CASE,
                    source_region_id=condition_region_id,
                    target_region_id=case_region_id,
                    target_line=case_target,
                    condition=success_condition,
                    condition_result=True,
                    metadata={"case_label": case_label},
                )
            )

            if not _body_has_disruptive_terminal(case.body):
                routes.append(
                    ControlRoute(
                        id=_route_id(owner_id, f"case_body_completion:{index}"),
                        owner_id=owner_id,
                        kind=ControlRouteKind.NORMAL_COMPLETION,
                        source_region_id=case_region_id,
                        target_line=fallthrough_target,
                        exit_kind=ExitKind.NORMAL,
                    )
                )

            routes.extend(
                _direct_region_disruptive_exit_routes(
                    owner_id=owner_id,
                    source_region_id=case_region_id,
                    body=case.body,
                )
            )

            if cls._is_unconditional_wildcard(case):
                has_unconditional_case = True
                break

            prior_failures.append(f"not ({case_label})")

        if not has_unconditional_case:
            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "match_fallthrough"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.MATCH_FALLTHROUGH,
                    source_region_id=condition_region_id,
                    target_line=fallthrough_target,
                    condition=(
                        cls._join_and(prior_failures)
                        if prior_failures
                        else "no match cases"
                    ),
                    condition_result=False,
                    exit_kind=(
                        ExitKind.NORMAL
                        if fallthrough_target is not None
                        else ExitKind.IMPLICIT_RETURN
                    ),
                    implicit=fallthrough_target is None,
                )
            )

        return ControlStatementDecomposition(
            description=f"match statement at line {stmt.lineno}",
            owner_kind=ControlOwnerKind.MATCH,
            control_id=owner_id,
            line=stmt.lineno,
            end_line=stmt.end_lineno,
            regions=regions,
            routes=routes,
            policies=[],
            source_construct="match",
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

            case_target = (
                case.body[0].lineno
                if case.body
                else select_target_from_chain(
                    context.next_stmt_lines,
                    case_lines,
                )
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

            no_match_condition = (
                cls._join_and(prior_failures) if prior_failures else "no match cases"
            )

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
                decomposition=ExecutionStatementDecomposition(
                    description=f"evaluates match {subject}",
                    conditional_targets=conditional_targets,
                )
            ),
            DecomposerResult(decomposition=cls.control_decomposition(stmt, context)),
        ]


class ForDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(
        cls, stmt: ast.For, context: DecompositionContext
    ) -> list[StatementPart]:
        return [StatementPart(stmt.iter)]

    @classmethod
    def control_decomposition(
        cls, stmt: ast.For | ast.AsyncFor, context: DecompositionContext
    ) -> ControlStatementDecomposition:
        owner_id = _for_control_id(stmt)
        source_construct = "async for" if isinstance(stmt, ast.AsyncFor) else "for"

        condition_region_id = _region_id(owner_id, "iterator")
        body_region_id = _region_id(owner_id, "loop_body")
        else_region_id = _region_id(owner_id, "loop_else")

        loop_body_lines = body_lines(stmt.body)
        loop_else_lines = body_lines(stmt.orelse)
        continuation_target = _continuation_target(
            context, loop_body_lines + loop_else_lines
        )
        body_target = _stmt_list_start(stmt.body)
        else_target = _stmt_list_start(stmt.orelse)

        target = ast.unparse(stmt.target)
        iter_expr = ast.unparse(stmt.iter)
        loop_expr = f"{source_construct} {target} in {iter_expr}"

        regions: list[ControlRegion] = [
            ControlRegion(
                id=condition_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.CONDITION,
                start_line=stmt.lineno,
                end_line=stmt.lineno,
                source_construct=source_construct,
                metadata={"target": target, "iter": iter_expr},
            ),
            ControlRegion(
                id=body_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.LOOP_BODY,
                start_line=_stmt_list_start(stmt.body),
                end_line=_stmt_list_end(stmt.body),
                source_construct=source_construct,
            ),
        ]

        exhausted_target_region_id: str | None = None
        exhausted_target_line = continuation_target
        if stmt.orelse:
            exhausted_target_region_id = else_region_id
            exhausted_target_line = else_target
            regions.append(
                ControlRegion(
                    id=else_region_id,
                    owner_id=owner_id,
                    kind=ControlRegionKind.LOOP_ELSE,
                    start_line=_stmt_list_start(stmt.orelse),
                    end_line=_stmt_list_end(stmt.orelse),
                    source_construct="else",
                )
            )

        routes: list[ControlRoute] = [
            ControlRoute(
                id=_route_id(owner_id, "enter_iterator"),
                owner_id=owner_id,
                kind=ControlRouteKind.ENTER,
                target_region_id=condition_region_id,
                target_line=stmt.lineno,
            ),
            ControlRoute(
                id=_route_id(owner_id, "loop_iteration"),
                owner_id=owner_id,
                kind=ControlRouteKind.LOOP_ITERATION,
                source_region_id=condition_region_id,
                target_region_id=body_region_id,
                target_line=body_target,
                condition=f"{loop_expr} has another iteration",
                condition_result=True,
            ),
            ControlRoute(
                id=_route_id(owner_id, "loop_exhausted"),
                owner_id=owner_id,
                kind=ControlRouteKind.LOOP_EXHAUSTED,
                source_region_id=condition_region_id,
                target_region_id=exhausted_target_region_id,
                target_line=exhausted_target_line,
                condition=f"{loop_expr} has no more iterations",
                condition_result=False,
                exit_kind=(
                    ExitKind.NORMAL
                    if exhausted_target_line is not None
                    else ExitKind.IMPLICIT_RETURN
                ),
                implicit=exhausted_target_line is None,
            ),
        ]

        if stmt.orelse:
            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "else_completion"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.NORMAL_COMPLETION,
                    source_region_id=else_region_id,
                    target_line=continuation_target,
                    exit_kind=ExitKind.NORMAL,
                )
            )

        routes.extend(
            _collect_loop_transfer_routes(
                owner_id=owner_id,
                body=stmt.body,
                source_region_id=body_region_id,
                decision_region_id=condition_region_id,
                decision_line=stmt.lineno,
                loop_exit_line=continuation_target,
            )
        )

        routes.extend(
            _direct_region_disruptive_exit_routes(
                owner_id=owner_id,
                source_region_id=body_region_id,
                body=stmt.body,
            )
        )

        if not _body_has_disruptive_terminal(stmt.body):
            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "body_next_iteration"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.LOOP_ITERATION,
                    source_region_id=body_region_id,
                    target_region_id=condition_region_id,
                    target_line=stmt.lineno,
                    exit_kind=ExitKind.NORMAL,
                    synthetic=True,
                )
            )

        return ControlStatementDecomposition(
            description=f"{source_construct} statement at line {stmt.lineno}",
            owner_kind=_loop_owner_kind(stmt),
            control_id=owner_id,
            line=stmt.lineno,
            end_line=stmt.end_lineno,
            regions=regions,
            routes=routes,
            policies=[],
            source_construct=source_construct,
        )

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
            false_terminates_via = (
                None if false_target is not None else "implicit-return"
            )
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
                decomposition=ExecutionStatementDecomposition(
                    description=f"evaluates whether {loop_expr} has another iteration",
                    conditional_targets=conditional_targets,
                )
            ),
            DecomposerResult(decomposition=cls.control_decomposition(stmt, context)),
        ]


class AsyncForDecomposer(ForDecomposer):
    pass


class WhileDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(
        cls, stmt: ast.While, context: DecompositionContext
    ) -> list[StatementPart]:
        return [StatementPart(stmt.test)]

    @classmethod
    def control_decomposition(
        cls, stmt: ast.While, context: DecompositionContext
    ) -> ControlStatementDecomposition:
        owner_id = _while_control_id(stmt)

        condition_region_id = _region_id(owner_id, "condition")
        body_region_id = _region_id(owner_id, "loop_body")
        else_region_id = _region_id(owner_id, "loop_else")

        loop_body_lines = body_lines(stmt.body)
        loop_else_lines = body_lines(stmt.orelse)
        continuation_target = _continuation_target(
            context, loop_body_lines + loop_else_lines
        )

        condition = ast.unparse(stmt.test)
        body_target = _stmt_list_start(stmt.body)
        exhausted_target_region_id: str | None = None
        exhausted_target_line = continuation_target

        regions: list[ControlRegion] = [
            ControlRegion(
                id=condition_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.CONDITION,
                start_line=stmt.lineno,
                end_line=stmt.lineno,
                source_construct="while",
                metadata={"condition": condition},
            ),
            ControlRegion(
                id=body_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.LOOP_BODY,
                start_line=_stmt_list_start(stmt.body),
                end_line=_stmt_list_end(stmt.body),
                source_construct="while",
            ),
        ]

        if stmt.orelse:
            exhausted_target_region_id = else_region_id
            exhausted_target_line = _stmt_list_start(stmt.orelse)
            regions.append(
                ControlRegion(
                    id=else_region_id,
                    owner_id=owner_id,
                    kind=ControlRegionKind.LOOP_ELSE,
                    start_line=_stmt_list_start(stmt.orelse),
                    end_line=_stmt_list_end(stmt.orelse),
                    source_construct="else",
                )
            )

        routes: list[ControlRoute] = [
            ControlRoute(
                id=_route_id(owner_id, "enter_condition"),
                owner_id=owner_id,
                kind=ControlRouteKind.ENTER,
                target_region_id=condition_region_id,
                target_line=stmt.lineno,
            ),
            ControlRoute(
                id=_route_id(owner_id, "condition_true"),
                owner_id=owner_id,
                kind=ControlRouteKind.LOOP_ITERATION,
                source_region_id=condition_region_id,
                target_region_id=body_region_id,
                target_line=body_target,
                condition=condition,
                condition_result=True,
            ),
            ControlRoute(
                id=_route_id(owner_id, "condition_false"),
                owner_id=owner_id,
                kind=ControlRouteKind.LOOP_EXHAUSTED,
                source_region_id=condition_region_id,
                target_region_id=exhausted_target_region_id,
                target_line=exhausted_target_line,
                condition=condition,
                condition_result=False,
                exit_kind=(
                    ExitKind.NORMAL
                    if exhausted_target_line is not None
                    else ExitKind.IMPLICIT_RETURN
                ),
                implicit=exhausted_target_line is None,
            ),
        ]

        if stmt.orelse:
            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "else_completion"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.NORMAL_COMPLETION,
                    source_region_id=else_region_id,
                    target_line=continuation_target,
                    exit_kind=ExitKind.NORMAL,
                )
            )

        routes.extend(
            _collect_loop_transfer_routes(
                owner_id=owner_id,
                body=stmt.body,
                source_region_id=body_region_id,
                decision_region_id=condition_region_id,
                decision_line=stmt.lineno,
                loop_exit_line=continuation_target,
            )
        )

        routes.extend(
            _direct_region_disruptive_exit_routes(
                owner_id=owner_id,
                source_region_id=body_region_id,
                body=stmt.body,
            )
        )

        if not _body_has_disruptive_terminal(stmt.body):
            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "body_next_iteration"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.LOOP_ITERATION,
                    source_region_id=body_region_id,
                    target_region_id=condition_region_id,
                    target_line=stmt.lineno,
                    exit_kind=ExitKind.NORMAL,
                    synthetic=True,
                )
            )

        return ControlStatementDecomposition(
            description=f"while statement at line {stmt.lineno}",
            owner_kind=ControlOwnerKind.WHILE,
            control_id=owner_id,
            line=stmt.lineno,
            end_line=stmt.end_lineno,
            regions=regions,
            routes=routes,
            policies=[],
            source_construct="while",
        )

    @classmethod
    def semantic_eis(
        cls, stmt: ast.While, source_lines: list[str], context: DecompositionContext
    ) -> list[DecomposerResult]:
        condition = ast.unparse(stmt.test)
        loop_body = body_lines(stmt.body)
        loop_body_target = stmt.body[0].lineno if stmt.body else None
        exit_target = (
            stmt.orelse[0].lineno
            if stmt.orelse
            else select_target_from_chain(
                context.next_stmt_lines,
                loop_body,
            )
        )
        return [
            DecomposerResult(
                decomposition=ExecutionStatementDecomposition(
                    description=f"{condition} initially false",
                    statement_outcome=StatementOutcome(
                        outcome=f"{condition} initially false",
                        target_line=exit_target,
                        skips_lines=loop_body,
                    ),
                )
            ),
            DecomposerResult(
                decomposition=ExecutionStatementDecomposition(
                    description=f"{condition} initially true",
                    statement_outcome=StatementOutcome(
                        outcome=f"{condition} initially true",
                        target_line=loop_body_target,
                    ),
                )
            ),
            DecomposerResult(decomposition=cls.control_decomposition(stmt, context)),
        ]


def _try_control_id(stmt: ast.Try) -> str:
    return f"try:{stmt.lineno}"


def _region_id(owner_id: str, suffix: str) -> str:
    return f"{owner_id}:{suffix}"


def _route_id(owner_id: str, suffix: str) -> str:
    return f"{owner_id}:route:{suffix}"


def _stmt_list_start(body: list[ast.stmt]) -> int | None:
    return body[0].lineno if body else None


def _stmt_list_end(body: list[ast.stmt]) -> int | None:
    return body[-1].end_lineno if body else None


class TryDecomposer(ControlOwnerDecomposer):
    @classmethod
    def control_decomposition(
        cls,
        stmt: ast.Try,
        context: DecompositionContext,
    ) -> ControlStatementDecomposition:
        owner_id = _try_control_id(stmt)
        protected_region_id = _region_id(owner_id, "protected_body")
        dispatch_region_id = _region_id(owner_id, "exception_dispatch")
        try_body_lines = body_lines(stmt.body)

        handler_lines: list[int] = []
        for handler in stmt.handlers:
            handler_lines.extend(body_lines(handler.body))

        else_lines = body_lines(stmt.orelse)
        finally_lines = body_lines(stmt.finalbody)

        continuation_target = _continuation_target(
            context,
            try_body_lines + handler_lines + else_lines + finally_lines,
        )

        regions: list[ControlRegion] = [
            ControlRegion(
                id=protected_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.PROTECTED_BODY,
                start_line=_stmt_list_start(stmt.body),
                end_line=_stmt_list_end(stmt.body),
                source_construct="try",
            ),
            ControlRegion(
                id=dispatch_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.EXCEPTION_DISPATCH,
                start_line=stmt.lineno,
                end_line=stmt.lineno,
                source_construct="except",
                synthetic=True,
            ),
        ]

        routes: list[ControlRoute] = [
            ControlRoute(
                id=_route_id(owner_id, "enter_protected_body"),
                owner_id=owner_id,
                kind=ControlRouteKind.ENTER,
                target_region_id=protected_region_id,
                target_line=_stmt_list_start(stmt.body),
            ),
            ControlRoute(
                id=_route_id(owner_id, "protected_exception_dispatch"),
                owner_id=owner_id,
                kind=ControlRouteKind.EXCEPTION_DISPATCH,
                source_region_id=protected_region_id,
                target_region_id=dispatch_region_id,
                target_line=stmt.lineno,
                exit_kind=ExitKind.RAISE,
                synthetic=True,
            ),
        ]

        for index, handler in enumerate(stmt.handlers):
            handler_region_id = _region_id(owner_id, f"exception_handler:{index}")

            if not _body_has_disruptive_terminal(handler.body):
                routes.append(
                    ControlRoute(
                        id=_route_id(owner_id, f"handler_completion:{index}"),
                        owner_id=owner_id,
                        kind=ControlRouteKind.NORMAL_COMPLETION,
                        source_region_id=handler_region_id,
                        target_line=continuation_target,
                        exit_kind=ExitKind.NORMAL,
                    )
                )

        for index, handler in enumerate(stmt.handlers):
            handler_region_id = _region_id(owner_id, f"exception_handler:{index}")
            handler_expr = (
                ast.unparse(handler.type) if handler.type is not None else "bare except"
            )

            regions.append(
                ControlRegion(
                    id=handler_region_id,
                    owner_id=owner_id,
                    kind=ControlRegionKind.EXCEPTION_HANDLER,
                    start_line=_stmt_list_start(handler.body),
                    end_line=_stmt_list_end(handler.body),
                    ordinal=index,
                    source_construct="except",
                    metadata={"handler_expr": handler_expr},
                )
            )

            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, f"handler_match:{index}"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.HANDLER_MATCH,
                    source_region_id=dispatch_region_id,
                    target_region_id=handler_region_id,
                    target_line=_stmt_list_start(handler.body),
                    condition=f"exception matches {handler_expr}",
                    condition_result=True,
                    exit_kind=ExitKind.HANDLED_EXCEPTION,
                    metadata={"handler_expr": handler_expr},
                )
            )

        routes.append(
            ControlRoute(
                id=_route_id(owner_id, "unhandled_exception"),
                owner_id=owner_id,
                kind=ControlRouteKind.UNHANDLED_EXCEPTION,
                source_region_id=dispatch_region_id,
                condition="exception does not match any local handler",
                condition_result=False,
                exit_kind=ExitKind.UNHANDLED_EXCEPTION,
                synthetic=True,
            )
        )

        if stmt.orelse:
            else_region_id = _region_id(owner_id, "success_continuation")

            regions.append(
                ControlRegion(
                    id=else_region_id,
                    owner_id=owner_id,
                    kind=ControlRegionKind.SUCCESS_CONTINUATION,
                    start_line=_stmt_list_start(stmt.orelse),
                    end_line=_stmt_list_end(stmt.orelse),
                    source_construct="else",
                )
            )

            if not _body_has_disruptive_terminal(stmt.body):
                routes.append(
                    ControlRoute(
                        id=_route_id(owner_id, "protected_normal_to_else"),
                        owner_id=owner_id,
                        kind=ControlRouteKind.NORMAL_COMPLETION,
                        source_region_id=protected_region_id,
                        target_region_id=else_region_id,
                        target_line=_stmt_list_start(stmt.orelse),
                        exit_kind=ExitKind.NORMAL,
                    )
                )

            if not _body_has_disruptive_terminal(stmt.orelse):
                routes.append(
                    ControlRoute(
                        id=_route_id(owner_id, "else_completion"),
                        owner_id=owner_id,
                        kind=ControlRouteKind.NORMAL_COMPLETION,
                        source_region_id=else_region_id,
                        target_line=continuation_target,
                        exit_kind=ExitKind.NORMAL,
                    )
                )

        if stmt.finalbody:
            finally_region_id = _region_id(owner_id, "post_execution")

            regions.append(
                ControlRegion(
                    id=finally_region_id,
                    owner_id=owner_id,
                    kind=ControlRegionKind.POST_EXECUTION,
                    start_line=_stmt_list_start(stmt.finalbody),
                    end_line=_stmt_list_end(stmt.finalbody),
                    source_construct="finally",
                )
            )

            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "post_execution_entry"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.POST_EXECUTION_ENTRY,
                    target_region_id=finally_region_id,
                    target_line=_stmt_list_start(stmt.finalbody),
                    preserves_prior_outcome=True,
                )
            )

            routes.append(
                ControlRoute(
                    id=_route_id(owner_id, "resume_after_finally"),
                    owner_id=owner_id,
                    kind=ControlRouteKind.RESUME_PRIOR_OUTCOME,
                    source_region_id=finally_region_id,
                    preserves_prior_outcome=True,
                    synthetic=True,
                )
            )

            policies = [
                PostExecutionPolicy(
                    owner_id=owner_id,
                    mechanism_kind=PostExecutionMechanismKind.TRY_FINALLY,
                    binding_mode=PostExecutionBindingMode.STRUCTURAL,
                    binding_scope=PostExecutionBindingScope.PROTECTED_REGION,
                    trigger_event=PostExecutionTriggerEvent.REGION_EXIT,
                    region_id=finally_region_id,
                    target_region_id=finally_region_id,
                    target_line=_stmt_list_start(stmt.finalbody),
                    applies_to=[
                        ExitKind.NORMAL,
                        ExitKind.RETURN,
                        ExitKind.RAISE,
                        ExitKind.BREAK,
                        ExitKind.CONTINUE,
                        ExitKind.HANDLED_EXCEPTION,
                        ExitKind.UNHANDLED_EXCEPTION,
                        ExitKind.IMPLICIT_RETURN,
                        ExitKind.TERMINAL,
                    ],
                    preserves_prior_outcome=True,
                    source_construct="finally",
                )
            ]
        else:
            policies = []

        routes.extend(
            _direct_region_disruptive_exit_routes(
                owner_id=owner_id,
                source_region_id=protected_region_id,
                body=stmt.body,
            )
        )

        for index, handler in enumerate(stmt.handlers):
            handler_region_id = _region_id(owner_id, f"exception_handler:{index}")

            routes.extend(
                _direct_region_disruptive_exit_routes(
                    owner_id=owner_id,
                    source_region_id=handler_region_id,
                    body=handler.body,
                )
            )

        if stmt.orelse:
            else_region_id = _region_id(owner_id, "success_continuation")

            routes.extend(
                _direct_region_disruptive_exit_routes(
                    owner_id=owner_id,
                    source_region_id=else_region_id,
                    body=stmt.orelse,
                )
            )

        if stmt.finalbody:
            finally_region_id = _region_id(owner_id, "post_execution")

            routes.extend(
                _direct_region_disruptive_exit_routes(
                    owner_id=owner_id,
                    source_region_id=finally_region_id,
                    body=stmt.finalbody,
                )
            )

        return ControlStatementDecomposition(
            description=f"try statement at line {stmt.lineno}",
            owner_kind=ControlOwnerKind.TRY,
            control_id=owner_id,
            line=stmt.lineno,
            end_line=stmt.end_lineno,
            regions=regions,
            routes=routes,
            policies=policies,
            source_construct="try",
        )

    @classmethod
    def semantic_eis(
        cls, stmt: ast.Try, source_lines: list[str], context: DecompositionContext
    ) -> list[DecomposerResult]:
        body_target = stmt.body[0].lineno if stmt.body else None

        except_lines: list[int] = []
        for handler in stmt.handlers:
            except_lines.extend(body_lines(handler.body))

        results = [
            DecomposerResult(
                decomposition=ExecutionStatementDecomposition(
                    description="enter the try body",
                    statement_outcome=StatementOutcome(
                        outcome="enter the try body",
                        target_line=body_target,
                        skips_lines=except_lines,
                    ),
                )
            )
        ]

        if stmt.orelse:
            results.append(
                DecomposerResult(
                    decomposition=ExecutionStatementDecomposition(
                        description="try succeeds without exception → enters else block",
                        statement_outcome=StatementOutcome(
                            outcome="try succeeds without exception → enters else block",
                            target_line=stmt.orelse[0].lineno,
                            skips_lines=except_lines,
                        ),
                    )
                )
            )

        if stmt.finalbody:
            results.append(
                DecomposerResult(
                    decomposition=ExecutionStatementDecomposition(
                        description="protected flow completes → enters finally block",
                        statement_outcome=StatementOutcome(
                            outcome="protected flow completes → enters finally block",
                            target_line=stmt.finalbody[0].lineno,
                            skips_lines=except_lines,
                        ),
                    )
                )
            )

        results.append(
            DecomposerResult(decomposition=cls.control_decomposition(stmt, context))
        )

        return results


class WithDecomposer(ControlOwnerDecomposer):
    @classmethod
    def statement_parts(
        cls, stmt: ast.With, context: DecompositionContext
    ) -> list[StatementPart]:
        return [StatementPart(item.context_expr) for item in stmt.items]

    @classmethod
    def control_decomposition(
        cls, stmt: ast.With | ast.AsyncWith, context: DecompositionContext
    ) -> ControlStatementDecomposition:
        owner_id = _with_control_id(stmt)
        source_construct = "async with" if isinstance(stmt, ast.AsyncWith) else "with"

        body_region_id = _region_id(owner_id, "body")
        post_region_id = _region_id(owner_id, "post_execution")
        continuation_target = _continuation_target(context, body_lines(stmt.body))
        contexts = ", ".join(ast.unparse(item.context_expr) for item in stmt.items)

        regions: list[ControlRegion] = [
            ControlRegion(
                id=body_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.BODY,
                start_line=_stmt_list_start(stmt.body),
                end_line=_stmt_list_end(stmt.body),
                source_construct=source_construct,
                metadata={"contexts": contexts},
            ),
            ControlRegion(
                id=post_region_id,
                owner_id=owner_id,
                kind=ControlRegionKind.POST_EXECUTION,
                start_line=stmt.lineno,
                end_line=stmt.lineno,
                source_construct=f"{source_construct} exit",
                synthetic=True,
            ),
        ]

        routes: list[ControlRoute] = [
            ControlRoute(
                id=_route_id(owner_id, "enter_body"),
                owner_id=owner_id,
                kind=ControlRouteKind.ENTER,
                target_region_id=body_region_id,
                target_line=_stmt_list_start(stmt.body),
            ),
            ControlRoute(
                id=_route_id(owner_id, "post_execution_entry"),
                owner_id=owner_id,
                kind=ControlRouteKind.POST_EXECUTION_ENTRY,
                source_region_id=body_region_id,
                target_region_id=post_region_id,
                target_line=stmt.lineno,
                preserves_prior_outcome=True,
                synthetic=True,
            ),
            ControlRoute(
                id=_route_id(owner_id, "resume_after_exit"),
                owner_id=owner_id,
                kind=ControlRouteKind.RESUME_PRIOR_OUTCOME,
                source_region_id=post_region_id,
                target_line=continuation_target,
                preserves_prior_outcome=True,
                synthetic=True,
            ),
        ]

        policies: list[PostExecutionPolicy] = [
            PostExecutionPolicy(
                owner_id=owner_id,
                mechanism_kind=_with_mechanism_kind(stmt),
                binding_mode=PostExecutionBindingMode.STRUCTURAL,
                binding_scope=_with_binding_scope(stmt),
                trigger_event=PostExecutionTriggerEvent.SCOPE_EXIT,
                region_id=body_region_id,
                target_region_id=post_region_id,
                target_line=stmt.lineno,
                applies_to=[
                    ExitKind.NORMAL,
                    ExitKind.RETURN,
                    ExitKind.RAISE,
                    ExitKind.BREAK,
                    ExitKind.CONTINUE,
                    ExitKind.HANDLED_EXCEPTION,
                    ExitKind.UNHANDLED_EXCEPTION,
                    ExitKind.IMPLICIT_RETURN,
                    ExitKind.TERMINAL,
                ],
                preserves_prior_outcome=True,
                source_construct=source_construct,
                detail=f"invoke context manager exit for {contexts}",
            )
        ]

        routes.extend(
            _direct_region_disruptive_exit_routes(
                owner_id=owner_id,
                source_region_id=body_region_id,
                body=stmt.body,
            )
        )

        return ControlStatementDecomposition(
            description=f"{source_construct} statement at line {stmt.lineno}",
            owner_kind=_with_owner_kind(stmt),
            control_id=owner_id,
            line=stmt.lineno,
            end_line=stmt.end_lineno,
            regions=regions,
            routes=routes,
            policies=policies,
            source_construct=source_construct,
        )

    @classmethod
    def semantic_eis(
        cls, stmt: ast.With, source_lines: list[str], context: DecompositionContext
    ) -> list[DecomposerResult]:
        contexts = ", ".join(ast.unparse(item.context_expr) for item in stmt.items)
        body_target = stmt.body[0].lineno if stmt.body else None
        return [
            DecomposerResult(
                decomposition=ExecutionStatementDecomposition(
                    description=f"with {contexts}: enters successfully",
                    statement_outcome=StatementOutcome(
                        outcome=f"with {contexts}: enters successfully",
                        target_line=body_target,
                    ),
                )
            ),
            DecomposerResult(
                decomposition=ExecutionStatementDecomposition(
                    description=f"with {contexts}: raises exception on entry",
                    disruptive_outcomes=[
                        DisruptiveOutcome(
                            outcome=f"with {contexts}: raises exception on entry",
                            is_terminal=True,
                            terminates_via="exception",
                        )
                    ],
                )
            ),
            DecomposerResult(decomposition=cls.control_decomposition(stmt, context)),
        ]


class AsyncWithDecomposer(WithDecomposer):
    pass
