from __future__ import annotations

import ast

from pybastion_common.models import DisruptiveOutcome, StatementOutcome
from .base import SingularStatementDecomposer
from .common import extract_all_operations, semantic
from .decomp_types import DecompositionContext, DecomposerResult, StatementPart


class AssignDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Assign, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)]

    @classmethod
    def semantic_eis(cls, stmt: ast.Assign, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


class AugAssignDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.AugAssign, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)]

    @classmethod
    def semantic_eis(cls, stmt: ast.AugAssign, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


class AnnAssignDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.AnnAssign, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)] if stmt.value else []

    @classmethod
    def semantic_eis(cls, stmt: ast.AnnAssign, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


class ExprDecomposer(SingularStatementDecomposer):
    @classmethod
    def decompose(cls, stmt: ast.Expr, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            return []
        return super().decompose(stmt, source_lines, context)

    @classmethod
    def statement_parts(cls, stmt: ast.Expr, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)]

    @classmethod
    def semantic_eis(cls, stmt: ast.Expr, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        if extract_all_operations(stmt.value):
            return []
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]


class ReturnDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Return, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.value)] if stmt.value else []

    @classmethod
    def semantic_eis(cls, stmt: ast.Return, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        ret_val = ast.unparse(stmt.value) if stmt.value else "None"
        return [
            DecomposerResult(
                description=f"returns {ret_val}",
                statement_outcome=StatementOutcome(
                    outcome=f"returns {ret_val}",
                    is_terminal=True,
                    terminates_via="return",
                ),
            )
        ]


class RaiseDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Raise, context: DecompositionContext) -> list[StatementPart]:
        return [StatementPart(stmt.exc)] if stmt.exc else []

    @classmethod
    def semantic_eis(cls, stmt: ast.Raise, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        exc = ast.unparse(stmt.exc) if stmt.exc else "exception"
        return [
            DecomposerResult(
                description=f"raises {exc}",
                disruptive_outcomes=[
                    DisruptiveOutcome(
                        outcome=f"raises {exc}",
                        is_terminal=True,
                        terminates_via="raise",
                    )
                ],
            )
        ]


class BreakDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Break, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        loop_owner = context.nearest_loop
        return [semantic("break", loop_owner.next_stmt_lines if loop_owner else context.next_stmt_lines)]


class ContinueDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Continue, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        loop_owner = context.nearest_loop
        return [semantic("continue", loop_owner.next_stmt_lines if loop_owner else context.next_stmt_lines)]


class PassDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Pass, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        return [semantic("pass", context.next_stmt_lines)]


class DeleteDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Delete, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        return [semantic(f"del {', '.join(ast.unparse(t) for t in stmt.targets)}", context.next_stmt_lines)]


class ImportDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Import, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        return [semantic(f"import {', '.join(alias.name for alias in stmt.names)}", context.next_stmt_lines)]


class ImportFromDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.ImportFrom, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        module = stmt.module or ""
        names = ", ".join(alias.name for alias in stmt.names)
        return [semantic(f"from {module} import {names}", context.next_stmt_lines)]


class AssertDecomposer(SingularStatementDecomposer):
    @classmethod
    def statement_parts(cls, stmt: ast.Assert, context: DecompositionContext) -> list[StatementPart]:
        parts = [StatementPart(stmt.test)]
        if stmt.msg:
            parts.append(StatementPart(stmt.msg))
        return parts

    @classmethod
    def semantic_eis(cls, stmt: ast.Assert, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        condition = ast.unparse(stmt.test)
        return [
            semantic(f"{condition} is true", context.next_stmt_lines),
            DecomposerResult(
                description=f"{condition} is false → raises AssertionError",
                disruptive_outcomes=[
                    DisruptiveOutcome(
                        outcome=f"{condition} is false → raises AssertionError",
                        is_terminal=True,
                        terminates_via="raise",
                    )
                ],
            ),
        ]


class GlobalDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Global, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        return [semantic(f"global {', '.join(stmt.names)}", context.next_stmt_lines)]


class NonlocalDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.Nonlocal, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        return [semantic(f"nonlocal {', '.join(stmt.names)}", context.next_stmt_lines)]


class DefaultDecomposer(SingularStatementDecomposer):
    @classmethod
    def semantic_eis(cls, stmt: ast.stmt, source_lines: list[str], context: DecompositionContext) -> list[
        DecomposerResult]:
        text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(text, context.next_stmt_lines)]
