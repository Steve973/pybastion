from __future__ import annotations

import ast

from .expressions import decompose_expression_node
from .decomp_types import DecompositionContext, DecomposerResult, StatementPart


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


class SingularStatementDecomposer(StatementDecomposer):
    pass
