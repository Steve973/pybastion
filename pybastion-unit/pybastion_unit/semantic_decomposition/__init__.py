from __future__ import annotations

import ast

from .base import StatementDecomposer
from .control import (
    AsyncForDecomposer,
    AsyncWithDecomposer,
    ForDecomposer,
    IfDecomposer,
    MatchDecomposer,
    TryDecomposer,
    WhileDecomposer,
    WithDecomposer,
)
from .decomp_types import ControlOwner, DecompositionContext, DecomposerResult, OwnerKind
from .simple import (
    AnnAssignDecomposer,
    AssertDecomposer,
    AssignDecomposer,
    AugAssignDecomposer,
    BreakDecomposer,
    ContinueDecomposer,
    DefaultDecomposer,
    DeleteDecomposer,
    ExprDecomposer,
    GlobalDecomposer,
    ImportDecomposer,
    ImportFromDecomposer,
    NonlocalDecomposer,
    PassDecomposer,
    RaiseDecomposer,
    ReturnDecomposer,
)


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


__all__ = [
    "ControlOwner",
    "DecompositionContext",
    "DecomposerResult",
    "OwnerKind",
    "decompose_statement",
]
