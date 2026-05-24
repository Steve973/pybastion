from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum


class OwnerKind(StrEnum):
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
