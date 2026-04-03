from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from typing_extensions import Self

from pybastion_common.models import ConditionalTarget, StatementOutcome, DisruptiveOutcome


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
