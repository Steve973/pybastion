from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pybastion_common.models import (
    ConditionalTarget,
    DisruptiveOutcome,
    StatementOutcome,
)


def _to_dict_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_to_dict_value(item) for item in value]
    if isinstance(value, tuple):
        return [_to_dict_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_dict_value(item) for key, item in value.items()}
    return value


def _add_if_present(result: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        result[key] = _to_dict_value(value)


def _add_if_true(result: dict[str, Any], key: str, value: bool) -> None:
    if value:
        result[key] = value


def _add_if_not_empty(result: dict[str, Any], key: str, value: Any) -> None:
    if value:
        result[key] = _to_dict_value(value)


class ControlOwnerKind(StrEnum):
    IF = "if"
    MATCH = "match"
    FOR = "for"
    ASYNC_FOR = "async_for"
    WHILE = "while"
    TRY = "try"
    WITH = "with"
    ASYNC_WITH = "async_with"
    OTHER = "other"


class ControlRegionKind(StrEnum):
    CONDITION = "condition"
    TRUE_BODY = "true_body"
    FALSE_BODY = "false_body"
    CASE_BODY = "case_body"
    LOOP_BODY = "loop_body"
    LOOP_ELSE = "loop_else"
    PROTECTED_BODY = "protected_body"
    EXCEPTION_DISPATCH = "exception_dispatch"
    EXCEPTION_HANDLER = "exception_handler"
    SUCCESS_CONTINUATION = "success_continuation"
    POST_EXECUTION = "post_execution"
    BODY = "body"
    OTHER = "other"


class ControlRouteKind(StrEnum):
    ENTER = "enter"
    CONDITIONAL_TRUE = "conditional_true"
    CONDITIONAL_FALSE = "conditional_false"
    MATCH_CASE = "match_case"
    MATCH_FALLTHROUGH = "match_fallthrough"
    LOOP_ITERATION = "loop_iteration"
    LOOP_EXHAUSTED = "loop_exhausted"
    LOOP_BREAK = "loop_break"
    LOOP_CONTINUE = "loop_continue"
    NORMAL_COMPLETION = "normal_completion"
    FALLTHROUGH = "fallthrough"
    EXCEPTION_DISPATCH = "exception_dispatch"
    HANDLER_MATCH = "handler_match"
    HANDLER_MISS = "handler_miss"
    UNHANDLED_EXCEPTION = "unhandled_exception"
    POST_EXECUTION_ENTRY = "post_execution_entry"
    RESUME_PRIOR_OUTCOME = "resume_prior_outcome"
    FUNCTION_RETURN = "function_return"
    TERMINAL = "terminal"
    OTHER = "other"


class ExitKind(StrEnum):
    NORMAL = "normal"
    RETURN = "return"
    RAISE = "raise"
    BREAK = "break"
    CONTINUE = "continue"
    HANDLED_EXCEPTION = "handled_exception"
    UNHANDLED_EXCEPTION = "unhandled_exception"
    IMPLICIT_RETURN = "implicit_return"
    TERMINAL = "terminal"
    OTHER = "other"


class SkipTargetKind(StrEnum):
    REGION = "region"
    EI = "ei"
    LINE = "line"
    OTHER = "other"


class SkipReason(StrEnum):
    ALTERNATE_EI = "alternate_ei"
    CONDITION_TRUE_SKIPS_FALSE_REGION = "condition_true_skips_false_region"
    CONDITION_FALSE_SKIPS_TRUE_REGION = "condition_false_skips_true_region"
    NORMAL_FLOW_BYPASSES_HANDLER = "normal_flow_bypasses_handler"
    EXCEPTION_FLOW_BYPASSES_SUCCESS_CONTINUATION = (
        "exception_flow_bypasses_success_continuation"
    )
    HANDLER_NOT_SELECTED = "handler_not_selected"
    UNHANDLED_EXCEPTION_BYPASSES_LOCAL_CONTINUATION = (
        "unhandled_exception_bypasses_local_continuation"
    )
    MATCH_CASE_NOT_SELECTED = "match_case_not_selected"
    MATCH_FALLTHROUGH = "match_fallthrough"
    LOOP_NOT_ENTERED = "loop_not_entered"
    LOOP_EXHAUSTED = "loop_exhausted"
    LOOP_ELSE_BYPASSED_BY_BREAK = "loop_else_bypassed_by_break"
    POST_EXECUTION_INTERCEPTS_EXIT = "post_execution_intercepts_exit"
    TERMINAL_FLOW = "terminal_flow"
    DISRUPTIVE_FLOW = "disruptive_flow"
    OTHER = "other"


class ControlPolicyKind(StrEnum):
    POST_EXECUTION = "post_execution"
    OTHER = "other"


class PostExecutionMechanismKind(StrEnum):
    TRY_FINALLY = "try_finally"
    CONTEXT_MANAGER_EXIT = "context_manager_exit"
    ASYNC_CONTEXT_MANAGER_EXIT = "async_context_manager_exit"
    ATEXIT_HANDLER = "atexit_handler"
    DEFER = "defer"
    SHUTDOWN_HOOK = "shutdown_hook"
    FRAMEWORK_TEARDOWN = "framework_teardown"
    OTHER = "other"


class PostExecutionBindingMode(StrEnum):
    STRUCTURAL = "structural"
    REGISTERED = "registered"
    OTHER = "other"


class PostExecutionBindingScope(StrEnum):
    PROTECTED_REGION = "protected_region"
    WITH_BODY = "with_body"
    ASYNC_WITH_BODY = "async_with_body"
    SCOPE = "scope"
    FUNCTION = "function"
    MODULE = "module"
    PROCESS = "process"
    FRAMEWORK_LIFECYCLE = "framework_lifecycle"
    OTHER = "other"


class PostExecutionTriggerEvent(StrEnum):
    REGION_EXIT = "region_exit"
    SCOPE_EXIT = "scope_exit"
    FUNCTION_EXIT = "function_exit"
    MODULE_EXIT = "module_exit"
    PROCESS_EXIT = "process_exit"
    REQUEST_EXIT = "request_exit"
    TEST_EXIT = "test_exit"
    SIGNAL = "signal"
    OTHER = "other"


class StatementDecompositionKind(StrEnum):
    EXECUTION = "execution"
    CONTROL = "control"
    RUNTIME_REGISTRATION = "runtime_registration"
    DIAGNOSTIC = "diagnostic"
    OTHER = "other"


@dataclass(frozen=True, kw_only=True)
class SourceAnchor:
    line: int | None = None
    end_line: int | None = None
    col_offset: int | None = None
    end_col_offset: int | None = None
    source_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        _add_if_present(result, "line", self.line)
        _add_if_present(result, "end_line", self.end_line)
        _add_if_present(result, "col_offset", self.col_offset)
        _add_if_present(result, "end_col_offset", self.end_col_offset)
        _add_if_present(result, "source_text", self.source_text)
        return result


@dataclass(frozen=True, kw_only=True)
class SkipTarget:
    kind: SkipTargetKind
    reason: SkipReason
    target_id: str | None = None
    line: int | None = None
    owner_id: str | None = None
    implicit: bool = False
    synthetic: bool = False
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.kind in {SkipTargetKind.REGION, SkipTargetKind.EI}
            and not self.target_id
        ):
            raise ValueError(f"{self.kind} skip target requires target_id")
        if self.kind == SkipTargetKind.LINE and self.line is None:
            raise ValueError("line skip target requires line")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "reason": self.reason.value,
        }
        _add_if_present(result, "target_id", self.target_id)
        _add_if_present(result, "line", self.line)
        _add_if_present(result, "owner_id", self.owner_id)
        _add_if_true(result, "implicit", self.implicit)
        _add_if_true(result, "synthetic", self.synthetic)
        _add_if_present(result, "detail", self.detail)
        _add_if_not_empty(result, "metadata", self.metadata)
        return result


@dataclass(frozen=True, kw_only=True)
class ControlRegion:
    id: str
    owner_id: str
    kind: ControlRegionKind
    start_line: int | None = None
    end_line: int | None = None
    ordinal: int | None = None
    source_construct: str | None = None
    implicit: bool = False
    synthetic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "owner_id": self.owner_id,
            "kind": self.kind.value,
        }
        _add_if_present(result, "start_line", self.start_line)
        _add_if_present(result, "end_line", self.end_line)
        _add_if_present(result, "ordinal", self.ordinal)
        _add_if_present(result, "source_construct", self.source_construct)
        _add_if_true(result, "implicit", self.implicit)
        _add_if_true(result, "synthetic", self.synthetic)
        _add_if_not_empty(result, "metadata", self.metadata)
        return result


@dataclass(frozen=True, kw_only=True)
class ControlRoute:
    id: str
    kind: ControlRouteKind
    owner_id: str
    source_region_id: str | None = None
    source_ei: str | None = None
    target_region_id: str | None = None
    target_ei: str | None = None
    target_line: int | None = None
    condition: str | None = None
    condition_result: bool | None = None
    exit_kind: ExitKind | None = None
    skips: list[SkipTarget] = field(default_factory=list)
    implicit: bool = False
    synthetic: bool = False
    preserves_prior_outcome: bool = False
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind.value,
            "owner_id": self.owner_id,
        }
        _add_if_present(result, "source_region_id", self.source_region_id)
        _add_if_present(result, "source_ei", self.source_ei)
        _add_if_present(result, "target_region_id", self.target_region_id)
        _add_if_present(result, "target_ei", self.target_ei)
        _add_if_present(result, "target_line", self.target_line)
        _add_if_present(result, "condition", self.condition)
        _add_if_present(result, "condition_result", self.condition_result)
        _add_if_present(result, "exit_kind", self.exit_kind)
        _add_if_not_empty(result, "skips", self.skips)
        _add_if_true(result, "implicit", self.implicit)
        _add_if_true(result, "synthetic", self.synthetic)
        _add_if_true(result, "preserves_prior_outcome", self.preserves_prior_outcome)
        _add_if_present(result, "detail", self.detail)
        _add_if_not_empty(result, "metadata", self.metadata)
        return result


@dataclass(frozen=True, kw_only=True)
class ControlPolicy:
    kind: ControlPolicyKind
    owner_id: str
    implicit: bool = False
    synthetic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "owner_id": self.owner_id,
        }
        _add_if_true(result, "implicit", self.implicit)
        _add_if_true(result, "synthetic", self.synthetic)
        _add_if_not_empty(result, "metadata", self.metadata)
        return result


@dataclass(frozen=True, kw_only=True)
class PostExecutionPolicy(ControlPolicy):
    kind: ControlPolicyKind = ControlPolicyKind.POST_EXECUTION
    mechanism_kind: PostExecutionMechanismKind = PostExecutionMechanismKind.OTHER
    binding_mode: PostExecutionBindingMode = PostExecutionBindingMode.STRUCTURAL
    binding_scope: PostExecutionBindingScope = PostExecutionBindingScope.OTHER
    trigger_event: PostExecutionTriggerEvent = PostExecutionTriggerEvent.OTHER
    region_id: str | None = None
    target_region_id: str | None = None
    target_ei: str | None = None
    target_line: int | None = None
    applies_to: list[ExitKind] = field(default_factory=list)
    preserves_prior_outcome: bool = True
    source_construct: str | None = None
    callable_expr: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        _add_if_present(result, "mechanism_kind", self.mechanism_kind)
        _add_if_present(result, "binding_mode", self.binding_mode)
        _add_if_present(result, "binding_scope", self.binding_scope)
        _add_if_present(result, "trigger_event", self.trigger_event)
        _add_if_present(result, "region_id", self.region_id)
        _add_if_present(result, "target_region_id", self.target_region_id)
        _add_if_present(result, "target_ei", self.target_ei)
        _add_if_present(result, "target_line", self.target_line)
        _add_if_not_empty(result, "applies_to", self.applies_to)
        _add_if_true(result, "preserves_prior_outcome", self.preserves_prior_outcome)
        _add_if_present(result, "source_construct", self.source_construct)
        _add_if_present(result, "callable_expr", self.callable_expr)
        _add_if_present(result, "detail", self.detail)
        return result


@dataclass(frozen=True, kw_only=True)
class ControlFlow:
    regions: list[ControlRegion] = field(default_factory=list)
    routes: list[ControlRoute] = field(default_factory=list)
    policies: list[ControlPolicy] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.regions and not self.routes and not self.policies

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        _add_if_not_empty(result, "regions", self.regions)
        _add_if_not_empty(result, "routes", self.routes)
        _add_if_not_empty(result, "policies", self.policies)
        return result

    @classmethod
    def from_decompositions(
        cls,
        decompositions: list[ControlStatementDecomposition],
    ) -> "ControlFlow":
        regions: list[ControlRegion] = []
        routes: list[ControlRoute] = []
        policies: list[ControlPolicy] = []

        for decomposition in decompositions:
            regions.extend(decomposition.regions)
            routes.extend(decomposition.routes)
            policies.extend(decomposition.policies)

        return cls(
            regions=regions,
            routes=routes,
            policies=policies,
        )


@dataclass(frozen=True, kw_only=True)
class StatementDecomposition:
    kind: StatementDecompositionKind
    description: str
    source: SourceAnchor | None = None
    implicit: bool = False
    synthetic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "description": self.description,
        }
        _add_if_present(result, "source", self.source)
        _add_if_true(result, "implicit", self.implicit)
        _add_if_true(result, "synthetic", self.synthetic)
        _add_if_not_empty(result, "metadata", self.metadata)
        return result


@dataclass(frozen=True, kw_only=True)
class ExecutionStatementDecomposition(StatementDecomposition):
    kind: StatementDecompositionKind = StatementDecompositionKind.EXECUTION
    statement_outcome: StatementOutcome | None = None
    conditional_targets: list[ConditionalTarget] | None = None
    disruptive_outcomes: list[DisruptiveOutcome] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        _add_if_present(result, "statement_outcome", self.statement_outcome)
        _add_if_not_empty(result, "conditional_targets", self.conditional_targets)
        _add_if_not_empty(result, "disruptive_outcomes", self.disruptive_outcomes)
        return result


@dataclass(frozen=True, kw_only=True)
class ControlStatementDecomposition(StatementDecomposition):
    kind: StatementDecompositionKind = StatementDecompositionKind.CONTROL
    owner_kind: ControlOwnerKind
    control_id: str
    line: int
    end_line: int | None = None
    regions: list[ControlRegion] = field(default_factory=list)
    routes: list[ControlRoute] = field(default_factory=list)
    policies: list[ControlPolicy] = field(default_factory=list)
    source_construct: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["owner_kind"] = self.owner_kind.value
        result["control_id"] = self.control_id
        result["line"] = self.line
        _add_if_present(result, "end_line", self.end_line)
        _add_if_not_empty(result, "regions", self.regions)
        _add_if_not_empty(result, "routes", self.routes)
        _add_if_not_empty(result, "policies", self.policies)
        _add_if_present(result, "source_construct", self.source_construct)
        return result


@dataclass(frozen=True, kw_only=True)
class DecomposerResult:
    decomposition: StatementDecomposition
    candidate_node: ast.AST | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def description(self) -> str:
        return self.decomposition.description

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "decomposition": self.decomposition.to_dict(),
        }
        if self.candidate_node is not None:
            result["candidate_node"] = {
                "node_type": type(self.candidate_node).__name__,
                "line": getattr(self.candidate_node, "lineno", None),
                "end_line": getattr(self.candidate_node, "end_lineno", None),
                "source": ast.unparse(self.candidate_node),
            }
        _add_if_not_empty(result, "metadata", self.metadata)
        return result
