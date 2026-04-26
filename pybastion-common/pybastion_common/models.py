#!/usr/bin/env python3
"""
Shared data models for the unit ledger analysis pipeline.

These models define contracts between pipeline stages:
- stage3_enumerate_callables.py (Stage 1)
- stage2_enumerate_exec_items.py (Stage 2)
- CFG path enumeration (Stage 3)
- Ledger transformation (Stage 4)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from typing_extensions import Self

from pybastion_unit.shared.callable_id_generation import ei_id_to_integration_id


@dataclass(slots=True)
class UnitBindingEntry:
    id: str
    kind: str
    binding_kind: str
    name: str
    fully_qualified_name: str
    parent_id: str
    owner_id: str
    lineno: int
    end_lineno: int
    ordinal_within_unit: int
    annotation: str | None = None
    value_kind: str | None = None
    value_expr: str | None = None

@dataclass(slots=True)
class UnitIndexEntry:
    id: str
    kind: str
    name: str
    fully_qualified_name: str
    parent_id: str | None
    owner_id: str
    lineno: int
    end_lineno: int
    ordinal_within_parent: int
    child_ids: list[str] = field(default_factory=list)
    is_async: bool = False


@dataclass(slots=True)
class UnitIndex:
    unit_id: str
    fully_qualified_name: str
    filepath: str
    language: str
    source_hash: str
    entries: list[UnitIndexEntry]
    bindings: list[UnitBindingEntry] = field(default_factory=list)


@dataclass(slots=True)
class ProjectIndex:
    source_root: str
    units: list[UnitIndex]


# =============================================================================
# Common Types
# =============================================================================

@dataclass
class TypeRef:
    """
    Type reference with optional generic arguments.

    Examples:
        int -> TypeRef(name='int')
        list[str] -> TypeRef(name='list', args=[TypeRef(name='str')])
        dict[str, Any] -> TypeRef(name='dict', args=[TypeRef(name='str'), TypeRef(name='Any')])
    """
    name: str
    args: list[TypeRef] = field(default_factory=list)

    @classmethod
    def from_annotation_ast(cls, annotation: ast.expr | None) -> Self | None:
        if annotation is None:
            return None

        if isinstance(annotation, ast.Name):
            return cls(name=annotation.id)

        if isinstance(annotation, ast.Attribute):
            return cls(name=ast.unparse(annotation))

        if isinstance(annotation, ast.Subscript):
            base_type = ast.unparse(annotation.value)
            args: list[TypeRef] = []

            if isinstance(annotation.slice, ast.Tuple):
                for elt in annotation.slice.elts:
                    arg_ref = cls.from_annotation_ast(elt)
                    if arg_ref is not None:
                        args.append(arg_ref)
            else:
                arg_ref = cls.from_annotation_ast(annotation.slice)
                if arg_ref is not None:
                    args.append(arg_ref)

            if base_type in {"Optional", "typing.Optional"} and len(args) == 1:
                return cls(name="Union", args=[args[0], cls(name="None")])

            return cls(name=base_type, args=args)

        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            left_ref = cls.from_annotation_ast(annotation.left)
            right_ref = cls.from_annotation_ast(annotation.right)

            args: list[TypeRef] = []
            if left_ref is not None:
                args.append(left_ref)
            if right_ref is not None:
                args.append(right_ref)

            return cls(name="Union", args=args)

        if isinstance(annotation, ast.Constant):
            if annotation.value is None:
                return cls(name="None")
            return cls(name=repr(annotation.value))

        if isinstance(annotation, ast.Tuple):
            return cls(name=ast.unparse(annotation))

        return cls(name=ast.unparse(annotation))

    @classmethod
    def from_annotation_string(cls, annotation: str) -> Self:
        expr = ast.parse(annotation, mode="eval").body
        result = cls.from_annotation_ast(expr)
        if result is None:
            raise ValueError(f"Could not parse annotation: {annotation}")
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TypeRef | None:
        """Parse from inventory dict format."""
        if not data:
            return None
        return cls(
            name=data['name'],
            args=[cls.from_dict(arg) for arg in data.get('args', []) if arg is not None]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {'name': self.name}
        if self.args:
            result['args'] = [arg.to_dict() for arg in self.args]
        return result

    def to_annotation_string(self) -> str:
        if not self.args:
            return self.name

        if self.name == "Union":
            return " | ".join(arg.to_annotation_string() for arg in self.args)

        inner = ", ".join(arg.to_annotation_string() for arg in self.args)
        return f"{self.name}[{inner}]"

    def to_resolver_string(self) -> str:
        if not self.args:
            return self.name

        if self.name == "Union":
            non_none_args = [arg for arg in self.args if arg.name != "None"]
            if len(non_none_args) == 1:
                return non_none_args[0].to_resolver_string()
            return " | ".join(arg.to_resolver_string() for arg in self.args)

        inner = ", ".join(arg.to_annotation_string() for arg in self.args)
        return f"{self.name}::{inner}"


@dataclass
class ParamSpec:
    """Parameter specification."""
    name: str
    type: TypeRef | None = None
    default: str | None = None  # Store as string representation

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParamSpec:
        """Parse from inventory dict format."""
        return cls(
            name=data['name'],
            type=TypeRef.from_dict(data['type']) if 'type' in data else None,
            default=data.get('default')
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {'name': self.name}
        if self.type:
            result['type'] = self.type.to_dict()
        if self.default is not None:
            result['default'] = self.default
        return result


# =============================================================================
# Execution Items (Branches)
# =============================================================================

@dataclass
class BranchConstraint:
    """
    Represents a logical constraint extracted from source code branches
    (flow control and operations).

    This class is utilized for constraint analysis, conflict detection,
    and logical evaluation of branching behavior during program execution.

    Attributes:
        expr (str): The actual constraint expression directly extracted
            from the source code. Examples include comparison conditions,
            type checks, and function calls.

        polarity (bool | None): Indicates whether the constraint is satisfied
            or not for logical conditions:
            - True: The condition is satisfied (e.g., "if condition is True").
            - False: The condition is not satisfied (e.g., "if condition is False").
            - None: Does not apply to operations or assignments.

        constraint_type (str): Type of constraint this instance represents.
            Common types include:
            - 'condition': Boolean conditions like 'if', 'while', and 'assert'.
            - 'assignment': Constraints from variable assignments.
            - 'operation': Execution of a function or operation.
            - 'exception': Exception constraint from 'try/except' blocks.
            - 'iteration': Constraints for loop iterations.
            - 'match_case': Constraints from pattern-matching statements.
            - 'terminal': Terminal nodes like 'return' or 'raise' statements.

        variables_read (set[str]): Variables referenced within the constraint.

        variables_written (set[str]): Variables being assigned or written to within
            the constraint.

        branch_id (str): Unique identifier for the corresponding branch in the
            source code. Useful for traceability in control-flow analysis.

        line (int): Line number of the source code where the constraint originates.

        operators (list[str]): List of operators used in the constraint expression
            (e.g., logical and comparison operators used in 'expr').

        operands (list[str]): List of operands referenced or manipulated in the
            constraint expression. Typically variables, literals, and function names.

        smt_expr (str | None): Precomputed SMT (Satisfiability Modulo Theories)
            representation of the constraint for solver-based evaluations, if available.

        operation_target (str | None): Fully qualified name of an operation or function
            being invoked in 'operation' constraints. Examples include
            'module.function_name'.

        exception_types (list[str]): List of exception types specifically handled
            by this constraint in 'exception' constraints.

        implies (list[str]): List of branch IDs logically implied by this constraint
            (e.g., "x > 10" implies "x > 5"). Used for optimization and analysis.

        excludes (list[str]): List of branch IDs that this constraint logically
            excludes as mutually exclusive (e.g., "x > 10" excludes "x < 5").

        metadata (dict[str, Any]): Additional metadata for extensibility and specialized
            constraint analysis. Example fields include:
            - 'loop_header': Indicates whether this is a loop entry constraint.
            - 'loop_id': Unique identifier for the surrounding loop structure.
            - 'mutually_exclusive_with': Specifies sets of mutually exclusive
              branch constraints.
    """

    expr: str
    """
    The actual constraint expression from source code.
    Examples: 'value > 0', 'len(items) == 0', 'isinstance(obj, dict)'
    For operations: the operation signature like 'validate(data)'
    """

    polarity: bool | None
    """
    Polarity of the constraint (meaningful for conditions):
    - True: condition is satisfied (if-true, while-true)
    - False: condition is not satisfied (if-false, while-false)
    - None: not applicable (operations, assignments)
    """

    constraint_type: str
    """
    Type of constraint this represents:
    - 'condition': Boolean condition (if/while/assert)
    - 'iteration': Loop iteration constraint (for/while)
    - 'exception': Exception handling constraint (try/except)
    - 'operation': Operation execution (function call, constructor)
    - 'assignment': Variable assignment
    - 'match_case': Pattern matching case
    - 'terminal': Terminal node (return/raise with no continuation)
    """

    variables_read: set[str] = field(default_factory=set)
    """Variables read/referenced in this constraint."""

    variables_written: set[str] = field(default_factory=set)
    """Variables written/assigned in this constraint."""

    branch_id: str = ""
    """Source branch ID for traceability."""

    line: int = 0
    """Source line number in the code."""

    operators: list[str] = field(default_factory=list)
    """
    Operators used in the constraint expression.
    Examples: ['>', '==', 'in', 'is', 'and']
    Used for SMT parsing and constraint analysis.
    """

    operands: list[str] = field(default_factory=list)
    """
    Operands in the constraint expression.
    Examples: ['value', '0'] for 'value > 0'
    Used for SMT constraint construction.
    """

    smt_expr: str | None = None
    """
    Pre-computed SMT expression if available.
    Can be populated during constraint extraction for performance.
    """

    operation_target: str | None = None
    """
    Fully qualified name of operation being called (for operation constraints).
    Example: 'project_resolution_engine.api.validate'
    """

    exception_types: list[str] = field(default_factory=list)
    """
    Specific exception types for exception constraints.
    Examples: ['ValueError', 'KeyError']
    """

    implies: list[str] = field(default_factory=list)
    """
    Branch IDs that this constraint logically implies.
    Example: '123' implies '234'
    Used for constraint propagation and optimization.
    """

    excludes: list[str] = field(default_factory=list)
    """
    Branch IDs that this constraint excludes (mutually exclusive).
    Example: '123' excludes '234'
    Used for conflict detection.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """
    Additional metadata for specialized analysis:
    - 'loop_header': bool - whether this is a loop entry constraint
    - 'loop_id': str - identifier for the containing loop
    - 'mutually_exclusive_with': set[str] - alternative branch IDs
    """

    def conflicts_with(self, other: BranchConstraint) -> bool:
        """
        Check if this constraint conflicts with another constraint.

        Constraints conflict if they cannot both be true in the same execution path.

        Args:
            other: Another constraint to check against

        Returns:
            True if constraints are mutually exclusive
        """
        # Same expression with opposite polarities = conflict
        if (self.expr == other.expr and
                self.constraint_type == 'condition' and
                self.polarity is not None and other.polarity is not None):
            return self.polarity != other.polarity

        # Check explicit exclusions
        if other.branch_id in self.excludes:
            return True

        return False

    def is_compatible_with(self, other: BranchConstraint) -> bool:
        """
        Check if this constraint can coexist with another in the same path.

        More permissive than conflicts_with - checks for compatibility
        rather than explicit conflicts.

        Args:
            other: Another constraint to check against

        Returns:
            True if constraints can coexist in the same execution path
        """
        if self.conflicts_with(other):
            return False

        # TODO: Add more sophisticated compatibility checks here:
        # - Variable state conflicts
        # - Type conflicts
        # - Value range conflicts

        return True

    def get_signature(self) -> str:
        """
        Get a signature representing this constraint.

        Used for grouping and comparison in path analysis.

        Returns:
            Signature string in format: "type:expr:polarity"

        Examples:
            "condition:value > 0:true"
            "operation:validate(data)"
            "exception:ValueError"
        """
        parts = [self.constraint_type, self.expr]

        if self.polarity is not None:
            parts.append('true' if self.polarity else 'false')

        return ':'.join(parts)

    @classmethod
    def from_branch(cls, branch) -> BranchConstraint | None:
        """
        Extract constraint from a Branch object.

        Args:
            branch: Branch object with constraint metadata

        Returns:
            BranchConstraint if the branch has any constraint data, None otherwise
        """
        if not branch.constraint_expr:
            return None

        return cls(
            expr=branch.constraint_expr,
            polarity=branch.constraint_polarity,
            constraint_type=branch.constraint_type or 'unknown',
            variables_read=branch.variables_read.copy(),
            variables_written=branch.variables_written.copy(),
            branch_id=branch.id,
            line=branch.line,
            operation_target=branch.metadata.get('operation_target'),
            exception_types=branch.metadata.get('exception_types', []),
            metadata=branch.metadata.copy()
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result: dict[str, Any] = {
            'expr': self.expr,
            'constraint_type': self.constraint_type,
        }

        if self.polarity is not None:
            result['polarity'] = self.polarity

        if self.variables_read:
            result['variables_read'] = sorted(list(self.variables_read))

        if self.variables_written:
            result['variables_written'] = sorted(list(self.variables_written))

        if self.branch_id:
            result['branch_id'] = self.branch_id

        if self.line > 0:
            result['line'] = self.line

        if self.operators:
            result['operators'] = self.operators

        if self.operands:
            result['operands'] = self.operands

        if self.smt_expr:
            result['smt_expr'] = self.smt_expr

        if self.operation_target:
            result['operation_target'] = self.operation_target

        if self.exception_types:
            result['exception_types'] = self.exception_types

        if self.implies:
            result['implies'] = self.implies

        if self.excludes:
            result['excludes'] = self.excludes

        if self.metadata:
            result['metadata'] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BranchConstraint:
        """Parse from dictionary format."""
        return cls(
            expr=data['expr'],
            polarity=data.get('polarity'),
            constraint_type=data['constraint_type'],
            variables_read=set(data.get('variables_read', [])),
            variables_written=set(data.get('variables_written', [])),
            branch_id=data.get('branch_id', ''),
            line=data.get('line', 0),
            operators=data.get('operators', []),
            operands=data.get('operands', []),
            smt_expr=data.get('smt_expr'),
            operation_target=data.get('operation_target'),
            exception_types=data.get('exception_types', []),
            implies=data.get('implies', []),
            excludes=data.get('excludes', []),
            metadata=data.get('metadata', {})
        )

    def __repr__(self) -> str:
        """Enhanced repr showing key constraint information."""
        parts = [f"BranchConstraint(type={self.constraint_type!r}"]

        # Truncate long expressions
        expr = self.expr
        if len(expr) > 40:
            expr = expr[:37] + "..."
        parts.append(f"expr={expr!r}")

        if self.polarity is not None:
            parts.append(f"polarity={self.polarity}")

        if self.branch_id:
            parts.append(f"branch={self.branch_id!r}")

        return ', '.join(parts) + ')'


@dataclass
class TargetHint:
    line: int | None = None
    role: str | None = None
    polarity: bool | None = None
    expr: str | None = None
    stmt_type: str | None = None
    skips_lines: list[int] = field(default_factory=list)
    skips_eis: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result: dict[str, Any] = {}
        if self.line is not None:
            result['line'] = self.line
        if self.role is not None:
            result['role'] = self.role
        if self.polarity is not None:
            result['polarity'] = self.polarity
        if self.expr is not None:
            result['expr'] = self.expr

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Parse from dictionary format."""
        return cls(
            line=data.get('line'),
            role=data.get('role'),
            polarity=data.get('polarity'),
            expr=data.get('expr'),
            stmt_type=data.get('stmt_type'),
        )


@dataclass
class StatementOutcome:
    outcome: str
    target_line: int | None = None
    target_ei: str | None = None
    skips_lines: list[int] = field(default_factory=list)
    skips_eis: list[str] = field(default_factory=list)
    is_terminal: bool = False
    terminates_via: str | None = None
    target_hint: TargetHint | None = None
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result: dict[str, Any] = {
            'outcome': self.outcome,
        }
        if self.target_line is not None:
            result['target_line'] = self.target_line
        if self.target_ei is not None:
            result['target_ei'] = self.target_ei
        if self.is_terminal:
            result['is_terminal'] = self.is_terminal
            result['terminates_via'] = self.terminates_via
        if self.skips_lines:
            result['skips_lines'] = self.skips_lines
        if self.skips_eis:
            result['skips_eis'] = self.skips_eis
        if self.target_hint is not None:
            result['target_hint'] = self.target_hint.to_dict()
        if self.synthetic:
            result['synthetic'] = self.synthetic
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Parse from dictionary format."""
        return cls(
            outcome=data['outcome'],
            target_line=data.get('target_line'),
            target_ei=data.get('target_ei'),
            is_terminal=data.get('is_terminal', False),
            terminates_via=data.get('terminates_via'),
            skips_lines=data.get('skips_lines', []),
            skips_eis=data.get('skips_eis', []),
            target_hint=TargetHint.from_dict(data.get('target_hint', {})) if data.get('target_hint') else None,
            synthetic=data.get('synthetic', False),
        )


@dataclass
class DisruptiveOutcome:
    outcome: str
    target_line: int | None = None
    target_ei: str | None = None
    skips_lines: list[int] = field(default_factory=list)
    skips_eis: list[str] = field(default_factory=list)
    is_terminal: bool = False
    terminates_via: str | None = None
    target_hint: TargetHint | None = None
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result: dict[str, Any] = {
            'outcome': self.outcome,
        }
        if self.target_line is not None:
            result['target_line'] = self.target_line
        if self.target_ei is not None:
            result['target_ei'] = self.target_ei
        if self.is_terminal:
            result['is_terminal'] = self.is_terminal
            result['terminates_via'] = self.terminates_via
        if self.skips_lines:
            result['skips_lines'] = self.skips_lines
        if self.skips_eis:
            result['skips_eis'] = self.skips_eis
        if self.target_hint is not None:
            result['target_hint'] = self.target_hint.to_dict()
        if self.synthetic:
            result['synthetic'] = self.synthetic
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Parse from dictionary format."""
        return cls(
            outcome=data['outcome'],
            target_line=data.get('target_line'),
            target_ei=data.get('target_ei'),
            is_terminal=data.get('is_terminal', False),
            terminates_via=data.get('terminates_via'),
            skips_lines=data.get('skips_lines', []),
            skips_eis=data.get('skips_eis', []),
            target_hint=TargetHint.from_dict(data.get('target_hint', {})) if data.get('target_hint') else None,
            synthetic=data.get('synthetic', False),
        )


@dataclass
class ConditionalTarget:
    target_condition: str
    condition_result: bool
    target_line: int | None = None
    target_ei: str | None = None
    skips_lines: list[int] = field(default_factory=list)
    skips_eis: list[str] = field(default_factory=list)
    is_terminal: bool = False
    terminates_via: str | None = None
    target_hint: TargetHint | None = None
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result: dict[str, Any] = {
            'target_condition': self.target_condition,
            'condition_result': self.condition_result,
        }
        if self.target_line is not None:
            result['target_line'] = self.target_line
        if self.target_ei is not None:
            result['target_ei'] = self.target_ei
        if self.is_terminal:
            result['is_terminal'] = self.is_terminal
            result['terminates_via'] = self.terminates_via
        if self.skips_lines:
            result['skips_lines'] = self.skips_lines
        if self.skips_eis:
            result['skips_eis'] = self.skips_eis
        if self.target_hint is not None:
            result['target_hint'] = self.target_hint.to_dict()
        if self.synthetic:
            result['synthetic'] = self.synthetic
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Parse from dictionary format."""
        return cls(
            target_condition=data['target_condition'],
            condition_result=data['condition_result'],
            target_line=data.get('target_line'),
            target_ei=data.get('target_ei'),
            is_terminal=data.get('is_terminal', False),
            terminates_via=data.get('terminates_via'),
            skips_lines=data.get('skips_lines', []),
            skips_eis=data.get('skips_eis', []),
            target_hint=TargetHint.from_dict(data.get('target_hint', {})) if data.get('target_hint') else None,
            synthetic=data.get('synthetic', False),
        )


@dataclass(frozen=True)
class OwnerInfo:
    stmt_type: str | None = None
    region: str | None = None
    line: int | None = None
    branch_id: str | None = None
    expr: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        result: dict[str, Any] = {}
        if self.stmt_type is not None:
            result['stmt_type'] = self.stmt_type
        if self.region is not None:
            result['region'] = self.region
        if self.line is not None:
            result['line'] = self.line
        if self.branch_id is not None:
            result['branch_id'] = self.branch_id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Parse from dictionary format."""
        return cls(
            stmt_type=data.get('stmt_type'),
            region=data.get('region'),
            line=data.get('line'),
            branch_id=data.get('branch_id'),
            expr=data.get('expr'),
        )


@dataclass
class Branch:
    """
    Execution Item (EI) representation with constraint tracking.

    Contract between stage2_enumerate_exec_items.py and downstream stages.
    Called "Branch" in current code but represents an Execution Item.

    Enhanced with constraint metadata to enable path feasibility analysis.

    In the newer model, Branch is a container for concrete outcome objects rather
    than a flattened holder of top-level target/terminal fields.
    """

    id: str
    """Unique identifier for the EI."""

    line: int
    """Line number of the EI statement."""

    condition: str
    """Condition expression for the EI."""

    description: str
    """Human-readable description of the EI itself."""

    decorators: list[dict[str, Any]] = field(default_factory=list)
    """
    Statement-level decorators (e.g., for feature flow tracing).
    """

    constraint: BranchConstraint | None = None
    """
    Constraint object containing all constraint-related metadata.
    Replaces scattered constraint_type, constraint_expr, constraint_polarity fields.
    """

    stmt_type: str | None = None
    """
    Statement type of the EI, e.g., 'if', 'for', 'try', 'raise', etc.
    Used for special handling of certain statement types.
    """

    statement_outcome: StatementOutcome | None = None
    """
    Singular normal statement outcome for this EI, when applicable.

    This is used for straight-line statements or other cases where the EI
    has one primary modeled outcome.
    """

    disruptive_outcomes: list[DisruptiveOutcome] | None = None
    """
    Disruptive outcomes for this EI, if any.

    These represent outcomes that disrupt the normal execution path, such as
    propagated exceptions or other non-standard flow outcomes.
    """

    conditional_targets: list[ConditionalTarget] | None = None
    """
    Conditional targets for this EI when it is a branching EI.

    Each ConditionalTarget represents one path-conditioned outcome of evaluating
    the control condition, including short-circuit-distinct boolean outcomes.
    """

    owner_info: OwnerInfo | None = None
    """
    Information about the owner of the EI, if available.
    
    For example, if a statement is part of a for loop, this field will contain
    information about the loop's control statement.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """
    Additional metadata for specialized analysis:
    - 'loop_header': bool - whether this is a loop entry point
    - 'loop_id': str - identifier for the containing loop
    - 'mutually_exclusive_with': set[str] - EI IDs that are alternatives
    """

    def __post_init__(self) -> None:
        """Validate branch structure."""
        if not self.id:
            raise ValueError("Branch ID cannot be empty")
        if self.line <= 0:
            raise ValueError(f"Invalid line number: {self.line}")

        present = sum([
            self.statement_outcome is not None,
            bool(self.conditional_targets),
            bool(self.disruptive_outcomes),
        ])

        if present == 0:
            raise ValueError(
                "Branch must have at least one of: statement_outcome, "
                "conditional_targets, disruptive_outcomes"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Parse from inventory dict format."""
        return cls(
            id=data["id"],
            line=data["line"],
            condition=data["condition"],
            description=data["description"],
            decorators=data.get("decorators", []),
            constraint=(
                BranchConstraint.from_dict(data["constraint"])
                if data.get("constraint") else None
            ),
            stmt_type=data.get("stmt_type"),
            statement_outcome=(
                StatementOutcome.from_dict(data["statement_outcome"])
                if data.get("statement_outcome") else None
            ),
            conditional_targets=(
                [ConditionalTarget.from_dict(ct) for ct in data["conditional_targets"]]
                if data.get("conditional_targets") else None
            ),
            disruptive_outcomes=(
                [DisruptiveOutcome.from_dict(o) for o in data["disruptive_outcomes"]]
                if data.get("disruptive_outcomes") else None
            ),
            owner_info=(
                OwnerInfo.from_dict(data["owner_info"]) if data.get("owner_info") else None
            ),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {
            "id": self.id,
            "line": self.line,
            "condition": self.condition,
            "description": self.description,
        }

        if self.decorators:
            result["decorators"] = self.decorators

        if self.constraint is not None:
            result["constraint"] = self.constraint.to_dict()

        if self.stmt_type is not None:
            result["stmt_type"] = self.stmt_type

        if self.statement_outcome is not None:
            result["statement_outcome"] = self.statement_outcome.to_dict()

        if self.conditional_targets:
            result["conditional_targets"] = [
                ct.to_dict() for ct in self.conditional_targets
            ]

        if self.disruptive_outcomes:
            result["disruptive_outcomes"] = [
                o.to_dict() for o in self.disruptive_outcomes
            ]

        if self.owner_info:
            result["owner_info"] = self.owner_info.to_dict()

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_ledger_ei_spec(self) -> dict[str, Any]:
        """
        Transform to ledger EiSpec format.

        Ledger format includes constraint metadata for downstream path analysis
        and test generation. Uses snake_case for consistency.
        """
        result: dict[str, Any] = {
            "id": self.id,
            "condition": self.condition,
            "description": self.description,
        }

        if self.constraint is not None:
            result["constraint"] = self.constraint.to_dict()

        if self.stmt_type is not None:
            result["stmt_type"] = self.stmt_type

        if self.statement_outcome is not None:
            result["statement_outcome"] = self.statement_outcome.to_dict()

        if self.conditional_targets:
            result["conditional_targets"] = [
                ct.to_dict() for ct in self.conditional_targets
            ]

        if self.disruptive_outcomes:
            result["disruptive_outcomes"] = [
                o.to_dict() for o in self.disruptive_outcomes
            ]

        if self.decorators:
            result["decorators"] = self.decorators

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def conflicts_with(self, other: Branch) -> bool:
        """
        Check if this branch's constraints conflict with another branch.

        Used for path feasibility analysis to detect impossible paths.

        Returns:
            True if constraints are mutually exclusive
        """
        if self.constraint is not None and other.constraint is not None:
            return self.constraint.conflicts_with(other.constraint)

        return False

    def get_constraint_signature(self) -> str:
        """
        Get a signature representing this branch's constraint.

        Used for grouping and comparison in path analysis.

        Examples:
            "condition:value > 0:true"
            "operation:validate(data)"
            "exception:ValueError"
        """
        if self.constraint is not None:
            return self.constraint.get_signature()

        return ""

    def is_alternative_to(self, other: Branch) -> bool:
        """
        Check if this branch is a mutually exclusive alternative to another.

        Branches are alternatives if they:
        1. Are on the same line (same decision point)
        2. Have the same condition but different descriptions

        Returns:
            True if branches are mutually exclusive alternatives
        """
        return (
                self.line == other.line
                and self.condition == other.condition
                and self.description != other.description
        )

    def __repr__(self) -> str:
        """Enhanced repr showing constraint and terminal information."""
        parts = [f"Branch(id={self.id!r}, line={self.line}"]

        if self.constraint is not None:
            parts.append(f"constraint={self.constraint.constraint_type!r}")

        if self.constraint is not None and self.constraint.expr:
            expr = self.constraint.expr
            if len(expr) > 40:
                expr = expr[:37] + "..."
            parts.append(f"expr={expr!r}")

        if self.constraint is not None and self.constraint.polarity is not None:
            parts.append(f"polarity={self.constraint.polarity}")

        if self.statement_outcome is not None and self.statement_outcome.is_terminal:
            parts.append(f"terminal={self.statement_outcome.terminates_via!r}")
        elif self.disruptive_outcomes:
            terminal_modes = sorted({
                outcome.terminates_via
                for outcome in self.disruptive_outcomes
                if outcome.is_terminal and outcome.terminates_via is not None
            })
            if terminal_modes:
                parts.append(f"terminal={terminal_modes!r}")

        return ", ".join(parts) + ")"


# =============================================================================
# Integration Points
# =============================================================================

class IntegrationType(str, Enum):
    """Type of integration point."""
    CALL = 'call'
    CONSTRUCT = 'construct'
    IMPORT = 'import'
    DISPATCH = 'dispatch'
    IO = 'io'
    OTHER = 'other'


class IntegrationCategory(str, Enum):
    """Category of integration after classification."""
    INTERUNIT = 'interunit'
    STDLIB = 'stdlib'
    EXTLIB = 'extlib'
    BOUNDARY = 'boundaries'
    UNKNOWN = 'unknown'


class ResolutionKind(str, Enum):
    EXACT = "exact"
    CONTRACT = "contract"
    NORMALIZED = "normalized"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


@dataclass
class IntegrationCandidate:
    """
    Integration point before categorization.

    Contract between stage3_enumerate_callables.py (Stage 1) and CFG enumeration (Stage 3).
    Stage 3 populates the execution_paths field.

    target:
        Original source-form target as written in code, or the closest raw form
        extracted from the EI/AST analysis.

    resolved_target:
        Best resolved/normalized target the analyzer can determine.
        When resolution does not improve the target, this should usually equal target.

    resolution_kind:
        Describes how strong the resolution is:
        - exact: resolved to a concrete callable FQN
        - contract: resolved to a contract/base/interface method FQN
        - normalized: improved or expanded form, but not necessarily a concrete callable
        - ambiguous: multiple plausible targets
        - unresolved: no improvement beyond the original target

    candidate_targets:
        Optional list of plausible resolved targets when resolution is ambiguous.

    resolution_basis:
        Optional explanation of how resolution occurred, such as:
        - "import_map"
        - "same_unit_local"
        - "param_type"
        - "class_field_type"
        - "contract_dispatch"
        - "fallback_original"
    """
    type: str  # IntegrationType value
    target: str
    line: int
    signature: str

    resolved_target: str | None = None
    resolution_kind: ResolutionKind = ResolutionKind.UNRESOLVED
    candidate_targets: list[str] = field(default_factory=list)
    resolution_basis: str | None = None
    execution_paths: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate integration candidate."""
        if not self.target:
            raise ValueError("Integration target cannot be empty")
        if self.line <= 0:
            raise ValueError(f"Invalid line number: {self.line}")

        if self.resolved_target is None:
            self.resolved_target = self.target

        if not self.resolution_kind:
            self.resolution_kind = ResolutionKind.UNRESOLVED

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Parse from inventory dict format."""
        target = data['target']
        return cls(
            type=data['type'],
            target=target,
            line=data['line'],
            signature=data['signature'],
            resolved_target=data.get('resolved_target', target),
            resolution_kind=ResolutionKind(data.get("resolution_kind", ResolutionKind.UNRESOLVED.value)),
            candidate_targets=data.get('candidate_targets', []),
            resolution_basis=data.get('resolution_basis'),
            execution_paths=data.get('execution_paths', [])
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {
            'type': self.type,
            'target': self.target,
            'line': self.line,
            'signature': self.signature,
        }

        if self.resolved_target is not None:
            result['resolved_target'] = self.resolved_target

        if self.resolution_kind:
            result["resolution_kind"] = self.resolution_kind.value

        if self.candidate_targets:
            result['candidate_targets'] = self.candidate_targets

        if self.resolution_basis:
            result['resolution_basis'] = self.resolution_basis

        if self.execution_paths:
            result['execution_paths'] = self.execution_paths

        return result

    def to_ledger_integration_fact(self) -> dict[str, Any]:
        """
        Transform to ledger IntegrationFact format.

        Returns:
            dict in ledger IntegrationFact format
        """
        integration_id: str | None = None
        if self.execution_paths and self.execution_paths[0]:
            last_ei = self.execution_paths[0][-1]
            integration_id = ei_id_to_integration_id(last_ei)

        fact: dict[str, Any] = {}

        if integration_id:
            fact['id'] = integration_id

        fact['target'] = self.target
        fact['resolved_target'] = self.resolved_target
        fact['resolution_kind'] = self.resolution_kind
        fact['kind'] = self.type
        fact['signature'] = self.signature
        fact['execution_paths'] = self.execution_paths

        if self.candidate_targets:
            fact['candidate_targets'] = self.candidate_targets

        if self.resolution_basis:
            fact['resolution_basis'] = self.resolution_basis

        return fact


# =============================================================================
# Callable Entries
# =============================================================================

@dataclass
class CallableSignatureInfo:
    signature: str | None = None
    visibility: str | None = None
    decorators: list[dict[str, Any]] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    params: list[ParamSpec] = field(default_factory=list)
    return_type: TypeRef | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            signature=data.get("signature"),
            visibility=data.get("visibility"),
            decorators=data.get("decorators", []),
            modifiers=data.get("modifiers", []),
            params=[ParamSpec.from_dict(p) for p in data.get("params", [])],
            return_type=TypeRef.from_dict(data.get("return_type")) if data.get("return_type") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.signature is not None:
            result["signature"] = self.signature
        if self.visibility is not None:
            result["visibility"] = self.visibility
        if self.decorators:
            result["decorators"] = self.decorators
        if self.modifiers:
            result["modifiers"] = self.modifiers
        if self.params:
            result["params"] = [param.to_dict() for param in self.params]
        if self.return_type is not None:
            result["return_type"] = self.return_type.to_dict()
        return result


@dataclass
class CallableHierarchyInfo:
    base_classes: list[str] = field(default_factory=list)
    resolved_base_classes: list[str] = field(default_factory=list)
    contract_base_classes: list[str] = field(default_factory=list)
    owner_class_fqn: str | None = None
    is_abstract: bool = False
    is_contract_method: bool = False
    overrides: list[str] = field(default_factory=list)
    implements_contract_method: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            base_classes=data.get("base_classes", []),
            resolved_base_classes=data.get("resolved_base_classes", []),
            contract_base_classes=data.get("contract_base_classes", []),
            owner_class_fqn=data.get("owner_class_fqn"),
            is_abstract=data.get("is_abstract", False),
            is_contract_method=data.get("is_contract_method", False),
            overrides=data.get("overrides", []),
            implements_contract_method=data.get("implements_contract_method", False),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.base_classes:
            result["base_classes"] = self.base_classes
        if self.resolved_base_classes:
            result["resolved_base_classes"] = self.resolved_base_classes
        if self.contract_base_classes:
            result["contract_base_classes"] = self.contract_base_classes
        if self.owner_class_fqn is not None:
            result["owner_class_fqn"] = self.owner_class_fqn
        if self.is_abstract:
            result["is_abstract"] = self.is_abstract
        if self.is_contract_method:
            result["is_contract_method"] = self.is_contract_method
        if self.overrides:
            result["overrides"] = self.overrides
        if self.implements_contract_method:
            result["implements_contract_method"] = self.implements_contract_method
        return result


@dataclass
class CallableAnalysisInfo:
    branches: list[Branch] = field(default_factory=list)
    integration_candidates: list[IntegrationCandidate] = field(default_factory=list)
    total_eis: int = 0
    needs_callable_analysis: bool = False
    is_stub: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            branches=[Branch.from_dict(b) for b in data.get("branches", [])],
            integration_candidates=[
                IntegrationCandidate.from_dict(c) for c in data.get("integration_candidates", [])
            ],
            total_eis=data.get("total_eis", 0),
            needs_callable_analysis=data.get("needs_callable_analysis", False),
            is_stub=data.get("is_stub", False),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.branches:
            result["branches"] = [branch.to_dict() for branch in self.branches]
        if self.integration_candidates:
            result["integration_candidates"] = [
                candidate.to_dict() for candidate in self.integration_candidates
            ]
        if self.total_eis:
            result["total_eis"] = self.total_eis
        if self.needs_callable_analysis:
            result["needs_callable_analysis"] = self.needs_callable_analysis
        if self.is_stub:
            result["is_stub"] = self.is_stub
        return result


@dataclass
class CallableEntry:
    id: str
    kind: str
    name: str
    line_start: int
    line_end: int
    children: list["CallableEntry"] = field(default_factory=list)

    signature_info: CallableSignatureInfo = field(default_factory=CallableSignatureInfo)
    hierarchy_info: CallableHierarchyInfo = field(default_factory=CallableHierarchyInfo)
    analysis_info: CallableAnalysisInfo = field(default_factory=CallableAnalysisInfo)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallableEntry":
        return cls(
            id=data["id"],
            kind=data["kind"],
            name=data["name"],
            line_start=data.get("line_start", 0),
            line_end=data.get("line_end", 0),
            children=[cls.from_dict(c) for c in data.get("children", [])],
            signature_info=CallableSignatureInfo.from_dict(data.get("signature_info", {})),
            hierarchy_info=CallableHierarchyInfo.from_dict(data.get("hierarchy_info", {})),
            analysis_info=CallableAnalysisInfo.from_dict(data.get("analysis_info", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }

        if self.children:
            result["children"] = [c.to_dict() for c in self.children]

        signature_dict = self.signature_info.to_dict()
        if signature_dict:
            result["signature_info"] = signature_dict

        hierarchy_dict = self.hierarchy_info.to_dict()
        if hierarchy_dict:
            result["hierarchy_info"] = hierarchy_dict

        analysis_dict = self.analysis_info.to_dict()
        if analysis_dict:
            result["analysis_info"] = analysis_dict

        return result


class ExternalNodeType(Enum):
    """
    Type of external placeholder node (when category=EXTERNAL_NODE).

    STDLIB: Standard library call (sorted, dict, list, etc.)
    EXTLIB: External library call (requests, numpy, etc.)
    BOUNDARY: System boundary call (file I/O, network, database, etc.)
    UNKNOWN: Unresolved or unknown call type
    """
    STDLIB = "stdlib"
    EXTLIB = "extlib"
    BOUNDARY = "boundary"
    UNKNOWN = "unknown"

    @classmethod
    def from_integration_category(cls, category: str) -> Self:
        """
        Map integration category string to ExternalNodeType.

        Args:
            category: Integration category ('stdlib', 'extlib', 'boundary', 'unknown')

        Returns:
            Corresponding ExternalNodeType
        """
        mapping = {
            'stdlib': cls.STDLIB,
            'extlib': cls.EXTLIB,
            'boundary': cls.BOUNDARY,
            'unknown': cls.UNKNOWN,
        }
        return mapping.get(category, cls.UNKNOWN)


class ExternalTargetType(Enum):
    STDLIB = "EXTERNAL_STDLIB"
    EXTLIB = "EXTERNAL_EXTLIB"
    BOUNDARY = "EXTERNAL_BOUNDARY"
    UNKNOWN = "EXTERNAL_UNKNOWN"

    @classmethod
    def from_category(cls, category: str) -> Self:
        """Map integration category to external target type."""
        mapping = {
            'stdlib': cls.STDLIB,
            'extlib': cls.EXTLIB,
            'boundary': cls.BOUNDARY,
            'unknown': cls.UNKNOWN,
            'interunit': cls.UNKNOWN,  # Fallback for unresolved interunit
        }
        return mapping.get(category, cls.UNKNOWN)


class CallNodeType(Enum):
    """
    Type of concrete call node (when category=CALL_NODE).

    LOCAL: Callable in the same module
    INTERUNIT: Callable in a different module within the same project
    """
    LOCAL = "local"
    INTERUNIT = "interunit"


class NodeCategory(Enum):
    """
    High-level category of a CFG node.

    CALL_NODE: Concrete callable that can be traced into (local or interunit)
    EXTERNAL_NODE: Placeholder for external calls requiring fixtures/mocks
    """
    CALL_NODE = "call_node"
    EXTERNAL_NODE = "external_node"
