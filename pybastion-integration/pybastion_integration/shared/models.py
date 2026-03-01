from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from typing_extensions import Self


class NodeCategory(Enum):
    """
    High-level category of a CFG node.

    CALL_NODE: Concrete callable that can be traced into (local or interunit)
    EXTERNAL_NODE: Placeholder for external calls requiring fixtures/mocks
    """
    CALL_NODE = "call_node"
    EXTERNAL_NODE = "external_node"


class CallNodeType(Enum):
    """
    Type of concrete call node (when category=CALL_NODE).

    LOCAL: Callable in the same module
    INTERUNIT: Callable in a different module within the same project
    """
    LOCAL = "local"
    INTERUNIT = "interunit"


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
    Example: 'x > 10' implies 'x > 5'
    Used for constraint propagation and optimization.
    """

    excludes: list[str] = field(default_factory=list)
    """
    Branch IDs that this constraint excludes (mutually exclusive).
    Example: 'x > 10' excludes 'x < 5'
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
class Branch:
    """
    Execution Item (EI) representation with constraint tracking.

    Contract between stage2_enumerate_exec_items.py and downstream stages.
    Called "Branch" in current code but represents an Execution Item.

    Enhanced with constraint metadata to enable path feasibility analysis.
    """
    # Core fields
    id: str
    """Unique identifier for the EI."""

    line: int
    """Line number of the EI statement."""

    condition: str
    """Condition expression for the EI."""

    outcome: str
    """
    Describes the EI outcome.
    """

    decorators: list[dict[str, Any]] = field(default_factory=list)
    """
    Statement-level decorators (e.g., for feature flow tracing).
    """

    constraint: BranchConstraint | None = None
    """
    Constraint object containing all constraint-related metadata.
    Replaces scattered constraint_type, constraint_expr, constraint_polarity fields.
    """

    is_terminal: bool = False
    """
    Whether this EI terminates execution flow:
    - Returns (explicit return statement)
    - Raises (explicit raise or unhandled exception propagation)
    - Breaks/continues (loop control flow)
    Terminal EIs have no successors in the CFG.
    """

    terminates_via: str | None = None
    """
    How this EI terminates, if is_terminal=True:
    - 'return': explicit return
    - 'raise': explicit raise
    - 'exception': unhandled exception propagation
    - 'break': loop break
    - 'continue': loop continue
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """
    Additional metadata for specialized analysis:
    - 'loop_header': bool - whether this is a loop entry point
    - 'loop_id': str - identifier for the containing loop
    - 'mutually_exclusive_with': set[str] - EI IDs that are alternatives
    """

    def __post_init__(self) -> None:
        """Validate branch structure and derive terminal status."""
        if not self.id:
            raise ValueError("Branch ID cannot be empty")
        if self.line <= 0:
            raise ValueError(f"Invalid line number: {self.line}")

        # Auto-detect terminal status from outcome if not explicitly set
        if not self.is_terminal:
            self.is_terminal, self.terminates_via = self._detect_terminal_status()

    def _detect_terminal_status(self) -> tuple[bool, str | None]:
        """
        Auto-detect if this EI is terminal based on outcome text.

        Returns:
            (is_terminal, terminates_via)
        """
        outcome_lower = self.outcome.lower()

        # Check for explicit return
        if any(indicator in outcome_lower for indicator in [
            '→ returns',
            'returns ',
            '→ return ',
            'return value'
        ]):
            return True, 'return'

        # Check for explicit raise
        if any(indicator in outcome_lower for indicator in [
            '→ raises',
            'raises ',
            '→ raise ',
        ]):
            return True, 'raise'

        # Check for exception propagation
        if 'exception propagates' in outcome_lower:
            return True, 'exception'

        # Check for loop control
        if '→ breaks' in outcome_lower or 'break' in outcome_lower:
            return True, 'break'

        if '→ continues' in outcome_lower:
            return True, 'continue'

        return False, None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Branch:
        """Parse from inventory dict format."""
        return cls(
            id=data['id'],
            line=data['line'],
            condition=data['condition'],
            outcome=data['outcome'],
            decorators=data.get('decorators', []),
            constraint=BranchConstraint.from_dict(data['constraint']) if data.get('constraint') else None,
            is_terminal=data.get('is_terminal', False),
            terminates_via=data.get('terminates_via'),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {
            'id': self.id,
            'line': self.line,
            'condition': self.condition,
            'outcome': self.outcome,
        }

        if self.decorators:
            result['decorators'] = self.decorators

        # Include constraint object if present
        if self.constraint is not None:
            result['constraint'] = self.constraint.to_dict()

        if self.is_terminal:
            result['is_terminal'] = self.is_terminal

        if self.terminates_via is not None:
            result['terminates_via'] = self.terminates_via

        if self.metadata:
            result['metadata'] = self.metadata

        return result

    def to_ledger_ei_spec(self) -> dict[str, str | bool | list[str] | dict[str, Any]]:
        """
        Transform to ledger EiSpec format.

        Ledger format includes constraint metadata for downstream path analysis
        and test generation. Uses snake_case for consistency.
        """
        result: dict[str, Any] = {
            'id': self.id,
            'condition': self.condition,
            'outcome': self.outcome,
        }

        # Include constraint object if present
        if self.constraint is not None:
            result['constraint'] = self.constraint.to_dict()

        if self.decorators:
            result['decorators'] = self.decorators

        if self.is_terminal:
            result['is_terminal'] = self.is_terminal
            if self.terminates_via:
                result['terminates_via'] = self.terminates_via

        if self.metadata:
            result['metadata'] = self.metadata

        return result

    def conflicts_with(self, other: Branch) -> bool:
        """
        Check if this branch's constraints conflict with another branch.

        Used for path feasibility analysis to detect impossible paths.

        Returns:
            True if constraints are mutually exclusive
        """
        # Use constraint object if available
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
        # Use constraint object if available
        if self.constraint is not None:
            return self.constraint.get_signature()

        return ''

    def is_alternative_to(self, other: Branch) -> bool:
        """
        Check if this branch is a mutually exclusive alternative to another.

        Branches are alternatives if they:
        1. Are on the same line (same decision point)
        2. Have the same condition but different outcomes

        Returns:
            True if branches are mutually exclusive alternatives
        """
        return (self.line == other.line and
                self.condition == other.condition and
                self.outcome != other.outcome)

    def __repr__(self) -> str:
        """Enhanced repr showing constraint information."""
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

        if self.is_terminal:
            parts.append(f"terminal={self.terminates_via!r}")

        return ', '.join(parts) + ')'