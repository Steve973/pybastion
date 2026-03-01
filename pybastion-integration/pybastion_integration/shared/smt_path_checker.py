"""
SMT-Based Path Feasibility Checker

Uses Z3 theorem prover to determine if execution paths are logically feasible
and generates witness values (test data) for feasible paths.
"""

import ast
from typing import Any
from dataclasses import dataclass

from z3 import (
    Solver, sat, unsat,
    Int, Real, Bool, String,
    And, Or, Not,
    IntVal, RealVal, BoolVal, StringVal
)

from pybastion_unit.shared.models import Branch


@dataclass
class PathFeasibilityResult:
    """Result of path feasibility analysis."""

    is_feasible: bool
    reason: str | None = None
    witness_values: dict[str, Any] | None = None
    constraint_count: int = 0
    solver_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'is_feasible': self.is_feasible,
            'reason': self.reason,
            'witness_values': self.witness_values,
            'constraint_count': self.constraint_count,
            'solver_time_ms': self.solver_time_ms
        }


class ConstraintExtractor:
    """Extract Z3 constraints from Python condition strings."""

    def __init__(self):
        self.variables = {}  # name -> Z3 variable
        self.type_hints = {}  # name -> inferred type ('int', 'bool', 'str')

    def get_or_create_variable(self, name: str, inferred_type: str = 'int'):
        """
        Get or create a Z3 variable with appropriate type.

        Args:
            name: Variable name
            inferred_type: 'int', 'real', 'bool', 'str'

        Returns:
            Z3 variable
        """
        if name in self.variables:
            return self.variables[name]

        # Create variable based on type
        if inferred_type == 'bool':
            var = Bool(name)
        elif inferred_type == 'real':
            var = Real(name)
        elif inferred_type == 'str':
            var = String(name)
        else:  # default to int
            var = Int(name)

        self.variables[name] = var
        self.type_hints[name] = inferred_type

        return var

    def parse_condition_to_z3(self, condition_expr: str, polarity: bool):
        """
        Parse a Python condition expression to Z3 constraint.

        Args:
            condition_expr: Python expression string (e.g., "x > 0")
            polarity: True if condition should be true, False if false

        Returns:
            Z3 constraint or None if parsing fails
        """
        try:
            tree = ast.parse(condition_expr, mode='eval')
            constraint = self._ast_to_z3(tree.body)

            if constraint is None:
                return None

            # Apply polarity (negate if False)
            if not polarity:
                constraint = Not(constraint)

            return constraint

        except Exception as e:
            # If we can't parse it, we can't create a constraint
            # This is okay - we'll just skip this constraint
            return None

    def _ast_to_z3(self, node: ast.AST):
        """
        Convert AST node to Z3 constraint.

        Supports:
        - Comparisons: <, <=, >, >=, ==, !=
        - Boolean operations: and, or, not
        - Arithmetic: +, -, *, //, %
        - Common functions: len(), abs(), isinstance()
        """

        # Comparison: x > 0, y == 5
        if isinstance(node, ast.Compare):
            return self._handle_compare(node)

        # Boolean operations: x and y, x or y
        elif isinstance(node, ast.BoolOp):
            return self._handle_boolop(node)

        # Unary operations: not x, -x
        elif isinstance(node, ast.UnaryOp):
            return self._handle_unaryop(node)

        # Binary operations: x + y, x * 2
        elif isinstance(node, ast.BinOp):
            return self._handle_binop(node)

        # Function calls: len(x), isinstance(x, int)
        elif isinstance(node, ast.Call):
            return self._handle_call(node)

        # Variable reference: x, value
        elif isinstance(node, ast.Name):
            return self.get_or_create_variable(node.id, 'bool')

        # Constants: 0, True, "hello"
        elif isinstance(node, ast.Constant):
            return self._handle_constant(node)

        # Unsupported node type
        else:
            return None

    def _handle_compare(self, node: ast.Compare):
        """Handle comparison operations."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return None

        op = node.ops[0]

        # Infer numeric type for variables in numeric comparisons
        numeric_ops = (ast.Gt, ast.GtE, ast.Lt, ast.LtE)
        if isinstance(op, numeric_ops):
            if isinstance(node.left, ast.Name):
                self.get_or_create_variable(node.left.id, 'int')
            if isinstance(node.comparators[0], ast.Name):
                self.get_or_create_variable(node.comparators[0].id, 'int')

        left = self._ast_to_z3(node.left)
        right = self._ast_to_z3(node.comparators[0])

        if left is None or right is None:
            return None

        if isinstance(op, ast.Gt):
            return left > right
        elif isinstance(op, ast.GtE):
            return left >= right
        elif isinstance(op, ast.Lt):
            return left < right
        elif isinstance(op, ast.LtE):
            return left <= right
        elif isinstance(op, ast.Eq):
            return left == right
        elif isinstance(op, ast.NotEq):
            return left != right
        elif isinstance(op, ast.Is):
            # Handle 'is None' specially
            if isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                # For now, treat 'x is None' as a boolean constraint
                # In a real implementation, it might need Option types
                return None
            return left == right
        elif isinstance(op, ast.IsNot):
            if isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                return None
            return left != right
        else:
            return None

    def _handle_boolop(self, node: ast.BoolOp):
        """Handle boolean operations (and, or)."""
        operands = [self._ast_to_z3(val) for val in node.values]
        operands = [op for op in operands if op is not None]

        if not operands:
            return None

        if isinstance(node.op, ast.And):
            return And(*operands)
        elif isinstance(node.op, ast.Or):
            return Or(*operands)
        else:
            return None

    def _handle_unaryop(self, node: ast.UnaryOp):
        """Handle unary operations (not, -)."""
        operand = self._ast_to_z3(node.operand)

        if operand is None:
            return None

        if isinstance(node.op, ast.Not):
            return Not(operand)
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            return None

    def _handle_binop(self, node: ast.BinOp):
        """Handle binary arithmetic operations."""
        left = self._ast_to_z3(node.left)
        right = self._ast_to_z3(node.right)

        if left is None or right is None:
            return None

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.FloorDiv):
            # Z3 integer division
            return left / right
        elif isinstance(node.op, ast.Mod):
            return left % right
        else:
            # Unsupported operation
            return None

    def _handle_call(self, node: ast.Call):
        """Handle function calls like len(), abs(), isinstance()."""
        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        else:
            return None

        # len(x) - treat as integer variable
        if func_name == 'len':
            if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                var_name = f"len_{node.args[0].id}"
                return self.get_or_create_variable(var_name, 'int')

        # abs(x) - absolute value
        elif func_name == 'abs':
            if len(node.args) == 1:
                arg = self._ast_to_z3(node.args[0])
                if arg is not None:
                    # Z3 doesn't have built-in abs, but we can constrain it
                    # For now, just return the arg (approximation)
                    return arg

        # isinstance(x, type) - treat as boolean
        elif func_name == 'isinstance':
            # Can't really model isinstance in Z3, skip it
            return None

        # callable(x) - treat as boolean
        elif func_name == 'callable':
            return None

        return None

    def _handle_constant(self, node: ast.Constant):
        """Handle constant values."""
        value = node.value

        if isinstance(value, bool):
            return BoolVal(value)
        elif isinstance(value, int):
            return IntVal(value)
        elif isinstance(value, float):
            return RealVal(value)
        elif isinstance(value, str):
            return StringVal(value)
        elif value is None:
            # Can't directly model None in Z3
            return None
        else:
            return None


class PathFeasibilityChecker:
    """Check path feasibility using SMT solving."""

    def __init__(self, timeout_ms: int = 5000):
        """
        Initialize path feasibility checker.

        Args:
            timeout_ms: Solver timeout in milliseconds
        """
        self.timeout_ms = timeout_ms

    def check_path(self, path_branches: list[Branch]) -> PathFeasibilityResult:
        """
        Check if a path through the CFG is feasible.

        Args:
            path_branches: List of Branch objects representing the path

        Returns:
            PathFeasibilityResult with feasibility status and witness values
        """
        import time
        start_time = time.time()

        solver = Solver()
        solver.set("timeout", self.timeout_ms)

        extractor = ConstraintExtractor()
        constraint_count = 0

        # Build constraints from path
        for branch in path_branches:
            # Only process condition-type branches
            if branch.constraint is None or branch.constraint.constraint_type != 'condition':
                continue

            if not branch.constraint.expr:
                continue

            # Parse condition to Z3
            constraint = extractor.parse_condition_to_z3(
                branch.constraint.expr,
                branch.constraint.polarity if branch.constraint.polarity is not None else True
            )

            if constraint is not None:
                solver.add(constraint)
                constraint_count += 1

        # Check satisfiability
        result = solver.check()
        elapsed_ms = (time.time() - start_time) * 1000

        if result == sat:
            # Path is feasible - extract witness values
            model = solver.model()
            witness = {}

            for var_name, var in extractor.variables.items():
                try:
                    value = model[var]
                    # Convert Z3 value to Python value
                    if value is not None:
                        witness[var_name] = self._z3_value_to_python(value)
                except:
                    pass

            return PathFeasibilityResult(
                is_feasible=True,
                reason=None,
                witness_values=witness if witness else None,
                constraint_count=constraint_count,
                solver_time_ms=elapsed_ms
            )

        elif result == unsat:
            # Path is infeasible
            return PathFeasibilityResult(
                is_feasible=False,
                reason="Constraints are unsatisfiable (conflicting conditions)",
                witness_values=None,
                constraint_count=constraint_count,
                solver_time_ms=elapsed_ms
            )

        else:  # unknown (timeout or too complex)
            # Treat unknown as feasible (conservative approach)
            return PathFeasibilityResult(
                is_feasible=True,
                reason=f"Solver timeout or complexity limit (treated as feasible)",
                witness_values=None,
                constraint_count=constraint_count,
                solver_time_ms=elapsed_ms
            )

    def _z3_value_to_python(self, z3_val) -> Any:
        """Convert Z3 value to Python value."""
        val_str = str(z3_val)

        # Try to convert to int
        try:
            return int(val_str)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(val_str)
        except ValueError:
            pass

        # Check for boolean
        if val_str == 'True':
            return True
        elif val_str == 'False':
            return False

        # Return as string
        return val_str


def filter_feasible_paths(
        all_paths: list[list[str]],
        branches: list[Branch],
        timeout_ms: int = 5000,
        verbose: bool = False,
        unit_name: str = ''
) -> tuple[list[list[str]], dict[str, PathFeasibilityResult]]:
    """
    Filter paths to only feasible ones using SMT solving.

    Args:
        all_paths: List of paths (each path is list of EI IDs)
        branches: List of all Branch objects
        timeout_ms: Solver timeout in milliseconds
        verbose: Print progress
        unit_name: Name of unit being processed (for verbose output)

    Returns:
        (feasible_paths, results_by_path_id)
    """
    checker = PathFeasibilityChecker(timeout_ms=timeout_ms)

    # Create branch lookup
    branch_map = {b.id: b for b in branches}

    feasible_paths = []
    results = {}
    total_paths = len(all_paths)

    for idx, path in enumerate(all_paths):
        if verbose:
            pct = int((idx + 1) / total_paths * 100)
            print(f"\r  progress: {pct}% ({idx + 1}/{total_paths} paths)",
                  end='', flush=True)

        # Get branches for this path
        path_branches = [branch_map[ei_id] for ei_id in path if ei_id in branch_map]

        # Check feasibility
        result = checker.check_path(path_branches)

        # Store result
        path_id = '->'.join(path)
        results[path_id] = result

        # Add to feasible list if feasible
        if result.is_feasible:
            feasible_paths.append(path)

    print("\r                                                          \r",
          end="", flush=True)

    return feasible_paths, results