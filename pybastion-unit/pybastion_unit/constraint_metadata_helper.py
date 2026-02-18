"""
Constraint Metadata Extraction Helpers

Functions to extract constraint metadata when creating Branch objects
in enumerate_exec_items.py. These create BranchConstraint objects
from outcome strings and AST statements.
"""

import ast
import re
from typing import Any

from models import BranchConstraint
from smt_path_checker import ConstraintExtractor


def _compute_smt_expr(expr: str, polarity: bool | None) -> str | None:
    """
    Pre-compute Z3 SMT expression string for a condition constraint.

    Returns serialized Z3 expression string, or None if expression
    cannot be parsed.

    Args:
        expr: Condition expression string (e.g. 'value > 0')
        polarity: True/False for condition polarity, None for non-conditions

    Returns:
        String representation of Z3 expression, or None
    """
    if polarity is None:
        return None
    try:
        extractor = ConstraintExtractor()
        z3_expr = extractor.parse_condition_to_z3(expr, polarity)
        if z3_expr is not None:
            return str(z3_expr)
    except Exception:
        pass
    return None


def extract_operators_and_operands(node: ast.expr) -> tuple[list[str], list[str]]:
    """
    Extract operators and operands from a condition expression AST node.

    Operators: comparison ops (>, ==, is, in, is not, not in),
               boolean ops (and, or), unary ops (not)
    Operands:  variable names and literal values

    Args:
        node: AST expression node (e.g. stmt.test for if/while/assert)

    Returns:
        (operators, operands) - both as lists of strings, preserving
        first-seen order and deduplicating
    """
    operators: list[str] = []
    operands: list[str] = []

    _CMP_OP_MAP = {
        ast.Eq: '==', ast.NotEq: '!=',
        ast.Lt: '<', ast.LtE: '<=',
        ast.Gt: '>', ast.GtE: '>=',
        ast.Is: 'is', ast.IsNot: 'is not',
        ast.In: 'in', ast.NotIn: 'not in',
    }
    _BOOL_OP_MAP = {ast.And: 'and', ast.Or: 'or'}
    _UNARY_OP_MAP = {ast.Not: 'not', ast.USub: '-', ast.UAdd: '+', ast.Invert: '~'}

    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            for op in child.ops:
                sym = _CMP_OP_MAP.get(type(op))
                if sym and sym not in operators:
                    operators.append(sym)

        elif isinstance(child, ast.BoolOp):
            sym = _BOOL_OP_MAP.get(type(child.op))
            if sym and sym not in operators:
                operators.append(sym)

        elif isinstance(child, ast.UnaryOp):
            sym = _UNARY_OP_MAP.get(type(child.op))
            if sym and sym not in operators:
                operators.append(sym)

        elif isinstance(child, ast.Name):
            if child.id not in ('True', 'False', 'None') and child.id not in operands:
                operands.append(child.id)

        elif isinstance(child, ast.Constant):
            val = repr(child.value)
            if val not in operands:
                operands.append(val)

    return operators, operands


def extract_variables_from_ast_node(node: ast.AST) -> set[str]:
    """
    Extract all variable names referenced in an AST node.

    Args:
        node: AST node to analyze

    Returns:
        Set of variable names (identifiers)
    """
    variables = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            # Exclude special names like True, False, None
            if child.id not in ('True', 'False', 'None'):
                variables.add(child.id)

    return variables


def parse_outcome_to_constraint(
        outcome: str,
        stmt: ast.stmt,
        branch_id: str = "",
        line: int = 0,
        call_node: ast.Call | None = None
) -> BranchConstraint | None:
    """
    Parse an outcome string and statement to create a BranchConstraint.

    This is called when creating Branch objects to populate the
    constraint field.

    Args:
        outcome: The outcome string (e.g., "value > 0 is true → enters if block")
        stmt: The AST statement this outcome came from
        branch_id: Branch/EI ID for traceability
        line: Line number in source code

    Returns:
        BranchConstraint object if constraint exists, None otherwise
    """
    # Split into condition and result if present
    if ' → ' in outcome:
        condition, result = outcome.split(' → ', 1)
    else:
        condition = 'executes'
        result = outcome

    outcome_lower = outcome.lower()
    result_lower = result.lower()

    # Will build these up based on constraint type
    constraint_type: str | None = None
    constraint_expr: str | None = None
    constraint_polarity: bool | None = None
    variables_read: set[str] = set()
    variables_written: set[str] = set()
    operators: list[str] = []
    operands: list[str] = []
    smt_expr: str | None = None
    metadata: dict[str, Any] = {}

    # Detect constraint type and extract metadata

    # 1. CONDITION (if/while/assert)
    if isinstance(stmt, (ast.If, ast.While, ast.Assert)):
        if 'is true' in condition or 'is false' in condition:
            # Extract the condition expression
            if isinstance(stmt, ast.If):
                expr = ast.unparse(stmt.test)
            elif isinstance(stmt, ast.While):
                expr = ast.unparse(stmt.test)
            elif isinstance(stmt, ast.Assert):
                expr = ast.unparse(stmt.test)
            else:
                expr = condition

            constraint_type = 'condition'
            constraint_expr = expr
            constraint_polarity = 'is true' in condition
            variables_read = extract_variables_from_ast_node(stmt.test)
            operators, operands = extract_operators_and_operands(stmt.test)
            smt_expr = _compute_smt_expr(expr, constraint_polarity)

    # 2. ITERATION (for/while loop outcomes)
    elif isinstance(stmt, (ast.For, ast.While)):
        if 'iterations' in outcome_lower or 'completes' in outcome_lower or 'breaks' in outcome_lower:
            constraint_type = 'iteration'
            if isinstance(stmt, ast.For):
                constraint_expr = f"for {ast.unparse(stmt.target)} in {ast.unparse(stmt.iter)}"
                variables_read = extract_variables_from_ast_node(stmt.iter)
                variables_written = extract_variables_from_ast_node(stmt.target)
            else:
                constraint_expr = ast.unparse(stmt.test)
                variables_read = extract_variables_from_ast_node(stmt.test)

    # 3. EXCEPTION (try/except, raise, or exception propagation)
    elif isinstance(stmt, ast.Try):
        constraint_type = 'exception'
        if 'raises' in outcome_lower:
            # Try to extract exception type
            match = re.search(r'raises (\w+)', result)
            if match:
                metadata['exception_types'] = [match.group(1)]

    # 4. OPERATION (function/method calls)
    elif 'executes →' in outcome and 'succeeds' in result_lower:
        # Extract operation from condition
        # Format: "executes → validate(data) succeeds"
        match = re.match(r'executes → (.+) succeeds', outcome)
        if match:
            operation = match.group(1)
            constraint_type = 'operation'
            constraint_expr = operation

            # Try to parse and extract variables
            try:
                expr_ast = ast.parse(operation, mode='eval')
                variables_read = extract_variables_from_ast_node(expr_ast)
            except:
                variables_read = set()

    # 5. EXCEPTION PROPAGATION (from operation)
    elif 'raises exception' in result_lower and 'propagates' in result_lower:
        # Extract operation from condition
        match = re.match(r'(.+) raises exception', condition)
        if match:
            operation = match.group(1)
            constraint_type = 'exception'
            constraint_expr = operation

    # 6. ASSIGNMENT
    elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        constraint_type = 'assignment'

        # Extract written variables
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
            written = set()
            for target in targets:
                written.update(extract_variables_from_ast_node(target))
            variables_written = written

            # Extract read variables from value
            if stmt.value:
                variables_read = extract_variables_from_ast_node(stmt.value)
        elif isinstance(stmt, ast.AnnAssign):
            variables_written = extract_variables_from_ast_node(stmt.target)
            if stmt.value:
                variables_read = extract_variables_from_ast_node(stmt.value)
        elif isinstance(stmt, ast.AugAssign):
            variables_written = extract_variables_from_ast_node(stmt.target)
            variables_read = extract_variables_from_ast_node(stmt.value)

    # 7. MATCH CASE
    elif isinstance(stmt, ast.Match):
        if 'match case' in outcome_lower:
            constraint_type = 'match_case'
            # Extract case pattern from outcome
            match = re.search(r'match case \d+: (.+?)(?:$| →)', outcome)
            if match:
                constraint_expr = match.group(1)

    operation_target = ast.unparse(call_node.func) if call_node is not None else None

    # If we found constraint data, create BranchConstraint
    if constraint_type and constraint_expr:
        return BranchConstraint(
            expr=constraint_expr,
            polarity=constraint_polarity,
            constraint_type=constraint_type,
            variables_read=variables_read,
            variables_written=variables_written,
            branch_id=branch_id,
            line=line,
            operators=operators,
            operands=operands,
            smt_expr=smt_expr,
            operation_target=operation_target,
            metadata=metadata
        )

    if operation_target:
        return BranchConstraint(
            expr=operation_target,
            polarity=None,
            constraint_type='operation',
            operation_target=operation_target,
            branch_id=branch_id,
            line=line
        )

    return None


def populate_constraint_relationships(branches: list) -> None:
    """
    Post-enumeration pass: populate implies and excludes on BranchConstraint objects.

    Must be called after all branches for a callable are built.
    Mutates branches in place.

    Rules:
      excludes: same expr, same line, opposite polarity — direct if/else pairs
      implies:  numeric comparisons where one bound is strictly stronger
                e.g. x > 10 implies x > 5, x > 0; excludes x < 10, x <= 10

    Args:
        branches: Complete list of Branch objects for one callable
    """
    # Only work on branches that have condition-type constraints with polarity
    condition_branches = [
        b for b in branches
        if b.constraint is not None
           and b.constraint.constraint_type == 'condition'
           and b.constraint.polarity is not None
    ]

    # --- EXCLUDES ---
    # Group by (expr, line) — branches sharing both are mutual alternatives
    from collections import defaultdict
    by_expr_line: dict[tuple[str, int], list] = defaultdict(list)
    for b in condition_branches:
        key = (b.constraint.expr, b.constraint.line)
        by_expr_line[key].append(b)

    for group in by_expr_line.values():
        if len(group) < 2:
            continue
        # Every branch in the group excludes every other
        for b in group:
            for other in group:
                if other.id != b.id and other.constraint.polarity != b.constraint.polarity:
                    if other.id not in b.constraint.excludes:
                        b.constraint.excludes.append(other.id)

    # --- IMPLIES ---
    # Only applies to numeric single-variable comparisons: x > N, x >= N, x < N, x <= N
    # For each such branch, find other branches on the same variable that are
    # logically weaker (i.e. must also be true whenever this one is true)
    numeric_ops = {'>', '>=', '<', '<='}

    def parse_numeric_constraint(b) -> tuple[str, str, int | float] | None:
        """
        Extract (variable, operator, value) from a simple numeric condition.
        Returns None if not a simple numeric comparison.
        """
        c = b.constraint
        if not c.operators or not c.operands:
            return None
        # Must be a single numeric operator
        ops = [o for o in c.operators if o in numeric_ops]
        if len(ops) != 1:
            return None
        op = ops[0]
        # Must have exactly one variable and one numeric constant
        # operands list is ordered by first-seen in ast.walk
        variables = [o for o in c.operands if not _is_numeric_literal(o)]
        constants = [o for o in c.operands if _is_numeric_literal(o)]
        if len(variables) != 1 or len(constants) != 1:
            return None
        try:
            val = int(constants[0]) if '.' not in constants[0] else float(constants[0])
        except ValueError:
            return None
        # Adjust operator if polarity is False (negated condition)
        if not c.polarity:
            op = _negate_op(op)
        return variables[0], op, val

    def _is_numeric_literal(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _negate_op(op: str) -> str:
        return {'>': '<=', '>=': '<', '<': '>=', '<=': '>'}[op]

    def _implies(op1: str, val1: int | float, op2: str, val2: int | float) -> bool:
        """
        Does (var op1 val1) logically imply (var op2 val2) for all values of var?

        Examples:
          x > 10 implies x > 5   (True: any x > 10 is also > 5)
          x > 10 implies x > 10  (True: same)
          x > 5  implies x > 10  (False: x=7 satisfies x>5 but not x>10)
          x > 10 implies x >= 10 (True)
          x >= 10 implies x > 9  (True for integers, approximate for floats)
        """
        # Both constraints on same side of number line
        if op1 in ('>', '>=') and op2 in ('>', '>='):
            # x > val1 implies x > val2 iff val1 >= val2 (strictly stronger lower bound)
            if op1 == '>' and op2 == '>':
                return val1 >= val2
            if op1 == '>' and op2 == '>=':
                return val1 >= val2
            if op1 == '>=' and op2 == '>':
                return val1 > val2
            if op1 == '>=' and op2 == '>=':
                return val1 >= val2
        if op1 in ('<', '<=') and op2 in ('<', '<='):
            # x < val1 implies x < val2 iff val1 <= val2
            if op1 == '<' and op2 == '<':
                return val1 <= val2
            if op1 == '<' and op2 == '<=':
                return val1 <= val2
            if op1 == '<=' and op2 == '<':
                return val1 < val2
            if op1 == '<=' and op2 == '<=':
                return val1 <= val2
        return False

    # Parse all branches into numeric form where possible
    numeric: list[tuple[object, str, str, int | float]] = []  # (branch, var, op, val)
    for b in condition_branches:
        parsed = parse_numeric_constraint(b)
        if parsed:
            var, op, val = parsed
            numeric.append((b, var, op, val))

    # Group by variable name — only compare branches on the same variable
    by_var: dict[str, list] = defaultdict(list)
    for entry in numeric:
        by_var[entry[1]].append(entry)

    for var_entries in by_var.values():
        for b1, var1, op1, val1 in var_entries:
            for b2, var2, op2, val2 in var_entries:
                if b1.id == b2.id:
                    continue
                if _implies(op1, val1, op2, val2):
                    if b2.id not in b1.constraint.implies:
                        b1.constraint.implies.append(b2.id)


def enrich_outcome_with_constraint(
        outcome: str,
        call_node: ast.Call | None,
        stmt: ast.stmt,
        ei_id: str,
        line: int
) -> tuple[str, str, BranchConstraint | None]:
    """
    Enrich an outcome string with the extracted constraint.

    Args:
        outcome: Outcome string from decompose function
        call_node: AST Call node if available, otherwise None
        stmt: AST statement
        ei_id: Execution item ID
        line: Line number

    Returns:
        (condition, result, constraint)
    """
    # Split outcome into condition and result
    if ' → ' in outcome:
        condition, result = outcome.split(' → ', 1)
    else:
        condition = 'executes'
        result = outcome.replace('executes: ', '')

    # Extract constraint
    constraint = parse_outcome_to_constraint(outcome, stmt, branch_id=ei_id, line=line, call_node=call_node)

    return condition, result, constraint
