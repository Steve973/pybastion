#!/usr/bin/env python3
"""
Execution Item Enumerator - Complete Python Statement Coverage

Enumerates all execution items (EIs) in Python source code.
Outputs YAML format for integration with pipeline.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Callable

import yaml

from callable_id_generation import generate_function_id, generate_ei_id, generate_assignment_id
from models import Branch


def load_callable_inventory(filepath: Path | None) -> dict[str, str]:
    """
    Load callable inventory file (FQN:ID pairs).

    Returns:
        Dict mapping fully qualified names to callable IDs
    """
    inventory = {}
    print(f"inventory file path: {filepath}")
    if not filepath or not filepath.exists():
        return inventory

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            fqn, callable_id = line.split(':', 1)
            inventory[fqn] = callable_id

    return inventory


def derive_fqn_from_path(filepath: Path, source_root: Path | None) -> str:
    """
    Convert file path to module FQN.

    Example:
        src/project/model/keys.py -> project.model.keys
    """
    if not source_root:
        return filepath.stem

    try:
        relative = filepath.relative_to(source_root)
    except ValueError:
        # filepath not relative to source_root, use name only
        return filepath.stem

    parts = list(relative.parts[:-1]) + [relative.stem]

    # Remove __init__ from end
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


# ============================================================================
# Operation Extraction
# ============================================================================

def extract_all_operations(node: ast.AST) -> list[ast.Call]:
    """
    Extract ALL Call nodes from an AST in execution order.

    For nested/chained calls like Path(fetch(url)).resolve():
    - Returns: [fetch(url), Path(...), Path(...).resolve()]
    - Execution order: innermost first (by depth), then left-to-right

    Returns:
        List of ast.Call nodes in execution order
    """
    operations = []

    # Collect all Call nodes with their depth
    def collect_calls_with_depth(n: ast.AST, depth: int = 0) -> None:
        """Recursively collect calls with their nesting depth."""
        if isinstance(n, ast.Call):
            # Record this call with its depth and position
            operations.append((n, depth, n.lineno, n.col_offset))

        # Recurse into children with increased depth
        for child in ast.iter_child_nodes(n):
            collect_calls_with_depth(child, depth + 1)

    collect_calls_with_depth(node)

    # Sort by: depth (deepest/innermost first), then line, then column
    # This gives us execution order: inner calls before outer calls
    operations.sort(key=lambda x: (-x[1], x[2], x[3]))

    # Return just the Call nodes
    return [op[0] for op in operations]


# ============================================================================
# Statement Decomposers
# ============================================================================

def decompose_if(stmt: ast.If, source_lines: list[str]) -> list[str]:
    """
    If statement: EIs for all operations in condition, then 2 EIs for true/false.

    For: if foo() and bar():
    Returns:
    - "executes → foo() succeeds"
    - "foo() raises exception → exception propagates"
    - "foo() returns true → continues to evaluate right side"
    - "foo() returns false → combined condition is false"
    - "executes → bar() succeeds"
    - "bar() raises exception → exception propagates"
    - "combined condition is true → enters if block"
    - "combined condition is false → continues"
    """
    eis = []

    # Extract all operations from the condition
    operations = extract_all_operations(stmt.test)

    # Generate EIs for each operation in the condition
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    # Now generate the condition true/false EIs
    condition = ast.unparse(stmt.test)

    # Check what's inside the if body for better descriptions
    if stmt.body:
        first_stmt = stmt.body[0]

        # If it raises, be specific
        if isinstance(first_stmt, ast.Raise):
            exc = ast.unparse(first_stmt.exc) if first_stmt.exc else "exception"
            eis.extend([
                f"{condition} is true → raises {exc}",
                f"{condition} is false → continues"
            ])
            return eis

        # If it returns, be specific
        if isinstance(first_stmt, ast.Return):
            ret_val = ast.unparse(first_stmt.value) if first_stmt.value else "None"
            eis.extend([
                f"{condition} is true → returns {ret_val}",
                f"{condition} is false → continues"
            ])
            return eis

    # Generic if
    eis.extend([
        f"{condition} is true → enters if block",
        f"{condition} is false → continues"
    ])
    return eis


def decompose_for(stmt: ast.For, source_lines: list[str]) -> list[str]:
    """
    For loop: EIs for operations in iterable, then 2 EIs (0 iterations, ≥1 iterations).
    For-else: EIs for operations, then 3 EIs (empty, completes without break, breaks).
    """
    eis = []

    # Extract all operations from the iterable expression
    operations = extract_all_operations(stmt.iter)

    # Generate EIs for each operation in the iterable
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    target = ast.unparse(stmt.target)
    iter_expr = ast.unparse(stmt.iter)

    if stmt.orelse:
        # For-else pattern
        eis.extend([
            f"for {target} in {iter_expr}: 0 iterations → else executes",
            f"for {target} in {iter_expr}: completes without break → else executes",
            f"for {target} in {iter_expr}: breaks → else skipped"
        ])
    else:
        # Regular for loop
        eis.extend([
            f"for {target} in {iter_expr}: 0 iterations",
            f"for {target} in {iter_expr}: ≥1 iterations"
        ])

    return eis


def decompose_while(stmt: ast.While, source_lines: list[str]) -> list[str]:
    """
    While loop: EIs for operations in condition, then 2 EIs (initially false, initially true).
    While-else: EIs for operations, then 3 EIs (initially false → else, completes → else, breaks → no else).
    """
    eis = []

    # Extract all operations from the condition
    operations = extract_all_operations(stmt.test)

    # Generate EIs for each operation in the condition
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    condition = ast.unparse(stmt.test)

    if stmt.orelse:
        # While-else pattern
        eis.extend([
            f"while {condition}: initially false → else executes",
            f"while {condition}: completes without break → else executes",
            f"while {condition}: breaks → else skipped"
        ])
    else:
        # Regular while loop
        eis.extend([
            f"while {condition}: initially false → 0 iterations",
            f"while {condition}: initially true → ≥1 iterations"
        ])

    return eis


def decompose_try(stmt: ast.Try, source_lines: list[str]) -> list[str]:
    """
    Try/except: EIs for exception types, then 1 + N EIs (success + each handler).
    Try/except/else: adds 1 EI for else block.
    Finally always executes, doesn't create separate EI for branching.
    """
    eis = []

    # Extract operations from exception type specifications
    for handler in stmt.handlers:
        if handler.type:
            operations = extract_all_operations(handler.type)
            for op in operations:
                op_str = ast.unparse(op)
                eis.append(f"executes → {op_str} succeeds")
                eis.append(f"{op_str} raises exception → exception propagates")

    eis.append("try block executes successfully")

    # Add EI for each exception handler
    for handler in stmt.handlers:
        if handler.type:
            exc_type = ast.unparse(handler.type)
            eis.append(f"raises {exc_type} → enters except handler")
        else:
            eis.append(f"raises exception → enters except handler")

    # Note: else block executes only if try succeeds (already covered by first EI)
    # Note: finally block always executes (not a branching point)

    return eis


def decompose_with(stmt: ast.With, source_lines: list[str]) -> list[str]:
    """With statement: EIs for context expressions, then 2 EIs (enters successfully, raises on entry)."""
    eis = []

    # Extract operations from all context expressions
    for item in stmt.items:
        operations = extract_all_operations(item.context_expr)
        for op in operations:
            op_str = ast.unparse(op)
            eis.append(f"executes → {op_str} succeeds")
            eis.append(f"{op_str} raises exception → exception propagates")

    contexts = [ast.unparse(item.context_expr) for item in stmt.items]
    context_str = ', '.join(contexts)
    eis.extend([
        f"with {context_str}: enters successfully",
        f"with {context_str}: raises exception on entry"
    ])

    return eis


def decompose_match(stmt: ast.Match, source_lines: list[str]) -> list[str]:
    """Match statement: EIs for subject expression, then N EIs (one per case)."""
    eis = []

    # Extract operations from the subject expression
    operations = extract_all_operations(stmt.subject)
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    for i, case in enumerate(stmt.cases):
        pattern = ast.unparse(case.pattern)

        # Check if case body contains a return statement
        has_return = any(isinstance(node, ast.Return) for node in case.body)

        if has_return:
            eis.append(f"match case {i + 1}: {pattern} → returns")
        else:
            eis.append(f"match case {i + 1}: {pattern}")

    return eis


def decompose_assert(stmt: ast.Assert, source_lines: list[str]) -> list[str]:
    """Assert statement: EIs for operations in test, then 2 EIs (assertion holds, assertion fails)."""
    eis = []

    # Extract operations from the assertion test
    operations = extract_all_operations(stmt.test)
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    test = ast.unparse(stmt.test)
    eis.extend([
        f"assert {test}: holds → continues",
        f"assert {test}: fails → raises AssertionError"
    ])

    return eis


def decompose_assignment(stmt: ast.Assign, source_lines: list[str]) -> list[str]:
    """
    Assignment: Enumerate all operations, then the assignment itself.
    Special cases:
    - List/dict/set comprehension: 2-3 EIs
    - Ternary expression: 4 EIs
    - Operations (calls, chained/nested): separate EIs for each
    """
    line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)

    # List comprehension
    if isinstance(stmt.value, ast.ListComp):
        return decompose_comprehension(stmt.value, "list", "[]")

    # Dict comprehension
    if isinstance(stmt.value, ast.DictComp):
        return decompose_comprehension(stmt.value, "dict", "{}")

    # Set comprehension
    if isinstance(stmt.value, ast.SetComp):
        return decompose_comprehension(stmt.value, "set", "set()")

    # Ternary expression (IfExp)
    if isinstance(stmt.value, ast.IfExp):
        return decompose_ternary(stmt.value)

    # Extract all operations (calls, in execution order)
    operations = extract_all_operations(stmt.value)

    if operations:
        eis = []
        for op_call in operations:
            call_str = ast.unparse(op_call)
            eis.append(f"executes → {call_str} succeeds")
            eis.append(f"{call_str} raises exception → exception propagates")

        # Only add "all operations succeed" EI if there are multiple operations
        if len(operations) > 1:
            eis.append(f"all operations succeed → {line_text}")

        return eis

    # Regular assignment (no operations)
    return [f"executes → {line_text}"]


def decompose_comprehension(comp: ast.ListComp | ast.DictComp | ast.SetComp, comp_type: str, empty_repr: str) -> list[
    str]:
    """Comprehension: EIs for operations in iterator/filter, then 3 EIs with filter or 2 without."""
    eis = []

    # Extract operations from the comprehension
    for gen in comp.generators:
        # Operations in the iterator expression
        operations = extract_all_operations(gen.iter)
        for op in operations:
            op_str = ast.unparse(op)
            eis.append(f"executes → {op_str} succeeds")
            eis.append(f"{op_str} raises exception → exception propagates")

        # Operations in filter conditions
        for if_clause in gen.ifs:
            operations = extract_all_operations(if_clause)
            for op in operations:
                op_str = ast.unparse(op)
                eis.append(f"executes → {op_str} succeeds")
                eis.append(f"{op_str} raises exception → exception propagates")

    has_filter = any(gen.ifs for gen in comp.generators)

    if has_filter:
        eis.extend([
            f"{comp_type} comprehension: source empty → {empty_repr}",
            f"{comp_type} comprehension: source has items, all filtered → {empty_repr}",
            f"{comp_type} comprehension: source has items, some pass filter → populated"
        ])
    else:
        eis.extend([
            f"{comp_type} comprehension: source empty → {empty_repr}",
            f"{comp_type} comprehension: source has items → populated"
        ])

    return eis


def contains_call_that_can_raise(node: ast.AST) -> tuple[bool, str | None]:
    """
    Check if an AST node contains function calls that can raise exceptions.

    Returns:
        (has_calls, call_description) - call_description is the unparsed call if found
    """
    operations = extract_all_operations(node)
    if operations:
        # Return the first operation found
        return True, ast.unparse(operations[0])
    return False, None


def decompose_ternary(ifexp: ast.IfExp) -> list[str]:
    """Ternary expression: EIs for operations in test/body/orelse, then 4 EIs (condition branches + value assignments)."""
    eis = []

    # Extract operations from the test condition
    operations = extract_all_operations(ifexp.test)
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    # Extract operations from true branch
    operations = extract_all_operations(ifexp.body)
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    # Extract operations from false branch
    operations = extract_all_operations(ifexp.orelse)
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    condition = ast.unparse(ifexp.test)
    true_val = ast.unparse(ifexp.body)
    false_val = ast.unparse(ifexp.orelse)

    eis.extend([
        f"{condition} is true → continues to true branch",
        f"{condition} is false → continues to false branch",
        f"true branch: assigns {true_val}",
        f"false branch: assigns {false_val}"
    ])

    return eis


def decompose_augassign(stmt: ast.AugAssign, source_lines: list[str]) -> list[str]:
    """Augmented assignment (+=, -=, etc.): EIs for operations in value, then 1 EI for assignment."""
    eis = []

    # Extract operations from the value being added/subtracted/etc
    operations = extract_all_operations(stmt.value)
    for op in operations:
        op_str = ast.unparse(op)
        eis.append(f"executes → {op_str} succeeds")
        eis.append(f"{op_str} raises exception → exception propagates")

    line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
    eis.append(f"executes → {line_text}")

    return eis


def decompose_annassign(stmt: ast.AnnAssign, source_lines: list[str]) -> list[str]:
    """Annotated assignment: EIs for operations in value (if present), then 1 EI for assignment."""
    eis = []

    # Extract operations from the value if it exists (annotated assignments can be declaration-only)
    if stmt.value:
        operations = extract_all_operations(stmt.value)
        for op in operations:
            op_str = ast.unparse(op)
            eis.append(f"executes → {op_str} succeeds")
            eis.append(f"{op_str} raises exception → exception propagates")

    line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
    eis.append(f"executes → {line_text}")

    return eis


def decompose_return(stmt: ast.Return, source_lines: list[str]) -> list[str]:
    """Return statement: Enumerate operations, then the return."""
    if stmt.value:
        ret_val = ast.unparse(stmt.value)

        # Extract all operations
        operations = extract_all_operations(stmt.value)

        if operations:
            eis = []
            for op_call in operations:
                call_str = ast.unparse(op_call)
                eis.append(f"executes → {call_str} succeeds")
                eis.append(f"{call_str} raises exception → exception propagates")
            # Final EI: return completes (only if all operations succeed)
            eis.append(f"all operations succeed → returns {ret_val}")
            return eis

        return [f"executes → returns {ret_val}"]
    else:
        return ["executes → returns None"]


def decompose_raise(stmt: ast.Raise, source_lines: list[str]) -> list[str]:
    """Raise statement: EIs for operations in exception, then 1 EI for raise."""
    eis = []

    if stmt.exc:
        # Extract operations from exception expression
        operations = extract_all_operations(stmt.exc)
        for op in operations:
            op_str = ast.unparse(op)
            eis.append(f"executes → {op_str} succeeds")
            eis.append(f"{op_str} raises exception → exception propagates")

        exc = ast.unparse(stmt.exc)
        eis.append(f"executes → raises {exc}")
        return eis
    else:
        return ["executes → re-raises current exception"]


def decompose_delete(stmt: ast.Delete, source_lines: list[str]) -> list[str]:
    """Delete statement: 1 EI."""
    targets = ', '.join(ast.unparse(t) for t in stmt.targets)
    return [f"executes: del {targets}"]


def decompose_pass(stmt: ast.Pass, source_lines: list[str]) -> list[str]:
    """Pass statement: 1 EI."""
    return ["executes: pass"]


def decompose_break(stmt: ast.Break, source_lines: list[str]) -> list[str]:
    """Break statement: 1 EI."""
    return ["executes: break"]


def decompose_continue(stmt: ast.Continue, source_lines: list[str]) -> list[str]:
    """Continue statement: 1 EI."""
    return ["executes: continue"]


def decompose_import(stmt: ast.Import, source_lines: list[str]) -> list[str]:
    """Import statement: 1 EI."""
    modules = ', '.join(alias.name for alias in stmt.names)
    return [f"executes: import {modules}"]


def decompose_importfrom(stmt: ast.ImportFrom, source_lines: list[str]) -> list[str]:
    """ImportFrom statement: 1 EI."""
    module = stmt.module or ""
    names = ', '.join(alias.name for alias in stmt.names)
    return [f"executes: from {module} import {names}"]


def decompose_expr(stmt: ast.Expr, source_lines: list[str]) -> list[str]:
    """Expression statement: Enumerate all operations."""
    # Skip docstrings (string literals as the first statement)
    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
        return []  # Docstrings don't create EIs

    line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)

    # Extract all operations
    operations = extract_all_operations(stmt.value)

    if operations:
        eis = []
        for op_call in operations:
            call_str = ast.unparse(op_call)
            eis.append(f"executes → {call_str} succeeds")
            eis.append(f"{call_str} raises exception → exception propagates")
        return eis

    return [f"executes → {line_text}"]


def decompose_global(stmt: ast.Global, source_lines: list[str]) -> list[str]:
    """Global statement: 1 EI."""
    names = ', '.join(stmt.names)
    return [f"executes → global {names}"]


def decompose_nonlocal(stmt: ast.Nonlocal, source_lines: list[str]) -> list[str]:
    """Nonlocal statement: 1 EI."""
    names = ', '.join(stmt.names)
    return [f"executes → nonlocal {names}"]


def decompose_asyncfor(stmt: ast.AsyncFor, source_lines: list[str]) -> list[str]:
    """Async for loop: Same as regular for."""
    return decompose_for(stmt, source_lines)  # type: ignore


def decompose_asyncwith(stmt: ast.AsyncWith, source_lines: list[str]) -> list[str]:
    """Async with statement: Same as regular with."""
    return decompose_with(stmt, source_lines)  # type: ignore


def decompose_default(stmt: ast.stmt, source_lines: list[str]) -> list[str]:
    """Default decomposer for unknown statement types."""
    line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
    return [f"executes → {line_text}"]


# Dispatch table mapping AST node types to decomposer functions
STATEMENT_DECOMPOSERS: dict[type[ast.stmt], Callable[[ast.stmt, list[str]], list[str]]] = {  # type: ignore[dict-item]
    # Conditionals
    ast.If: decompose_if,
    ast.Match: decompose_match,

    # Loops
    ast.For: decompose_for,
    ast.While: decompose_while,

    # Exception handling
    ast.Try: decompose_try,
    ast.With: decompose_with,

    # Assignments
    ast.Assign: decompose_assignment,
    ast.AugAssign: decompose_augassign,
    ast.AnnAssign: decompose_annassign,

    # Imports
    ast.Import: decompose_import,
    ast.ImportFrom: decompose_importfrom,

    # Flow control
    ast.Return: decompose_return,
    ast.Raise: decompose_raise,
    ast.Break: decompose_break,
    ast.Continue: decompose_continue,
    ast.Pass: decompose_pass,

    # Other statements
    ast.Delete: decompose_delete,
    ast.Assert: decompose_assert,
    ast.Expr: decompose_expr,
    ast.Global: decompose_global,
    ast.Nonlocal: decompose_nonlocal,

    # Async variants
    ast.AsyncFor: decompose_asyncfor,
    ast.AsyncWith: decompose_asyncwith,
}


# ============================================================================
# Main Decomposition Function
# ============================================================================

def decompose_statement(stmt: ast.stmt, source_lines: list[str]) -> list[str]:
    """
    Decompose a statement into outcome descriptions (EIs).

    Uses dispatch table to route to appropriate handler.
    Falls back to default handler for unknown statement types.
    """
    decomposer = STATEMENT_DECOMPOSERS.get(type(stmt), decompose_default)
    return decomposer(stmt, source_lines)


# ============================================================================
# AST Traversal
# ============================================================================

def get_all_statements(node: ast.AST) -> list[ast.stmt]:
    """Get all statements in an AST node, including nested ones."""
    statements: list[ast.stmt] = []

    for child in ast.walk(node):
        if isinstance(child, ast.stmt):
            statements.append(child)

    # Sort by line number
    statements.sort(key=lambda s: s.lineno)

    return statements


# ============================================================================
# Result structures
# ============================================================================

class FunctionResult:
    """Result of EI enumeration for a single function."""

    def __init__(self, name: str, line_start: int, line_end: int, branches: list[Branch]) -> None:
        self.name = name
        self.line_start = line_start
        self.line_end = line_end
        self.branches = branches
        self.total_eis = len(branches)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for YAML output."""
        return {
            'name': self.name,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'total_eis': self.total_eis,
            'branches': [b.to_dict() for b in self.branches]
        }


def enumerate_function_eis(
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        callable_id
) -> FunctionResult:
    """
    Enumerate all EIs in a function.

    Returns FunctionResult with Branch objects.
    """
    branches: list[Branch] = []
    ei_counter = 1

    # Get all statements in the function (including nested)
    statements = get_all_statements(func_node)

    # Filter to only statements inside this function's line range
    statements = [
        s for s in statements
        if func_node.lineno <= s.lineno <= func_node.end_lineno
    ]

    # Remove the function definition itself
    statements = [s for s in statements if s != func_node]

    for stmt in statements:
        outcomes = decompose_statement(stmt, source_lines)

        if outcomes:  # Skip empty (like docstrings)
            for outcome in outcomes:
                ei_id = generate_ei_id(callable_id, ei_counter)

                # Split outcome into condition and result
                if ' → ' in outcome:
                    condition, result = outcome.split(' → ', 1)
                else:
                    condition = 'executes'
                    result = outcome.replace('executes: ', '')

                branches.append(
                    Branch(
                        id=ei_id,
                        line=stmt.lineno,
                        condition=condition,
                        outcome=result
                    )
                )

                ei_counter += 1

    return FunctionResult(
        name=func_node.name,
        line_start=func_node.lineno,
        line_end=func_node.end_lineno,
        branches=branches
    )


# ============================================================================
# File Processing
# ============================================================================

class CallableFinder(ast.NodeVisitor):
    """Find all callables with proper FQN tracking."""

    def __init__(self, module_fqn: str, source_lines: list[str], inventory: dict[str, str], unit_id: str,
                 target_name: str | None):
        self.module_fqn = module_fqn
        self.source_lines = source_lines
        self.inventory = inventory
        self.unit_id = unit_id
        self.target_name = target_name
        self.results: list[FunctionResult] = []
        self.fqn_stack = [module_fqn] if module_fqn else []
        self.func_counter = 1
        self.assignment_counter = 1

    def visit_Assign(self, node) -> None:
        self._process_assignment(node)

    def visit_AnnAssign(self, node) -> None:
        self._process_assignment(node)

    def visit_AugAssign(self, node) -> None:
        self._process_assignment(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Push class onto FQN stack
        self.fqn_stack.append(node.name)

        # Visit children (methods)
        self.generic_visit(node)

        # Pop class
        self.fqn_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node)

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Skip if we're looking for a specific function
        if self.target_name and node.name != self.target_name:
            return

        # Build FQN
        if self.fqn_stack:
            fqn = f"{'.'.join(self.fqn_stack)}.{node.name}"
        else:
            fqn = node.name

        # Get callable ID from inventory or generate
        callable_id = self.inventory.get(fqn)
        if not callable_id:
            callable_id = generate_function_id(self.unit_id, self.func_counter)
            print(f"Warning: {fqn} not in inventory, generated {callable_id}")

        # Enumerate EIs for this callable
        result = enumerate_function_eis(node, self.source_lines, callable_id)
        self.results.append(result)

        self.func_counter += 1

    def _process_assignment(self, node: ast.Assign | ast.AnnAssign | ast.AugAssign):
        if not isinstance(node.value, ast.Call):
            return

        # Get target name(s) - Assign has targets (list), others have target (single)
        if isinstance(node, ast.Assign):
            # For multiple targets like a = b = value, just use the first one
            if not node.targets:
                return
            first_target = node.targets[0]
            if not isinstance(first_target, ast.Name):
                return  # Skip non-Name targets (tuples, attributes, etc.)
            target_name = first_target.id
        else:  # AnnAssign or AugAssign
            if not isinstance(node.target, ast.Name):
                return
            target_name = node.target.id

        fqn = f"{self.module_fqn}.{target_name}"
        callable_id = self.inventory.get(fqn)
        if not callable_id:
            callable_id = generate_assignment_id(self.unit_id, self.assignment_counter)
            print(f"Warning: {fqn} not in inventory, generated {callable_id}")

        self.assignment_counter += 1
        branches: list[Branch] = []
        ei_counter = 0

        match node:
            case ast.Assign():
                outcomes = decompose_assignment(node, self.source_lines)
            case ast.AnnAssign():
                outcomes = decompose_annassign(node, self.source_lines)
            case ast.AugAssign():
                outcomes = decompose_augassign(node, self.source_lines)

        if outcomes:
            for outcome in outcomes:
                ei_counter += 1
                ei_id = generate_ei_id(callable_id, ei_counter)
                if ' → ' in outcome:
                    condition, result = outcome.split(' → ', 1)
                else:
                    condition = 'executes'
                    result = outcome.replace('executes: ', '')
                branches.append(
                    Branch(
                        id=ei_id,
                        line=node.lineno,
                        condition=condition,
                        outcome=result
                    )
                )

            function_result = FunctionResult(
                name=target_name,
                line_start=node.lineno,
                line_end=node.end_lineno,
                branches=branches
            )

            self.results.append(function_result)


def enumerate_file(
        filepath: Path,
        unit_id: str,
        function_name: str | None = None,
        callable_inventory: dict[str, str] | None = None,
        module_fqn: str | None = None
) -> list[FunctionResult]:
    """
    Enumerate EIs for all functions in a file (or just one).

    Args:
        filepath: Path to Python file
        unit_id: Unit ID (fallback if inventory not available)
        function_name: Optional specific function to enumerate
        callable_inventory: Dict of FQN -> callable ID
        module_fqn: Module fully qualified name
    """

    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    source_lines = source.split('\n')
    tree = ast.parse(source)

    inventory = callable_inventory or {}

    # Use visitor to track class context
    finder = CallableFinder(module_fqn or "", source_lines, inventory, unit_id, function_name)
    finder.visit(tree)

    return finder.results


def format_for_yaml(results: list[FunctionResult]) -> dict[str, Any]:
    """Format results as dict for YAML output."""
    if not results:
        return {}

    return {
        'module': "unknown",
        'functions': [r.to_dict() for r in results]
    }


def format_outcome_map_text(result: FunctionResult) -> str:
    """Format the branches for display."""
    lines: list[str] = []
    lines.append(f"=== {result.name} (lines {result.line_start}-{result.line_end}) ===")
    lines.append(f"Total EIs: {result.total_eis}")
    lines.append("")
    lines.append("Execution Items:")

    for branch in result.branches:
        lines.append(f"\n{branch.id} (Line {branch.line}):")
        lines.append(f"  Condition: {branch.condition}")
        lines.append(f"  Outcome: {branch.outcome}")

    return '\n'.join(lines)


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Enumerate Execution Items (EIs) from Python source',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enumerate all functions in a file (YAML output)
  %(prog)s mymodule.py --output mymodule_eis.yaml

  # Enumerate a specific function
  %(prog)s mymodule.py --function validate_typed_dict

  # Human-readable text output
  %(prog)s mymodule.py --text
        """
    )

    parser.add_argument('file', type=Path, help='Python source file')
    parser.add_argument('--unit-id', '-u', required=True, help='Unit ID (required)')
    parser.add_argument('--function', '-f', help='Specific function name to enumerate')
    parser.add_argument('--callable-inventory', type=Path, help='Callable inventory file (FQN:ID pairs)')
    parser.add_argument('--source-root', type=Path, help='Source root for deriving FQN')
    parser.add_argument('--text', action='store_true', help='Output human-readable text instead of YAML')
    parser.add_argument('--output', '-o', type=Path, help='Save output to file')

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    # Load callable inventory if provided
    inventory = load_callable_inventory(args.callable_inventory) if args.callable_inventory else {}

    # Derive module FQN if source root provided
    module_fqn = None
    if args.source_root:
        module_fqn = derive_fqn_from_path(args.file, args.source_root)

    # Enumerate
    results = enumerate_file(args.file, args.unit_id, args.function, inventory, module_fqn)

    if not results:
        if args.function:
            print(f"Error: Function '{args.function}' not found in {args.file}")
        else:
            print(f"Error: No functions found in {args.file}")
        return 1

    # Format output
    if args.text:
        # Human-readable format
        output = '\n\n'.join(format_outcome_map_text(r) for r in results)
    else:
        # YAML format (default for pipeline)
        data = format_for_yaml(results)
        # Set module name from filename
        data['module'] = args.file.stem
        output = yaml.dump(data, sort_keys=False, allow_unicode=True, width=float('inf'))

    # Save or print
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Saved to {args.output}")
    else:
        print(output)

    return 0


if __name__ == '__main__':
    exit(main())