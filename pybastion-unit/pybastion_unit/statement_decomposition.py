"""
Statement Decomposers

Provides a class hierarchy for decomposing Python AST statements into
Execution Item (EI) outcome descriptions, with associated ast.Call nodes
for operation_target population in BranchConstraint.

Usage in enumerate_exec_items.py:
    from statement_decomposers import decompose_statement

Each outcome is returned as a tuple:
    (outcome_str, ast.Call | None)

Where the ast.Call is the specific call node that generated the outcome,
or None for semantic EIs (condition branches, loop boundaries, etc.).
This allows downstream constraint enrichment to populate operation_target
directly from the AST rather than re-parsing strings.
"""

from __future__ import annotations

import ast

# Type alias for a single EI outcome with its originating call node (if any)
EIOutcome = tuple[str, ast.Call | None]


# ============================================================================
# Core Helpers
# ============================================================================

def extract_all_operations(node: ast.AST) -> list[ast.Call]:
    """
    Extract ALL Call nodes from an AST in execution order.

    For nested/chained calls like Path(fetch(url)).resolve():
    - Returns: [fetch(url), Path(...), Path(...).resolve()]
    - Execution order: innermost first (by depth), then left-to-right
    """
    operations: list[tuple[ast.Call, int, int, int]] = []

    def collect(n: ast.AST, depth: int = 0) -> None:
        if isinstance(n, ast.Call):
            operations.append((n, depth, n.lineno, n.col_offset))
        for child in ast.iter_child_nodes(n):
            collect(child, depth + 1)

    collect(node)
    operations.sort(key=lambda x: (-x[1], x[2], x[3]))
    return [op[0] for op in operations]


def operation_eis(expressions: list[ast.AST]) -> list[EIOutcome]:
    """
    Generate success/exception EI pairs for all operations in the given expressions.

    For each ast.Call found (in execution order), emits:
      ("executes → {op} succeeds", call_node)
      ("{op} raises exception → exception propagates", call_node)

    Args:
        expressions: AST nodes to extract operations from

    Returns:
        List of (outcome_str, ast.Call) pairs
    """
    results: list[EIOutcome] = []
    for expr in expressions:
        for op in extract_all_operations(expr):
            op_str = ast.unparse(op)
            results.append((f"executes → {op_str} succeeds", op))
            results.append((f"{op_str} raises exception → exception propagates", op))
    return results


def semantic(outcome: str) -> EIOutcome:
    """Convenience wrapper for a semantic EI with no associated call node."""
    return (outcome, None)


# ============================================================================
# Base Decomposer
# ============================================================================

class StatementDecomposer:
    """
    Base class for statement decomposers.

    Implements the template method pattern:
      decompose() = operation_eis(expressions()) + semantic_eis()

    Subclasses override:
      - expressions(): which AST sub-nodes to extract operations from
      - semantic_eis(): the statement-specific outcome EIs

    The base decompose() method guarantees that operation EIs always come
    before semantic EIs, matching the existing enumeration order.
    """

    def expressions(self, stmt: ast.stmt) -> list[ast.AST]:
        """
        Return the AST sub-expressions to extract operations from.

        Override in subclasses to specify which parts of the statement
        contain executable operations (calls) that need EIs.
        """
        return []

    def semantic_eis(self, stmt: ast.stmt, source_lines: list[str]) -> list[EIOutcome]:
        """
        Return the statement-specific semantic EIs.

        These are the branching outcomes that are unique to this statement
        type (e.g., condition true/false, loop 0/≥1 iterations, etc.).
        Override in subclasses.
        """
        return []

    def decompose(self, stmt: ast.stmt, source_lines: list[str]) -> list[EIOutcome]:
        """
        Template method: operation EIs followed by semantic EIs.

        Do not override this — override expressions() and semantic_eis().
        """
        results = operation_eis(self.expressions(stmt))
        results.extend(self.semantic_eis(stmt, source_lines))
        return results


# ============================================================================
# Conditional Decomposers
# ============================================================================

class IfDecomposer(StatementDecomposer):
    """
    If statement: EIs for all operations in condition, then true/false branch EIs.

    For: if foo() and bar():
        executes → foo() succeeds
        foo() raises exception → exception propagates
        executes → bar() succeeds
        bar() raises exception → exception propagates
        <condition> is true → enters if block   (or specific return/raise)
        <condition> is false → continues
    """

    def expressions(self, stmt: ast.If) -> list[ast.AST]:
        return [stmt.test]

    def semantic_eis(self, stmt: ast.If, source_lines: list[str]) -> list[EIOutcome]:
        condition = ast.unparse(stmt.test)

        if stmt.body:
            first = stmt.body[0]

            if isinstance(first, ast.Raise):
                exc = ast.unparse(first.exc) if first.exc else "exception"
                return [
                    semantic(f"{condition} is true → raises {exc}"),
                    semantic(f"{condition} is false → continues"),
                ]

            if isinstance(first, ast.Return):
                ret_val = ast.unparse(first.value) if first.value else "None"
                return [
                    semantic(f"{condition} is true → returns {ret_val}"),
                    semantic(f"{condition} is false → continues"),
                ]

        return [
            semantic(f"{condition} is true → enters if block"),
            semantic(f"{condition} is false → continues"),
        ]


class MatchDecomposer(StatementDecomposer):
    """
    Match statement: EIs for subject expression operations, then one EI per case.
    """

    def expressions(self, stmt: ast.Match) -> list[ast.AST]:
        return [stmt.subject]

    def semantic_eis(self, stmt: ast.Match, source_lines: list[str]) -> list[EIOutcome]:
        results: list[EIOutcome] = []
        for i, case in enumerate(stmt.cases):
            pattern = ast.unparse(case.pattern)
            has_return = any(isinstance(n, ast.Return) for n in case.body)
            if has_return:
                results.append(semantic(f"match case {i + 1}: {pattern} → returns"))
            else:
                results.append(semantic(f"match case {i + 1}: {pattern}"))
        return results


# ============================================================================
# Loop Decomposers
# ============================================================================

class ForDecomposer(StatementDecomposer):
    """
    For loop: EIs for operations in iterable, then iteration boundary EIs.
    For-else: 3 EIs (empty, completes without break, breaks).
    """

    def expressions(self, stmt: ast.For) -> list[ast.AST]:
        return [stmt.iter]

    def semantic_eis(self, stmt: ast.For, source_lines: list[str]) -> list[EIOutcome]:
        target = ast.unparse(stmt.target)
        iter_expr = ast.unparse(stmt.iter)

        if stmt.orelse:
            return [
                semantic(f"for {target} in {iter_expr}: 0 iterations → else executes"),
                semantic(f"for {target} in {iter_expr}: completes without break → else executes"),
                semantic(f"for {target} in {iter_expr}: breaks → else skipped"),
            ]
        return [
            semantic(f"for {target} in {iter_expr}: 0 iterations"),
            semantic(f"for {target} in {iter_expr}: ≥1 iterations"),
        ]


class AsyncForDecomposer(ForDecomposer):
    """Async for loop — identical structure to regular for."""
    pass


class WhileDecomposer(StatementDecomposer):
    """
    While loop: EIs for operations in condition, then iteration boundary EIs.
    While-else: 3 EIs (initially false → else, completes → else, breaks → no else).
    """

    def expressions(self, stmt: ast.While) -> list[ast.AST]:
        return [stmt.test]

    def semantic_eis(self, stmt: ast.While, source_lines: list[str]) -> list[EIOutcome]:
        condition = ast.unparse(stmt.test)

        if stmt.orelse:
            return [
                semantic(f"while {condition}: initially false → else executes"),
                semantic(f"while {condition}: completes without break → else executes"),
                semantic(f"while {condition}: breaks → else skipped"),
            ]
        return [
            semantic(f"while {condition}: initially false → 0 iterations"),
            semantic(f"while {condition}: initially true → ≥1 iterations"),
        ]


# ============================================================================
# Exception Handling Decomposers
# ============================================================================

class TryDecomposer(StatementDecomposer):
    """
    Try/except: EIs for exception type expressions, then success + handler EIs.
    """

    def expressions(self, stmt: ast.Try) -> list[ast.AST]:
        # Operations can appear in exception type specifications (rare but possible)
        return [handler.type for handler in stmt.handlers if handler.type]

    def semantic_eis(self, stmt: ast.Try, source_lines: list[str]) -> list[EIOutcome]:
        results: list[EIOutcome] = [semantic("try block executes successfully")]

        for handler in stmt.handlers:
            if handler.type:
                exc_type = ast.unparse(handler.type)
                results.append(semantic(f"raises {exc_type} → enters except handler"))
            else:
                results.append(semantic("raises exception → enters except handler"))

        return results


class WithDecomposer(StatementDecomposer):
    """
    With statement: EIs for all context expressions, then entry EIs.
    """

    def expressions(self, stmt: ast.With) -> list[ast.AST]:
        return [item.context_expr for item in stmt.items]

    def semantic_eis(self, stmt: ast.With, source_lines: list[str]) -> list[EIOutcome]:
        contexts = [ast.unparse(item.context_expr) for item in stmt.items]
        context_str = ', '.join(contexts)
        return [
            semantic(f"with {context_str}: enters successfully"),
            semantic(f"with {context_str}: raises exception on entry"),
        ]


class AsyncWithDecomposer(WithDecomposer):
    """Async with statement — identical structure to regular with."""
    pass


# ============================================================================
# Assignment Decomposers
# ============================================================================

class AssignDecomposer(StatementDecomposer):
    """
    Assignment: EIs for all operations in the value, then the assignment itself.
    Handles comprehensions and ternary expressions as special cases.
    """

    def decompose(self, stmt: ast.Assign, source_lines: list[str]) -> list[EIOutcome]:
        if isinstance(stmt.value, ast.ListComp):
            return ComprehensionDecomposer().decompose_comp(stmt.value, "list", "[]")
        if isinstance(stmt.value, ast.DictComp):
            return ComprehensionDecomposer().decompose_comp(stmt.value, "dict", "{}")
        if isinstance(stmt.value, ast.SetComp):
            return ComprehensionDecomposer().decompose_comp(stmt.value, "set", "set()")
        if isinstance(stmt.value, ast.IfExp):
            return TernaryDecomposer().decompose_ternary(stmt.value)

        ops = extract_all_operations(stmt.value)

        if not ops:
            line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
            return [semantic(f"executes → {line_text}")]

        results = operation_eis([stmt.value])

        if len(ops) > 1:
            line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
            results.append(semantic(f"all operations succeed → {line_text}"))

        return results


class AugAssignDecomposer(StatementDecomposer):
    """Augmented assignment (+=, -=, etc.): EIs for operations in value, then assignment EI."""

    def expressions(self, stmt: ast.AugAssign) -> list[ast.AST]:
        return [stmt.value]

    def semantic_eis(self, stmt: ast.AugAssign, source_lines: list[str]) -> list[EIOutcome]:
        line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(f"executes → {line_text}")]


class AnnAssignDecomposer(StatementDecomposer):
    """Annotated assignment: EIs for operations in value (if present), then assignment EI."""

    def expressions(self, stmt: ast.AnnAssign) -> list[ast.AST]:
        return [stmt.value] if stmt.value else []

    def semantic_eis(self, stmt: ast.AnnAssign, source_lines: list[str]) -> list[EIOutcome]:
        line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(f"executes → {line_text}")]


# ============================================================================
# Expression Decomposers (helpers, not in dispatch table directly)
# ============================================================================

class ComprehensionDecomposer:
    """Helper for list/dict/set comprehension EI generation."""

    def decompose_comp(
            self,
            comp: ast.ListComp | ast.DictComp | ast.SetComp,
            comp_type: str,
            empty_repr: str
    ) -> list[EIOutcome]:
        # Operations in iterators and filter conditions
        exprs: list[ast.AST] = []
        for gen in comp.generators:
            exprs.append(gen.iter)
            exprs.extend(gen.ifs)

        results = operation_eis(exprs)

        has_filter = any(gen.ifs for gen in comp.generators)
        if has_filter:
            results.extend([
                semantic(f"{comp_type} comprehension: source empty → {empty_repr}"),
                semantic(f"{comp_type} comprehension: source has items, all filtered → {empty_repr}"),
                semantic(f"{comp_type} comprehension: source has items, some pass filter → populated"),
            ])
        else:
            results.extend([
                semantic(f"{comp_type} comprehension: source empty → {empty_repr}"),
                semantic(f"{comp_type} comprehension: source has items → populated"),
            ])

        return results


class TernaryDecomposer:
    """Helper for ternary expression (IfExp) EI generation."""

    def decompose_ternary(self, ifexp: ast.IfExp) -> list[EIOutcome]:
        results = operation_eis([ifexp.test, ifexp.body, ifexp.orelse])

        condition = ast.unparse(ifexp.test)
        true_val = ast.unparse(ifexp.body)
        false_val = ast.unparse(ifexp.orelse)

        results.extend([
            semantic(f"{condition} is true → continues to true branch"),
            semantic(f"{condition} is false → continues to false branch"),
            semantic(f"true branch: assigns {true_val}"),
            semantic(f"false branch: assigns {false_val}"),
        ])

        return results


# ============================================================================
# Flow Control Decomposers
# ============================================================================

class ReturnDecomposer(StatementDecomposer):
    """Return statement: EIs for operations in return value, then return EI."""

    def decompose(self, stmt: ast.Return, source_lines: list[str]) -> list[EIOutcome]:
        if not stmt.value:
            return [semantic("executes → returns None")]

        ret_val = ast.unparse(stmt.value)
        results = operation_eis([stmt.value])

        if results:
            results.append(semantic(f"all operations succeed → returns {ret_val}"))
        else:
            results.append(semantic(f"executes → returns {ret_val}"))

        return results


class RaiseDecomposer(StatementDecomposer):
    """Raise statement: EIs for operations in exception expression, then raise EI."""

    def decompose(self, stmt: ast.Raise, source_lines: list[str]) -> list[EIOutcome]:
        if not stmt.exc:
            return [semantic("executes → re-raises current exception")]

        results = operation_eis([stmt.exc])
        exc = ast.unparse(stmt.exc)
        results.append(semantic(f"executes → raises {exc}"))
        return results


class AssertDecomposer(StatementDecomposer):
    """Assert statement: EIs for operations in test, then assertion holds/fails EIs."""

    def expressions(self, stmt: ast.Assert) -> list[ast.AST]:
        return [stmt.test]

    def semantic_eis(self, stmt: ast.Assert, source_lines: list[str]) -> list[EIOutcome]:
        test = ast.unparse(stmt.test)
        return [
            semantic(f"assert {test}: holds → continues"),
            semantic(f"assert {test}: fails → raises AssertionError"),
        ]


class ExprDecomposer(StatementDecomposer):
    """
    Expression statement: EIs for all operations.
    Skips docstrings (string literal expressions).
    """

    def decompose(self, stmt: ast.Expr, source_lines: list[str]) -> list[EIOutcome]:
        # Skip docstrings
        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            return []

        results = operation_eis([stmt.value])

        if not results:
            line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
            return [semantic(f"executes → {line_text}")]

        return results


# ============================================================================
# Simple Single-EI Decomposers
# ============================================================================

class DeleteDecomposer(StatementDecomposer):
    def semantic_eis(self, stmt: ast.Delete, source_lines: list[str]) -> list[EIOutcome]:
        targets = ', '.join(ast.unparse(t) for t in stmt.targets)
        return [semantic(f"executes: del {targets}")]


class PassDecomposer(StatementDecomposer):
    def semantic_eis(self, stmt: ast.Pass, source_lines: list[str]) -> list[EIOutcome]:
        return [semantic("executes: pass")]


class BreakDecomposer(StatementDecomposer):
    def semantic_eis(self, stmt: ast.Break, source_lines: list[str]) -> list[EIOutcome]:
        return [semantic("executes: break")]


class ContinueDecomposer(StatementDecomposer):
    def semantic_eis(self, stmt: ast.Continue, source_lines: list[str]) -> list[EIOutcome]:
        return [semantic("executes: continue")]


class ImportDecomposer(StatementDecomposer):
    def semantic_eis(self, stmt: ast.Import, source_lines: list[str]) -> list[EIOutcome]:
        modules = ', '.join(alias.name for alias in stmt.names)
        return [semantic(f"executes: import {modules}")]


class ImportFromDecomposer(StatementDecomposer):
    def semantic_eis(self, stmt: ast.ImportFrom, source_lines: list[str]) -> list[EIOutcome]:
        module = stmt.module or ""
        names = ', '.join(alias.name for alias in stmt.names)
        return [semantic(f"executes: from {module} import {names}")]


class GlobalDecomposer(StatementDecomposer):
    def semantic_eis(self, stmt: ast.Global, source_lines: list[str]) -> list[EIOutcome]:
        names = ', '.join(stmt.names)
        return [semantic(f"executes → global {names}")]


class NonlocalDecomposer(StatementDecomposer):
    def semantic_eis(self, stmt: ast.Nonlocal, source_lines: list[str]) -> list[EIOutcome]:
        names = ', '.join(stmt.names)
        return [semantic(f"executes → nonlocal {names}")]


class DefaultDecomposer(StatementDecomposer):
    """Fallback decomposer for unknown statement types."""

    def semantic_eis(self, stmt: ast.stmt, source_lines: list[str]) -> list[EIOutcome]:
        line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(f"executes → {line_text}")]


# ============================================================================
# Dispatch Table
# ============================================================================

_DECOMPOSERS: dict[type[ast.stmt], StatementDecomposer] = {
    # Conditionals
    ast.If: IfDecomposer(),
    ast.Match: MatchDecomposer(),

    # Loops
    ast.For: ForDecomposer(),
    ast.While: WhileDecomposer(),
    ast.AsyncFor: AsyncForDecomposer(),

    # Exception handling
    ast.Try: TryDecomposer(),
    ast.With: WithDecomposer(),
    ast.AsyncWith: AsyncWithDecomposer(),

    # Assignments
    ast.Assign: AssignDecomposer(),
    ast.AugAssign: AugAssignDecomposer(),
    ast.AnnAssign: AnnAssignDecomposer(),

    # Imports
    ast.Import: ImportDecomposer(),
    ast.ImportFrom: ImportFromDecomposer(),

    # Flow control
    ast.Return: ReturnDecomposer(),
    ast.Raise: RaiseDecomposer(),
    ast.Break: BreakDecomposer(),
    ast.Continue: ContinueDecomposer(),
    ast.Pass: PassDecomposer(),

    # Other
    ast.Delete: DeleteDecomposer(),
    ast.Assert: AssertDecomposer(),
    ast.Expr: ExprDecomposer(),
    ast.Global: GlobalDecomposer(),
    ast.Nonlocal: NonlocalDecomposer(),
}

_DEFAULT_DECOMPOSER = DefaultDecomposer()


# ============================================================================
# Public API
# ============================================================================

def decompose_statement(stmt: ast.stmt, source_lines: list[str]) -> list[EIOutcome]:
    """
    Decompose a statement into (outcome_str, ast.Call | None) pairs.

    Each tuple contains:
      - outcome_str: human-readable EI description
      - ast.Call | None: the originating call node (for operation_target
        population in BranchConstraint), or None for semantic EIs

    Args:
        stmt: AST statement node to decompose
        source_lines: Source file lines for line-text extraction

    Returns:
        List of EIOutcome tuples
    """
    decomposer = _DECOMPOSERS.get(type(stmt), _DEFAULT_DECOMPOSER)
    return decomposer.decompose(stmt, source_lines)
