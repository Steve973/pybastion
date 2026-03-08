"""
Statement Decomposers

Provides a class hierarchy for decomposing Python AST statements into
Execution Item (EI) outcome descriptions, with associated ast.Call nodes
for operation_target population in BranchConstraint.

Usage in stage2_enumerate_exec_items.py:
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
from dataclasses import dataclass

from pybastion_unit.shared.knowledge_base import NO_OP_CALLS


@dataclass
class DecomposerResult:
    outcome: str
    call_node: ast.Call | None = None
    target_line: int | None = None
    skips_lines: list[int] | None = None

    def __len__(self) -> int:
        count = 1
        if self.call_node:
            count += 1
        if self.target_line:
            count += 1
        return count

# ============================================================================
# Core Helpers
# ============================================================================

def _is_no_op_call(node: ast.Call) -> bool:
    """Return True if this call is a known no-op that should not generate EIs."""
    func = node.func
    if isinstance(func, ast.Name):
        # Match by tail: 'cast' matches 'typing.cast'
        return any(fqn.endswith(f'.{func.id}') for fqn in NO_OP_CALLS)
    if isinstance(func, ast.Attribute):
        # Match fully qualified: 'typing.cast'
        return ast.unparse(func) in NO_OP_CALLS
    return False


def extract_all_operations(node: ast.AST) -> list[ast.Call]:
    """
    Extract ALL Call nodes from an AST in execution order.

    For nested/chained calls like Path(fetch(url)).resolve():
    - Returns: [fetch(url), Path(...), Path(...).resolve()]
    - Execution order: innermost first (by depth), then left-to-right

    Known no-op calls (e.g. typing.cast) are excluded.
    """
    operations: list[tuple[ast.Call, int, int, int]] = []

    def collect(n: ast.AST, depth: int = 0) -> None:
        if isinstance(n, ast.Call):
            operations.append((n, depth, n.lineno, n.col_offset))
        for child in ast.iter_child_nodes(n):
            collect(child, depth + 1)

    collect(node)
    operations.sort(key=lambda x: (-x[1], x[2], x[3]))
    return [op[0] for op in operations if not _is_no_op_call(op[0])]


def operation_eis(expressions: list[ast.AST]) -> list[DecomposerResult]:
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
    results: list[DecomposerResult] = []
    for expr in expressions:
        for op in extract_all_operations(expr):
            op_str = ast.unparse(op)
            results.extend(
                [
                    DecomposerResult(outcome=f"executes → {op_str} succeeds", call_node=op),
                    DecomposerResult(outcome=f"{op_str} raises exception → exception propagates", call_node=op)
                ]
            )
    return results


def semantic(outcome: str) -> DecomposerResult:
    """Convenience wrapper for a semantic EI with no associated call node."""
    return DecomposerResult(outcome, None)


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

    @classmethod
    def expressions(cls, stmt: ast.stmt) -> list[ast.AST]:
        """
        Return the AST sub-expressions to extract operations from.

        Override in subclasses to specify which parts of the statement
        contain executable operations (calls) that need EIs.
        """
        return []

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.stmt,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        """
        Return the statement-specific semantic EIs.

        These are the branching outcomes that are unique to this statement
        type (e.g., condition true/false, loop 0/≥1 iterations, etc.).
        Override in subclasses.
        """
        return []

    @classmethod
    def decompose(
            cls,
            stmt: ast.stmt,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        """
        Template method: operation EIs followed by semantic EIs.

        Do not override this — override expressions() and semantic_eis().
        """
        results = operation_eis(cls.expressions(stmt))
        results.extend(cls.semantic_eis(stmt, source_lines, next_stmt_line))
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

    @classmethod
    def expressions(cls, stmt: ast.If) -> list[ast.AST]:
        # When we have a BoolOp, BoolOpDecomposer handles operations
        # Otherwise return the test expression for operation extraction
        if isinstance(stmt.test, ast.BoolOp):
            return []  # BoolOpDecomposer will handle it
        return [stmt.test]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.If,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        condition = ast.unparse(stmt.test)

        # Get target lines for branches
        true_target = stmt.body[0].lineno if stmt.body else None
        false_target = stmt.orelse[0].lineno if stmt.orelse else next_stmt_line

        # Get line ranges for skips_eis
        if_body_lines = []
        if stmt.body:
            if_body_lines = list(range(stmt.body[0].lineno, stmt.body[-1].end_lineno + 1))

        else_body_lines = []
        if stmt.orelse:
            else_body_lines = list(range(stmt.orelse[0].lineno, stmt.orelse[-1].end_lineno + 1))

        # Check if condition is a BoolOp - if so, delegate to BoolOpDecomposer
        if isinstance(stmt.test, ast.BoolOp):
            # ... existing BoolOp handling ...
            pass

        # Rest of existing logic for non-BoolOp conditions
        if stmt.body:
            first = stmt.body[0]

            if isinstance(first, ast.Raise):
                exc = ast.unparse(first.exc) if first.exc else "exception"
                return [
                    DecomposerResult(
                        outcome=f"{condition} is true → raises {exc}",
                        target_line=true_target,
                        skips_lines=else_body_lines  # True skips else
                    ),
                    DecomposerResult(
                        outcome=f"{condition} is false → continues",
                        target_line=false_target,
                        skips_lines=if_body_lines  # False skips if body
                    )
                ]

            if isinstance(first, ast.Return):
                ret_val = ast.unparse(first.value) if first.value else "None"
                return [
                    DecomposerResult(
                        outcome=f"{condition} is true → returns {ret_val}",
                        target_line=true_target,
                        skips_lines=else_body_lines
                    ),
                    DecomposerResult(
                        outcome=f"{condition} is false → continues",
                        target_line=false_target,
                        skips_lines=if_body_lines
                    )
                ]

        return [
            DecomposerResult(
                outcome=f"{condition} is true → enters if block",
                target_line=true_target,
                skips_lines=else_body_lines
            ),
            DecomposerResult(
                outcome=f"{condition} is false → continues",
                target_line=false_target,
                skips_lines=if_body_lines
            )
        ]


class MatchDecomposer(StatementDecomposer):
    """
    Match statement: EIs for subject expression operations, then one EI per case.
    """

    @classmethod
    def expressions(cls, stmt: ast.Match) -> list[ast.AST]:
        return [stmt.subject]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Match,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        results: list[DecomposerResult] = []
        for i, case in enumerate(stmt.cases):
            pattern = ast.unparse(case.pattern)
            case_target = case.body[0].lineno if case.body else None
            has_return = any(isinstance(n, ast.Return) for n in case.body)
            if has_return:
                results.append(DecomposerResult(f"match case {i + 1}: {pattern} → returns", None, case_target))
            else:
                results.append(DecomposerResult(f"match case {i + 1}: {pattern}", None, case_target))
        return results


# ============================================================================
# Loop Decomposers
# ============================================================================

class ForDecomposer(StatementDecomposer):
    """
    For loop: EIs for operations in iterable, then iteration boundary EIs.
    For-else: 3 EIs (empty, completes without break, breaks).
    """

    @classmethod
    def expressions(cls, stmt: ast.For) -> list[ast.AST]:
        return [stmt.iter]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.For,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        target = ast.unparse(stmt.target)
        iter_expr = ast.unparse(stmt.iter)

        # Get body line range for skips_eis
        if stmt.body:
            body_start = stmt.body[0].lineno
            body_end = stmt.body[-1].end_lineno
            body_lines = list(range(body_start, body_end + 1))
        else:
            body_lines = []

        # Get target lines for branches
        loop_body_target = stmt.body[0].lineno if stmt.body else None
        exit_target = stmt.orelse[0].lineno if stmt.orelse else next_stmt_line

        return [
            DecomposerResult(
                outcome=f"for {target} in {iter_expr}: 0 iterations",
                target_line=exit_target,
                skips_lines=body_lines
            ),
            DecomposerResult(
                outcome=f"for {target} in {iter_expr}: ≥1 iterations",
                target_line=loop_body_target
            )
        ]


class AsyncForDecomposer(ForDecomposer):
    """Async for loop — identical structure to regular for."""
    pass


class WhileDecomposer(StatementDecomposer):
    """
    While loop: EIs for operations in condition, then iteration boundary EIs.
    While-else: 3 EIs (initially false → else, completes → else, breaks → no else).
    """

    @classmethod
    def expressions(cls, stmt: ast.While) -> list[ast.AST]:
        return [stmt.test]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.While,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        condition = ast.unparse(stmt.test)

        # Get body line range for skips_eis
        if stmt.body:
            body_start = stmt.body[0].lineno
            body_end = stmt.body[-1].end_lineno
            body_lines = list(range(body_start, body_end + 1))
        else:
            body_lines = []

        # Get target lines
        loop_body_target = stmt.body[0].lineno if stmt.body else None
        exit_target = stmt.orelse[0].lineno if stmt.orelse else next_stmt_line

        return [
            DecomposerResult(
                outcome=f"{condition} initially false",
                target_line=exit_target,
                skips_lines=body_lines
            ),
            DecomposerResult(
                outcome=f"{condition} initially true",
                target_line=loop_body_target
            )
        ]


# ============================================================================
# Exception Handling Decomposers
# ============================================================================

class TryDecomposer(StatementDecomposer):
    """
    Try/except: EIs for exception type expressions, then success + handler EIs.
    """

    @classmethod
    def expressions(cls, stmt: ast.Try) -> list[ast.AST]:
        # Operations can appear in exception type specifications (rare but possible)
        return [handler.type for handler in stmt.handlers if handler.type]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Try,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        # Success case goes to finalbody or next stmt after try
        if stmt.finalbody:
            success_target = stmt.finalbody[0].lineno
        else:
            success_target = next_stmt_line

        results: list[DecomposerResult] = [
            DecomposerResult("try block executes successfully", None, success_target)
        ]

        for handler in stmt.handlers:
            # Each handler goes to its body
            handler_target = handler.body[0].lineno if handler.body else None

            if handler.type:
                exc_type = ast.unparse(handler.type)
                results.append(DecomposerResult(f"raises {exc_type} → enters except handler", None, handler_target))
            else:
                results.append(DecomposerResult("raises exception → enters except handler", None, handler_target))

        return results


class WithDecomposer(StatementDecomposer):
    """
    With statement: EIs for all context expressions, then entry EIs.
    """

    @classmethod
    def expressions(cls, stmt: ast.With) -> list[ast.AST]:
        return [item.context_expr for item in stmt.items]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.With,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
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

    @classmethod
    def decompose(
            cls,
            stmt: ast.Assign,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        if isinstance(stmt.value, ast.ListComp):
            return ComprehensionDecomposer.decompose_comp(stmt.value, "list", "[]")
        if isinstance(stmt.value, ast.DictComp):
            return ComprehensionDecomposer.decompose_comp(stmt.value, "dict", "{}")
        if isinstance(stmt.value, ast.SetComp):
            return ComprehensionDecomposer.decompose_comp(stmt.value, "set", "set()")
        if isinstance(stmt.value, ast.IfExp):
            return TernaryDecomposer.decompose_ternary(stmt.value)
        if isinstance(stmt.value, ast.BoolOp):
            return BoolOpDecomposer.decompose_boolop(stmt.value, next_stmt_line)

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

    @classmethod
    def expressions(cls, stmt: ast.AugAssign) -> list[ast.AST]:
        return [stmt.value]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.AugAssign,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        ops = extract_all_operations(stmt.value) if stmt.value else []
        if ops:
            return []
        line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(f"executes → {line_text}")]


class AnnAssignDecomposer(StatementDecomposer):
    """Annotated assignment: EIs for operations in value (if present), then assignment EI."""

    @classmethod
    def expressions(cls, stmt: ast.AnnAssign) -> list[ast.AST]:
        return [stmt.value] if stmt.value else []

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.AnnAssign,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        ops = extract_all_operations(stmt.value) if stmt.value else []
        if ops:
            return []
        line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(f"executes → {line_text}")]


# ============================================================================
# Expression Decomposers (helpers, not in dispatch table directly)
# ============================================================================

class ComprehensionDecomposer:
    """Helper for list/dict/set comprehension EI generation."""

    @classmethod
    def decompose_comp(
            cls,
            comp: ast.ListComp | ast.DictComp | ast.SetComp,
            comp_type: str,
            empty_repr: str
    ) -> list[DecomposerResult]:
        # Operations in iterators and filter conditions
        exprs: list[ast.AST] = []
        for gen in comp.generators:
            exprs.append(gen.iter)
            exprs.extend(gen.ifs)

        results = operation_eis(exprs)

        # Extract the iterator expression for constraint
        iter_expr = ast.unparse(comp.generators[0].iter) if comp.generators else "source"

        has_filter = any(gen.ifs for gen in comp.generators)
        if has_filter:
            results.extend([
                semantic(f"{iter_expr} is empty → {empty_repr}"),
                semantic(f"{iter_expr} has items, all filtered → {empty_repr}"),
                semantic(f"{iter_expr} has items, some pass filter → populated"),
            ])
        else:
            results.extend([
                semantic(f"{iter_expr} is empty → {empty_repr}"),
                semantic(f"{iter_expr} has items → populated"),
            ])

        return results


class TernaryDecomposer:
    """Helper for ternary expression (IfExp) EI generation."""

    @classmethod
    def decompose_ternary(cls, ifexp: ast.IfExp) -> list[DecomposerResult]:
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


class BoolOpDecomposer:
    """Helper for short-circuit boolean expression (or/and) EI generation."""

    @classmethod
    def decompose_boolop(cls, boolop: ast.BoolOp, next_stmt_line: int | None = None) -> list[DecomposerResult]:
        """
        Decompose `x or y` and `x and y` into short-circuit branches.

        For `x or y`:
            - Operations in x
            - x is truthy → uses x
            - x is falsy → continues to y
            - Operations in y
            - All operations succeed → uses y

        For `x and y`:
            - Operations in x
            - x is falsy → uses x (short-circuits)
            - x is truthy → continues to y
            - Operations in y
            - All operations succeed → uses y
        """
        results: list[DecomposerResult] = []

        # BoolOp has: op (Or | And) and values (list of expressions)
        # For simplicity, handle binary case: x op y
        # Multi-operand like x or y or z would need recursive handling

        if len(boolop.values) == 2:
            left = boolop.values[0]
            right = boolop.values[1]

            # Operations in left operand
            results.extend(operation_eis([left]))

            left_str = ast.unparse(left)
            right_str = ast.unparse(right)

            if isinstance(boolop.op, ast.Or):
                # x or y: if x is truthy, use x; else evaluate y
                results.extend([
                    semantic(f"{left_str} is true → uses {left_str}"),
                    semantic(f"{left_str} is false → continues to {right_str}"),
                ])

                # Operations in right operand
                results.extend(operation_eis([right]))

                # Generate final outcome
                right_ops = extract_all_operations(right)
                if right_ops:
                    results.append(semantic(f"all operations succeed → uses {right_str}"))
                else:
                    results.append(semantic(f"uses {right_str}"))

            elif isinstance(boolop.op, ast.And):
                # x and y: if x is falsy, use x; else evaluate y
                results.extend([
                    semantic(f"{left_str} is false → uses {left_str}"),
                    semantic(f"{left_str} is true → continues to {right_str}"),
                ])

                # Operations in right operand
                results.extend(operation_eis([right]))

                # Generate final outcome
                right_ops = extract_all_operations(right)
                if right_ops:
                    results.append(semantic(f"all operations succeed → uses {right_str}"))
                else:
                    results.append(semantic(f"uses {right_str}"))

        else:
            # Multi-operand case: x or y or z
            # For now, fall back to treating the whole thing as one expression
            results.extend(operation_eis([boolop]))

        # Wrap only short-circuit branches with target_line
        wrapped = []
        for result in results:
            outcome = result.outcome
            call_node = result.call_node
            if 'is true → uses' in outcome or 'is false → uses' in outcome:
                wrapped.append(DecomposerResult(outcome, call_node, next_stmt_line))
            else:
                wrapped.append(DecomposerResult(outcome, call_node))
        return wrapped


# ============================================================================
# Flow Control Decomposers
# ============================================================================

class ReturnDecomposer(StatementDecomposer):
    """Return statement: EIs for operations in return value, then return EI."""

    @classmethod
    def decompose(
            cls,
            stmt: ast.Return,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
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

    @classmethod
    def decompose(
            cls,
            stmt: ast.Raise,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        if not stmt.exc:
            return [semantic("executes → re-raises current exception")]

        results = operation_eis([stmt.exc])
        exc = ast.unparse(stmt.exc)
        results.append(semantic(f"executes → raises {exc}"))
        return results


class AssertDecomposer(StatementDecomposer):
    """Assert statement: EIs for operations in test, then assertion holds/fails EIs."""

    @classmethod
    def expressions(cls, stmt: ast.Assert) -> list[ast.AST]:
        return [stmt.test]

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Assert,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
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

    @classmethod
    def decompose(
            cls,
            stmt: ast.Expr,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
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
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Delete,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        targets = ', '.join(ast.unparse(t) for t in stmt.targets)
        return [semantic(f"executes: del {targets}")]


class PassDecomposer(StatementDecomposer):
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Pass,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        return [semantic("executes: pass")]


class BreakDecomposer(StatementDecomposer):
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Break,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        return [semantic("executes: break")]


class ContinueDecomposer(StatementDecomposer):
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Continue,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        return [semantic("executes: continue")]


class ImportDecomposer(StatementDecomposer):
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Import,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        modules = ', '.join(alias.name for alias in stmt.names)
        return [semantic(f"executes: import {modules}")]


class ImportFromDecomposer(StatementDecomposer):
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.ImportFrom,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        module = stmt.module or ""
        names = ', '.join(alias.name for alias in stmt.names)
        return [semantic(f"executes: from {module} import {names}")]


class GlobalDecomposer(StatementDecomposer):
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Global,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        names = ', '.join(stmt.names)
        return [semantic(f"executes → global {names}")]


class NonlocalDecomposer(StatementDecomposer):
    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.Nonlocal,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        names = ', '.join(stmt.names)
        return [semantic(f"executes → nonlocal {names}")]


class DefaultDecomposer(StatementDecomposer):
    """Fallback decomposer for unknown statement types."""

    @classmethod
    def semantic_eis(
            cls,
            stmt: ast.stmt,
            source_lines: list[str],
            next_stmt_line: int | None = None
    ) -> list[DecomposerResult]:
        line_text = source_lines[stmt.lineno - 1].strip() if stmt.lineno <= len(source_lines) else ast.unparse(stmt)
        return [semantic(f"executes → {line_text}")]


# ============================================================================
# Dispatch Table
# ============================================================================

_DECOMPOSERS: dict[type[ast.stmt], type[StatementDecomposer]] = {
    # Conditionals
    ast.If: IfDecomposer,
    ast.Match: MatchDecomposer,

    # Loops
    ast.For: ForDecomposer,
    ast.While: WhileDecomposer,
    ast.AsyncFor: AsyncForDecomposer,

    # Exception handling
    ast.Try: TryDecomposer,
    ast.With: WithDecomposer,
    ast.AsyncWith: AsyncWithDecomposer,

    # Assignments
    ast.Assign: AssignDecomposer,
    ast.AugAssign: AugAssignDecomposer,
    ast.AnnAssign: AnnAssignDecomposer,

    # Imports
    ast.Import: ImportDecomposer,
    ast.ImportFrom: ImportFromDecomposer,

    # Flow control
    ast.Return: ReturnDecomposer,
    ast.Raise: RaiseDecomposer,
    ast.Break: BreakDecomposer,
    ast.Continue: ContinueDecomposer,
    ast.Pass: PassDecomposer,

    # Other
    ast.Delete: DeleteDecomposer,
    ast.Assert: AssertDecomposer,
    ast.Expr: ExprDecomposer,
    ast.Global: GlobalDecomposer,
    ast.Nonlocal: NonlocalDecomposer,
}

_DEFAULT_DECOMPOSER = DefaultDecomposer


# ============================================================================
# Public API
# ============================================================================

def decompose_statement(
        stmt: ast.stmt,
        source_lines: list[str],
        next_stmt_line: int | None = None) -> list[DecomposerResult]:
    """
    Decompose a statement into (outcome_str, ast.Call | None) or (outcome_str, ast.Call | None, target_line) tuples.

    Each tuple contains:
      - outcome_str: human-readable EI description
      - ast.Call | None: the originating call node (for operation_target
        population in BranchConstraint), or None for semantic EIs
      - target_line (optional): line number where execution continues for control flow branches

    Args:
        stmt: AST statement node to decompose
        source_lines: Source file lines for line-text extraction
        next_stmt_line: Line number of the next statement after this one (for control flow exits)

    Returns:
        List of EIOutcome or LineOutcome tuples
    """
    decomposer = _DECOMPOSERS.get(type(stmt), _DEFAULT_DECOMPOSER)
    return decomposer.decompose(stmt, source_lines, next_stmt_line)
