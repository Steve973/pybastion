# Python Statement Coverage Reference

This document lists all Python statement types and how they're handled by enumerate_eis_v2.py

## ✅ Fully Covered (26 statement types)

### Control Flow (6)
- **ast.If** → 2 EIs (true/false branches)
- **ast.For** → 2 EIs (0 iterations, ≥1 iterations) or 3 EIs with else
- **ast.While** → 2 EIs (initially false, initially true) or 3 EIs with else
- **ast.Try** → 1 + N EIs (success + each exception handler)
- **ast.With** → 2 EIs (enters successfully, raises on entry)
- **ast.Match** → N EIs (one per case)

### Assignments (3)
- **ast.Assign** → 1 EI (or 2-4 for comprehensions/ternary)
  - List/dict/set comprehension → 2-3 EIs (with/without filter)
  - Ternary expression (IfExp) → 4 EIs (condition branches + assignments)
- **ast.AugAssign** → 1 EI (+=, -=, etc.)
- **ast.AnnAssign** → 1 EI (annotated assignment)

### Flow Control (5)
- **ast.Return** → 1 EI
- **ast.Raise** → 1 EI
- **ast.Break** → 1 EI
- **ast.Continue** → 1 EI
- **ast.Pass** → 1 EI

### Other Statements (6)
- **ast.Delete** → 1 EI
- **ast.Assert** → 2 EIs (assertion holds, assertion fails)
- **ast.Expr** → 1 EI (expression statement, usually function call)
- **ast.Global** → 1 EI
- **ast.Nonlocal** → 1 EI
- **ast.Expr** (docstrings) → 0 EIs (skipped)

### Async Variants (2)
- **ast.AsyncFor** → 2-3 EIs (same as For)
- **ast.AsyncWith** → 2 EIs (same as With)

## Coverage Stats

**Total statement types in Python 3.10+:** ~30
**Covered by decomposers:** 26
**Coverage:** ~87%

## Not Explicitly Covered (handled by default)

These are either rare or don't create branching:

- **ast.Import** - Import statement (non-executable at runtime)
- **ast.ImportFrom** - From import statement (non-executable at runtime)
- **ast.ClassDef** - Class definition (container, not executable)
- **ast.FunctionDef** - Function definition (container, not executable)
- **ast.AsyncFunctionDef** - Async function definition (container, not executable)

These are handled by `decompose_default()` which returns a single EI.

## Special Patterns Recognized

### Comprehensions
- List comprehension with filter → 3 EIs
- List comprehension without filter → 2 EIs
- Dict comprehension (same pattern)
- Set comprehension (same pattern)
- Generator expression (same pattern)

### Ternary Expression
- Conditional expression (IfExp) → 4 EIs
  - Condition true → continues to true branch
  - Condition false → continues to false branch
  - True branch: assigns value
  - False branch: assigns value

### For-else / While-else
- Loop with else clause → 3 EIs
  - 0 iterations → else executes
  - Completes without break → else executes
  - Breaks → else skipped

### If with Raise/Return
- If statement with raise in body → 2 EIs with specific descriptions
- If statement with return in body → 2 EIs with specific descriptions

## Edge Cases and Notes

### Nested Statements
All statements are enumerated, including nested ones. For example:
```python
if condition:     # 2 EIs
    raise Error   # 1 EI (nested, but still counted)
```
Total: 3 EIs

This creates some redundancy but is harmless for test generation.

### Generator Expressions
Currently handled as comprehensions when assigned. Standalone generator expressions in expression statements get 1 EI.

### Walrus Operator (:=)
Treated as part of the containing expression. The assignment itself doesn't create separate EIs.

### Short-circuit Evaluation
`and` and `or` operators don't create separate EIs - they're part of the condition being evaluated.

### Chained Comparisons
`1 < x < 10` is treated as a single condition evaluation, not separate comparisons.

## Extensibility

To add a new statement type:

1. Create decomposer function:
```python
def decompose_mystmt(stmt: ast.MyStmt, source_lines: list[str]) -> list[str]:
    """MyStmt: N EIs (description)."""
    # Logic here
    return ["outcome 1", "outcome 2"]
```

2. Add to dispatch table:
```python
STATEMENT_DECOMPOSERS = {
    # ...
    ast.MyStmt: decompose_mystmt,
}
```

That's it! The framework handles the rest.