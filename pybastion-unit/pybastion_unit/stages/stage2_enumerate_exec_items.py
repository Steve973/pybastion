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
from typing import Any

import yaml

from pybastion_unit.helpers.constraint_metadata_helper import enrich_outcome_with_constraint, \
    populate_constraint_relationships
from pybastion_unit.helpers.statement_decomposition import decompose_statement
from pybastion_unit.shared.callable_id_generation import generate_function_id, generate_ei_id, generate_assignment_id
from pybastion_unit.shared.models import Branch

ENUM_BASES = {'Enum', 'IntEnum', 'StrEnum', 'Flag', 'IntFlag'}


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
            for outcome, call_node in outcomes:
                ei_id = generate_ei_id(callable_id, ei_counter)

                condition, result, constraint = enrich_outcome_with_constraint(
                    outcome,
                    call_node,
                    stmt,
                    ei_id,
                    stmt.lineno
                )

                branches.append(
                    Branch(
                        id=ei_id,
                        line=stmt.lineno,
                        condition=condition,
                        outcome=result,
                        constraint=constraint
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
        self.function_depth = 0

    @staticmethod
    def _is_enum_class(node: ast.ClassDef) -> bool:
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in ENUM_BASES:
                return True
            if isinstance(base, ast.Attribute) and base.attr in ENUM_BASES:
                return True
        return False

    def _enumerate_and_record(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, fqn: str) -> None:
        callable_id = self.inventory.get(fqn)
        if not callable_id:
            callable_id = generate_function_id(self.unit_id, self.func_counter)
            print(f"Warning: {fqn} not in inventory, generated {callable_id}")

        result = enumerate_function_eis(node, self.source_lines, callable_id)
        populate_constraint_relationships(result.branches)
        self.results.append(result)
        self.func_counter += 1

    def visit_Assign(self, node) -> None:
        self._process_assignment(node)

    def visit_AnnAssign(self, node) -> None:
        if self.function_depth == 0:
            self._process_assignment(node)

    def visit_AugAssign(self, node) -> None:
        self._process_assignment(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.fqn_stack.append(node.name)

        if self._is_enum_class(node):
            # Emit the enum class itself as an entry so it appears in the ledger
            fqn = '.'.join(self.fqn_stack)
            self._enumerate_and_record(node, fqn)
            # Don't generic_visit — enum members aren't callable children
        else:
            self.generic_visit(node)

        self.fqn_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_depth += 1
        self._process_function(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_depth += 1
        self._process_function(node)
        self.function_depth -= 1

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
        self._enumerate_and_record(node, fqn)

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

        outcomes = decompose_statement(node, self.source_lines)

        if outcomes:
            for outcome, call_node in outcomes:
                ei_counter += 1
                ei_id = generate_ei_id(callable_id, ei_counter)

                # Extract constraint metadata
                condition, result, constraint = enrich_outcome_with_constraint(
                    outcome,
                    call_node,
                    node,
                    ei_id,
                    node.lineno,
                )

                branches.append(
                    Branch(
                        id=ei_id,
                        line=node.lineno,
                        condition=condition,
                        outcome=result,
                        constraint=constraint
                    )
                )

            function_result = FunctionResult(
                name=target_name,
                line_start=node.lineno,
                line_end=node.end_lineno,
                branches=branches
            )
            populate_constraint_relationships(branches)
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
