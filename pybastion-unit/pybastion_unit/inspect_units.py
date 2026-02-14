#!/usr/bin/env python3
"""
Generate callable ID inventory for a Python project.

Walks all Python files, discovers all callables (functions, methods, nested functions),
assigns IDs using callable_id_generation, and outputs FQN:ID mappings.

Usage:
    python inspect_units.py <source_root> --output <inventory_file>

Example:
    python inspect_units.py src --output dist/inspect/callable-inventory.txt
"""

import argparse
import ast
import sys
from pathlib import Path

from callable_id_generation import (
    generate_assignment_id,
    generate_unit_id,
    generate_class_id,
    generate_method_id,
    generate_function_id,
    generate_nested_function_id,
    generate_nested_class_id,
)


def derive_fqn(filepath: Path, source_root: Path) -> str:
    """
    Convert file path to fully qualified module name.

    Example:
        src/project/model/keys.py -> project.model.keys
    """
    relative = filepath.relative_to(source_root)
    parts = list(relative.parts[:-1]) + [relative.stem]

    # Remove __init__ from end
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


class CallableIDVisitor(ast.NodeVisitor):
    """
    AST visitor that discovers all callables and assigns IDs.

    Maintains counters for functions, classes, methods, and nested structures
    to generate deterministic IDs matching the callable_id_generation module.
    """

    def __init__(self, unit_id: str, module_fqn: str):
        self.unit_id = unit_id
        self.module_fqn = module_fqn
        self.mappings: dict[str, str] = {}

        # Counters
        self.function_counter = 0
        self.class_counter = 0
        self.assignment_counter = 0  # For module-level assignments
        self.method_counters: dict[str, int] = {}  # class_id -> counter
        self.nested_function_counters: dict[str, int] = {}  # parent_id -> counter
        self.nested_class_counters: dict[str, int] = {}  # parent_id -> counter

        # Context stack for tracking FQN and parent IDs
        self.fqn_stack = [module_fqn]
        self.id_stack = [unit_id]

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment statement."""
        # Only track module-level assignments (not inside functions/classes)
        if self.id_stack[-1] == self.unit_id:
            # Module-level assignment
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignment_counter += 1
                    assignment_id = generate_assignment_id(self.unit_id, self.assignment_counter)
                    fqn = f"{self.module_fqn}.{target.id}"
                    self.mappings[fqn] = assignment_id

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit an annotated assignment statement."""
        # Only track module-level assignments (not inside functions/classes)
        if self.id_stack[-1] == self.unit_id:
            # Module-level annotated assignment
            if isinstance(node.target, ast.Name):
                self.assignment_counter += 1
                assignment_id = generate_assignment_id(self.unit_id, self.assignment_counter)
                fqn = f"{self.module_fqn}.{node.target.id}"
                self.mappings[fqn] = assignment_id

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Visit an augmented assignemnt statement."""
        # Only track module-level assignments (not inside functions/classes)
        if self.id_stack[-1] == self.unit_id:
            # Module-level annotated assignment
            if isinstance(node.target, ast.Name):
                self.assignment_counter += 1
                assignment_id = generate_assignment_id(self.unit_id, self.assignment_counter)
                fqn = f"{self.module_fqn}.{node.target.id}"
                self.mappings[fqn] = assignment_id

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        parent_id = self.id_stack[-1]
        parent_fqn = self.fqn_stack[-1]

        # Determine if this is a nested class
        if parent_id == self.unit_id:
            # Top-level class
            self.class_counter += 1
            class_id = generate_class_id(self.unit_id, self.class_counter)
        else:
            # Nested class
            if parent_id not in self.nested_class_counters:
                self.nested_class_counters[parent_id] = 0
            self.nested_class_counters[parent_id] += 1
            class_id = generate_nested_class_id(parent_id, self.nested_class_counters[parent_id])

        # Build FQN
        fqn = f"{parent_fqn}.{node.name}"

        # Record mapping
        self.mappings[fqn] = class_id

        # Initialize method counter for this class
        self.method_counters[class_id] = 0

        # Push context and visit children
        self.fqn_stack.append(fqn)
        self.id_stack.append(class_id)
        self.generic_visit(node)
        self.id_stack.pop()
        self.fqn_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""
        self._visit_function_or_async(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition."""
        self._visit_function_or_async(node)

    def _visit_function_or_async(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Handle both sync and async functions."""
        parent_id = self.id_stack[-1]
        parent_fqn = self.fqn_stack[-1]

        # Determine context: unit-level function, method, or nested function
        if parent_id == self.unit_id:
            # Top-level function
            self.function_counter += 1
            callable_id = generate_function_id(self.unit_id, self.function_counter)
        elif parent_id.endswith(tuple(f'_C{i:03d}' for i in range(1, 1000))):
            # Parent is a class -> this is a method
            self.method_counters[parent_id] += 1
            callable_id = generate_method_id(parent_id, self.method_counters[parent_id])
        else:
            # Nested function
            if parent_id not in self.nested_function_counters:
                self.nested_function_counters[parent_id] = 0
            self.nested_function_counters[parent_id] += 1
            callable_id = generate_nested_function_id(parent_id, self.nested_function_counters[parent_id])

        # Build FQN
        fqn = f"{parent_fqn}.{node.name}"

        # Record mapping
        self.mappings[fqn] = callable_id

        # Push context and visit children (for nested functions/classes)
        self.fqn_stack.append(fqn)
        self.id_stack.append(callable_id)
        self.generic_visit(node)
        self.id_stack.pop()
        self.fqn_stack.pop()


def process_file(filepath: Path, source_root: Path) -> dict[str, str]:
    """
    Process a single Python file and return FQN->ID mappings.

    Returns:
        dict mapping fully qualified names to callable IDs
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return {}

    # Derive module FQN and unit ID
    module_fqn = derive_fqn(filepath, source_root)
    unit_id = generate_unit_id(module_fqn)

    # Visit AST and collect mappings
    visitor = CallableIDVisitor(unit_id, module_fqn)
    visitor.visit(tree)

    return visitor.mappings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate callable ID inventory for a Python project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s src --output dist/inspect/callable-inventory.txt

Output format:
  Each line: <fully_qualified_name>:<callable_id>

  Example:
    project.model.WheelKey:U12345678_C001
    project.model.WheelKey.__init__:U12345678_C001_M001
    project.model.WheelKey.identifier:U12345678_C001_M002
        """
    )

    parser.add_argument(
        'source_root',
        type=Path,
        help='Source root directory containing Python files'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        required=True,
        help='Output file path for callable inventory'
    )

    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if not source_root.exists():
        print(f"Error: Source root not found: {source_root}", file=sys.stderr)
        return 1

    if not source_root.is_dir():
        print(f"Error: Source root is not a directory: {source_root}", file=sys.stderr)
        return 1

    # Find all Python files
    py_files = sorted(source_root.rglob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]

    if not py_files:
        print(f"Warning: No Python files found in {source_root}", file=sys.stderr)
        return 1

    print(f"Processing {len(py_files)} Python files...")

    # Collect all mappings
    all_mappings: dict[str, str] = {}
    for py_file in py_files:
        mappings = process_file(py_file, source_root)
        all_mappings.update(mappings)

    print(f"Found {len(all_mappings)} callables")

    # Sort by FQN for consistent output
    sorted_mappings = sorted(all_mappings.items())

    # Write to output file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for fqn, callable_id in sorted_mappings:
            f.write(f"{fqn}:{callable_id}\n")

    print(f"Wrote callable inventory to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())