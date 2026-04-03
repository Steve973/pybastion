#!/usr/bin/env python3
"""
Build a deterministic Python unit index.

This upgrades the original stage 1 from a flat FQN -> ID inventory pass into a
structured unit index pass while preserving deterministic traversal and the
existing callable ID generation rules.

Outputs:
  1. A structured JSON index containing per-unit metadata and discovered entries
  2. Optionally, a legacy flat inventory file in the original <fqn>:<id> format

Usage:
    python stage1_inspect_units_thicc.py <source_root> --output <index.json>
    python stage1_inspect_units_thicc.py <source_root> --output <index.json> \
        --legacy-output <callable-inventory.txt>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pybastion_common.models import UnitIndexEntry, UnitIndex, ProjectIndex
from pybastion_unit.shared.callable_id_generation import (
    generate_assignment_id,
    generate_class_id,
    generate_function_id,
    generate_method_id,
    generate_nested_class_id,
    generate_nested_function_id,
    generate_unit_id,
)

ScopeKind = Literal[
    "unit",
    "class",
    "nested_class",
    "unit_function",
    "method",
    "nested_function",
]

EntryKind = Literal[
    "module_assignment",
    "class",
    "nested_class",
    "unit_function",
    "method",
    "nested_function",
]


@dataclass(slots=True)
class ScopeFrame:
    id: str
    fqn: str
    kind: ScopeKind


def derive_fqn(filepath: Path, source_root: Path) -> str:
    """Convert file path to a fully qualified module name."""
    relative = filepath.relative_to(source_root)
    parts = list(relative.parts[:-1]) + [relative.stem]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def hash_source(source: str) -> str:
    """Return a stable content hash for the source text."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class UnitIndexVisitor(ast.NodeVisitor):
    """
    Deterministic AST visitor that builds a structured unit index.

    This intentionally stays at the scope / identity layer. It does not perform
    statement decomposition, CFG construction, or EI generation.
    """

    def __init__(self, unit_id: str, module_fqn: str):
        self.unit_id = unit_id
        self.module_fqn = module_fqn
        self.entries: list[UnitIndexEntry] = []
        self.entry_by_id: dict[str, UnitIndexEntry] = {}

        # Legacy flat mapping is kept because it is cheap to derive and useful
        # during migration.
        self.mappings: dict[str, str] = {}

        # Global top-level counters.
        self.function_counter = 0
        self.class_counter = 0
        self.assignment_counter = 0

        # Per-parent counters.
        self.method_counters: dict[str, int] = {}
        self.nested_function_counters: dict[str, int] = {}
        self.nested_class_counters: dict[str, int] = {}
        self.ordinal_counters: dict[str, int] = {unit_id: 0}

        self.scope_stack: list[ScopeFrame] = [
            ScopeFrame(id=unit_id, fqn=module_fqn, kind="unit")
        ]

    def current_scope(self) -> ScopeFrame:
        return self.scope_stack[-1]

    def next_ordinal(self, owner_id: str) -> int:
        current = self.ordinal_counters.get(owner_id, 0) + 1
        self.ordinal_counters[owner_id] = current
        return current

    def add_entry(
            self,
            *,
            entry_id: str,
            kind: EntryKind,
            name: str,
            fqn: str,
            parent_id: str | None,
            owner_id: str,
            lineno: int,
            end_lineno: int,
            ordinal_within_parent: int,
            is_async: bool = False,
    ) -> None:
        entry = UnitIndexEntry(
            id=entry_id,
            kind=kind,
            name=name,
            fully_qualified_name=fqn,
            parent_id=parent_id,
            owner_id=owner_id,
            lineno=lineno,
            end_lineno=end_lineno,
            ordinal_within_parent=ordinal_within_parent,
            is_async=is_async,
        )
        self.entries.append(entry)
        self.entry_by_id[entry_id] = entry
        self.mappings[fqn] = entry_id

        if parent_id and parent_id in self.entry_by_id:
            self.entry_by_id[parent_id].child_ids.append(entry_id)

    def visit_Assign(self, node: ast.Assign) -> None:
        scope = self.current_scope()
        if scope.kind == "unit":
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignment_counter += 1
                    assignment_id = generate_assignment_id(self.unit_id, self.assignment_counter)
                    fqn = f"{self.module_fqn}.{target.id}"
                    self.add_entry(
                        entry_id=assignment_id,
                        kind="module_assignment",
                        name=target.id,
                        fqn=fqn,
                        parent_id=self.unit_id,
                        owner_id=self.unit_id,
                        lineno=node.lineno,
                        end_lineno=getattr(node, "end_lineno", node.lineno),
                        ordinal_within_parent=self.next_ordinal(self.unit_id),
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        scope = self.current_scope()
        if scope.kind == "unit" and isinstance(node.target, ast.Name):
            self.assignment_counter += 1
            assignment_id = generate_assignment_id(self.unit_id, self.assignment_counter)
            fqn = f"{self.module_fqn}.{node.target.id}"
            self.add_entry(
                entry_id=assignment_id,
                kind="module_assignment",
                name=node.target.id,
                fqn=fqn,
                parent_id=self.unit_id,
                owner_id=self.unit_id,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                ordinal_within_parent=self.next_ordinal(self.unit_id),
            )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        scope = self.current_scope()
        if scope.kind == "unit" and isinstance(node.target, ast.Name):
            self.assignment_counter += 1
            assignment_id = generate_assignment_id(self.unit_id, self.assignment_counter)
            fqn = f"{self.module_fqn}.{node.target.id}"
            self.add_entry(
                entry_id=assignment_id,
                kind="module_assignment",
                name=node.target.id,
                fqn=fqn,
                parent_id=self.unit_id,
                owner_id=self.unit_id,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                ordinal_within_parent=self.next_ordinal(self.unit_id),
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent_scope = self.current_scope()
        parent_id = parent_scope.id
        parent_fqn = parent_scope.fqn

        if parent_scope.kind == "unit":
            self.class_counter += 1
            class_id = generate_class_id(self.unit_id, self.class_counter)
            kind: EntryKind = "class"
        else:
            self.nested_class_counters[parent_id] = self.nested_class_counters.get(parent_id, 0) + 1
            class_id = generate_nested_class_id(parent_id, self.nested_class_counters[parent_id])
            kind = "nested_class"

        fqn = f"{parent_fqn}.{node.name}"
        self.method_counters[class_id] = 0
        self.ordinal_counters.setdefault(class_id, 0)

        self.add_entry(
            entry_id=class_id,
            kind=kind,
            name=node.name,
            fqn=fqn,
            parent_id=parent_id,
            owner_id=parent_id,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            ordinal_within_parent=self.next_ordinal(parent_id),
        )

        self.scope_stack.append(
            ScopeFrame(
                id=class_id,
                fqn=fqn,
                kind="class" if kind == "class" else "nested_class",
            )
        )
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node, is_async=True)

    def _visit_function_like(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef,
            *,
            is_async: bool,
    ) -> None:
        parent_scope = self.current_scope()
        parent_id = parent_scope.id
        parent_fqn = parent_scope.fqn

        if parent_scope.kind == "unit":
            self.function_counter += 1
            callable_id = generate_function_id(self.unit_id, self.function_counter)
            kind: EntryKind = "unit_function"
        elif parent_scope.kind in {"class", "nested_class"}:
            self.method_counters[parent_id] = self.method_counters.get(parent_id, 0) + 1
            callable_id = generate_method_id(parent_id, self.method_counters[parent_id])
            kind = "method"
        else:
            self.nested_function_counters[parent_id] = self.nested_function_counters.get(parent_id, 0) + 1
            callable_id = generate_nested_function_id(
                parent_id,
                self.nested_function_counters[parent_id],
            )
            kind = "nested_function"

        fqn = f"{parent_fqn}.{node.name}"
        self.ordinal_counters.setdefault(callable_id, 0)

        self.add_entry(
            entry_id=callable_id,
            kind=kind,
            name=node.name,
            fqn=fqn,
            parent_id=parent_id,
            owner_id=parent_id,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            ordinal_within_parent=self.next_ordinal(parent_id),
            is_async=is_async,
        )

        nested_scope_kind: ScopeKind
        match kind:
            case "unit_function":
                nested_scope_kind = "unit_function"
            case "method":
                nested_scope_kind = "method"
            case "nested_function":
                nested_scope_kind = "nested_function"
            case _:
                raise ValueError(f"Unexpected function-like entry kind: {kind}")

        self.scope_stack.append(
            ScopeFrame(
                id=callable_id,
                fqn=fqn,
                kind=nested_scope_kind,
            )
        )
        self.generic_visit(node)
        self.scope_stack.pop()


def process_file(filepath: Path, source_root: Path) -> tuple[UnitIndex, dict[str, str]] | None:
    """Parse one Python file and build a structured unit index."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        print(f"Syntax error in {filepath}: {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"Error parsing {filepath}: {exc}", file=sys.stderr)
        return None

    module_fqn = derive_fqn(filepath, source_root)
    unit_id = generate_unit_id(module_fqn)

    visitor = UnitIndexVisitor(unit_id, module_fqn)
    visitor.visit(tree)

    unit_index = UnitIndex(
        unit_id=unit_id,
        fully_qualified_name=module_fqn,
        filepath=str(filepath),
        language="python",
        source_hash=hash_source(source),
        entries=visitor.entries,
    )
    return unit_index, visitor.mappings


def build_project_index(source_root: Path) -> tuple[ProjectIndex, dict[str, str]]:
    """Build a project-wide index for all Python units under the source root."""
    py_files = sorted(source_root.rglob("*.py"))
    py_files = [path for path in py_files if path.name != "__init__.py"]

    if not py_files:
        raise ValueError(f"No Python files found in {source_root}")

    units: list[UnitIndex] = []
    all_mappings: dict[str, str] = {}

    for py_file in py_files:
        result = process_file(py_file, source_root)
        if result is None:
            continue
        unit_index, mappings = result
        units.append(unit_index)
        all_mappings.update(mappings)

    project_index = ProjectIndex(source_root=str(source_root), units=units)
    return project_index, all_mappings


def write_project_index(output_path: Path, project_index: ProjectIndex) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(project_index)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_legacy_inventory(output_path: Path, mappings: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{fqn}:{entry_id}" for fqn, entry_id in sorted(mappings.items())]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Python unit index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s src --output dist/inspect/unit-index.json
  %(prog)s src --output dist/inspect/unit-index.json --legacy-output dist/inspect/callable-inventory.txt
        """,
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help="Source root directory containing Python files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSON file for the structured unit index",
    )
    parser.add_argument(
        "--legacy-output",
        type=Path,
        help="Optional legacy flat inventory output in <fqn>:<id> format",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()

    if not source_root.exists():
        print(f"Error: Source root not found: {source_root}", file=sys.stderr)
        return 1
    if not source_root.is_dir():
        print(f"Error: Source root is not a directory: {source_root}", file=sys.stderr)
        return 1

    try:
        project_index, mappings = build_project_index(source_root)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    write_project_index(args.output, project_index)
    if args.legacy_output:
        write_legacy_inventory(args.legacy_output, mappings)

    total_entries = sum(len(unit.entries) for unit in project_index.units)
    print(f"Indexed {len(project_index.units)} Python units")
    print(f"Discovered {total_entries} indexed entries")
    print(f"Wrote structured unit index to {args.output}")
    if args.legacy_output:
        print(f"Wrote legacy flat inventory to {args.legacy_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
