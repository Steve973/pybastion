#!/usr/bin/env python3
"""
Shared AST indexing and type extraction helpers.

Purpose:
- centralize per-unit AST indexing
- build import maps and local return type maps
- extract class field types using the shared TypeRef model
- support both stage 1 project-wide registry building and stage 3 unit analysis

This module intentionally stops at metadata extraction.
It does not perform chain resolution or integration classification.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from pybastion_common.models import TypeRef


@dataclass(frozen=True)
class ModuleAstIndex:
    unit_fqn: str
    class_nodes: dict[str, ast.ClassDef]
    callable_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    nodes_by_fqn_and_line: dict[tuple[str, int], ast.AST]
    assignment_nodes_by_fqn_and_line: dict[tuple[str, int], ast.Assign | ast.AnnAssign | ast.AugAssign]
    module_symbol_fqns: set[str]


@dataclass(frozen=True)
class IndexedUnitTypes:
    """
    Shared extracted type metadata for one module.

    Attributes:
        source_path: File path for the source unit.
        unit_fqn: Fully qualified module name.
        import_map: Local symbol to imported fully qualified name.
        interunit_imports: Imported symbols that resolve to in-project names.
        local_return_types: Local function or method simple name to declared return type.
        ast_index: Shared AST index for the module.
        field_types_by_class: Class FQN to field name to type reference.
    """
    source_path: str
    unit_fqn: str
    import_map: dict[str, str]
    interunit_imports: set[str]
    local_return_types: dict[str, TypeRef]
    ast_index: ModuleAstIndex
    field_types_by_class: dict[str, dict[str, TypeRef]]


class ModuleAstIndexer(ast.NodeVisitor):
    def __init__(self, unit_fqn: str):
        self.unit_fqn = unit_fqn
        self.scope_stack: list[str] = [unit_fqn]

        self.class_nodes: dict[str, ast.ClassDef] = {}
        self.callable_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.nodes_by_fqn_and_line: dict[tuple[str, int], ast.AST] = {}
        self.assignment_nodes_by_fqn_and_line: dict[
            tuple[str, int],
            ast.Assign | ast.AnnAssign | ast.AugAssign,
        ] = {}
        self.module_symbol_fqns: set[str] = set()

    def current_prefix(self) -> str:
        return ".".join(self.scope_stack)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        fqn = f"{self.current_prefix()}.{node.name}"
        self.class_nodes[fqn] = node
        self.nodes_by_fqn_and_line[(fqn, node.lineno)] = node
        self.module_symbol_fqns.add(fqn)

        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _visit_function_like(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        fqn = f"{self.current_prefix()}.{node.name}"
        self.callable_nodes[fqn] = node
        self.nodes_by_fqn_and_line[(fqn, node.lineno)] = node
        self.module_symbol_fqns.add(fqn)

        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                fqn = f"{self.unit_fqn}.{target.id}"
                self.assignment_nodes_by_fqn_and_line[(fqn, node.lineno)] = node
                self.module_symbol_fqns.add(fqn)
                break
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            fqn = f"{self.unit_fqn}.{node.target.id}"
            self.assignment_nodes_by_fqn_and_line[(fqn, node.lineno)] = node
            self.module_symbol_fqns.add(fqn)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            fqn = f"{self.unit_fqn}.{node.target.id}"
            self.assignment_nodes_by_fqn_and_line[(fqn, node.lineno)] = node
            self.module_symbol_fqns.add(fqn)
        self.generic_visit(node)


def parse_module(source_path: Path) -> ast.Module:
    source = source_path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(source_path))


def self_attribute_name(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "self":
        return expr.attr
    return None


def build_import_map(
        tree: ast.Module,
        project_fqns: set[str] | None = None,
) -> tuple[dict[str, str], set[str]]:
    """
    Build an import map for a module.

    Returns:
        (
            import_map,
            interunit_imports,
        )

    interunit_imports is populated only when project_fqns is provided.
    """
    import_map: dict[str, str] = {}
    interunit_imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                import_map[local_name] = alias.name
                if project_fqns is not None and alias.name in project_fqns:
                    interunit_imports.add(local_name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                fqn = f"{module}.{alias.name}" if module else alias.name
                import_map[local_name] = fqn
                if project_fqns is not None and fqn in project_fqns:
                    interunit_imports.add(local_name)

    return import_map, interunit_imports


def build_local_return_type_map(tree: ast.Module) -> dict[str, TypeRef]:
    """
    Build a simple-name -> declared return TypeRef map for locally declared callables.
    """
    local_return_types: dict[str, TypeRef] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            type_ref = TypeRef.from_annotation_ast(node.returns)
            if type_ref is not None:
                local_return_types[node.name] = type_ref

    return local_return_types


def build_module_index(tree: ast.Module, unit_fqn: str) -> ModuleAstIndex:
    indexer = ModuleAstIndexer(unit_fqn)
    indexer.visit(tree)
    return ModuleAstIndex(
        unit_fqn=unit_fqn,
        class_nodes=indexer.class_nodes,
        callable_nodes=indexer.callable_nodes,
        nodes_by_fqn_and_line=indexer.nodes_by_fqn_and_line,
        assignment_nodes_by_fqn_and_line=indexer.assignment_nodes_by_fqn_and_line,
        module_symbol_fqns=indexer.module_symbol_fqns,
    )


def build_field_types_by_class(
        *,
        class_nodes: dict[str, ast.ClassDef],
        import_map: dict[str, str],
        local_return_types: dict[str, TypeRef],
) -> dict[str, dict[str, TypeRef]]:
    """
    Build field type metadata for each class in a unit.

    Extraction sources:
    - annotated class attributes
    - annotated self attributes in __init__
    - self.field = param where param is typed
    - self.field = local_callable(...) where the local callable has a declared return type

    Important:
    - preserves TypeRef structure
    - normalizes top-level imported type names through import_map
    """
    field_types_by_class: dict[str, dict[str, TypeRef]] = {}

    for class_fqn, class_node in class_nodes.items():
        field_map: dict[str, TypeRef] = {}

        # Class body annotations
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign):
                attr_name = self_attribute_name(stmt.target)
                type_ref = TypeRef.from_annotation_ast(stmt.annotation)

                if attr_name and type_ref is not None:
                    field_map[attr_name] = type_ref
                elif isinstance(stmt.target, ast.Name) and type_ref is not None:
                    field_map[stmt.target.id] = type_ref

        # __init__ based extraction
        init_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        for stmt in class_node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
                init_node = stmt
                break

        if init_node is not None:
            param_types: dict[str, TypeRef] = {}

            for arg in init_node.args.args + init_node.args.kwonlyargs:
                type_ref = TypeRef.from_annotation_ast(arg.annotation)
                if type_ref is not None:
                    param_types[arg.arg] = type_ref

            for stmt in ast.walk(init_node):
                if isinstance(stmt, ast.AnnAssign):
                    attr_name = self_attribute_name(stmt.target)
                    type_ref = TypeRef.from_annotation_ast(stmt.annotation)
                    if attr_name and type_ref is not None:
                        field_map[attr_name] = type_ref

                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    attr_name = self_attribute_name(stmt.targets[0])
                    if not attr_name:
                        continue

                    value = stmt.value

                    if isinstance(value, ast.Name) and value.id in param_types:
                        field_map[attr_name] = param_types[value.id]
                        continue

                    if isinstance(value, ast.Call):
                        func = value.func
                        if isinstance(func, ast.Name) and func.id in local_return_types:
                            field_map[attr_name] = local_return_types[func.id]
                            continue
                        if isinstance(func, ast.Attribute) and func.attr in local_return_types:
                            field_map[attr_name] = local_return_types[func.attr]
                            continue

        # Normalize imported top-level type names
        normalized: dict[str, TypeRef] = {}
        for field_name, type_ref in field_map.items():
            if type_ref.name in import_map:
                normalized[field_name] = TypeRef(
                    name=import_map[type_ref.name],
                    args=type_ref.args,
                )
            else:
                normalized[field_name] = type_ref

        field_types_by_class[class_fqn] = normalized

    return field_types_by_class


def inspect_unit_types(
    *,
    unit_fqn: str,
    source_path: Path | None = None,
    tree: ast.Module | None = None,
    ast_index: ModuleAstIndex | None = None,
    project_fqns: set[str] | None = None,
) -> IndexedUnitTypes:
    """
    Build shared type metadata for one module.

    Exactly one of:
    - source_path
    - tree
    - (tree and ast_index)
    should be supplied as the source of truth.

    If ast_index is provided, it is reused instead of rebuilding the module index.
    """
    if source_path is None and tree is None:
        raise ValueError("Provide source_path or tree")

    if tree is None:
        assert source_path is not None
        tree = parse_module(source_path)
        resolved_source_path = str(source_path)
    else:
        resolved_source_path = str(source_path) if source_path is not None else ""

    import_map, interunit_imports = build_import_map(tree, project_fqns=project_fqns)
    local_return_types = build_local_return_type_map(tree)

    if ast_index is None:
        ast_index = build_module_index(tree, unit_fqn)

    field_types_by_class = build_field_types_by_class(
        class_nodes=ast_index.class_nodes,
        import_map=import_map,
        local_return_types=local_return_types,
    )

    return IndexedUnitTypes(
        source_path=resolved_source_path,
        unit_fqn=unit_fqn,
        import_map=import_map,
        interunit_imports=interunit_imports,
        local_return_types=local_return_types,
        ast_index=ast_index,
        field_types_by_class=field_types_by_class,
    )


def serialize_type_map(type_map: dict[str, TypeRef]) -> dict[str, dict]:
    return {key: value.to_dict() for key, value in type_map.items()}


def serialize_nested_type_map(
        type_map: dict[str, dict[str, TypeRef]],
) -> dict[str, dict[str, dict]]:
    return {
        owner: {
            field_name: field_type.to_dict()
            for field_name, field_type in fields.items()
        }
        for owner, fields in type_map.items()
    }
