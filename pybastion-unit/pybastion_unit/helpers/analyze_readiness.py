#!/usr/bin/env python3
"""
PyBastion source readiness scanner.

This is a source-only preflight scanner. It examines a Python source tree before
PyBastion inventory, graph, or test-spec generation runs.

The report is intentionally findings-focused. It exists to show code locations
that may be worth improving so PyBastion can resolve call targets, infer receiver
and return types, model seams, and extract useful integration test specifications.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

from pybastion_unit.helpers.type_indexing import inspect_unit_types
from pybastion_unit.stages.stage1_inspect_units import derive_fqn, process_file

Severity = Literal["blocking", "high_impact", "advisory", "info"]
CallableNode = ast.FunctionDef | ast.AsyncFunctionDef

SEVERITY_ORDER: dict[str, int] = {
    "blocking": 0,
    "high_impact": 1,
    "advisory": 2,
    "info": 3,
}

BUILTIN_RECEIVERS: set[str] = {
    "object",
    "type",
    "super",
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "bytearray",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
}

DYNAMIC_DISPATCH_NAMES: set[str] = {
    "getattr",
    "setattr",
    "hasattr",
    "globals",
    "locals",
    "eval",
    "exec",
    "__import__",
}

MECHANICAL_NAME_HINTS: tuple[str, ...] = (
    "from_dict",
    "to_dict",
    "from_mapping",
    "to_mapping",
    "from_json",
    "to_json",
    "serialize",
    "deserialize",
    "asdict",
    "model_validate",
    "summary",
)

UTILITY_NAME_HINTS: tuple[str, ...] = (
    "normalize",
    "canonicalize",
    "sanitize",
    "validate",
    "coerce",
    "convert",
    "format",
    "parse_",
)

CONTAINER_NAMES: set[str] = {
    "list",
    "List",
    "Sequence",
    "Iterable",
    "Iterator",
    "Collection",
    "set",
    "Set",
    "tuple",
    "Tuple",
    "dict",
    "Dict",
    "Mapping",
    "MutableMapping",
}

MODELABLE_PREDICATE_METHODS: set[str] = {
    "startswith",
    "endswith",
    "exists",
    "is_file",
    "is_dir",
    "is_absolute",
    "is_relative_to",
    "isdigit",
    "isalnum",
    "isalpha",
    "islower",
    "isupper",
    "match",
    "fullmatch",
    "contains",
}

MODELABLE_PREDICATE_FUNCTIONS: set[str] = {
    "isinstance",
    "issubclass",
    "callable",
    "bool",
    "len",
}


@dataclass(slots=True)
class Finding:
    finding_type: str
    severity: Severity
    file: str
    line: int
    message: str
    suggestion: str | None = None
    example: str | None = None
    callable_name: str | None = None
    symbol: str | None = None
    observed: str | None = None
    confidence: str = "medium"


@dataclass(slots=True)
class FunctionContext:
    node: CallableNode
    class_name: str | None
    qualified_name: str
    param_annotations: dict[str, str | None]
    return_annotation: str | None
    local_annotations: dict[str, str]
    local_assignments: dict[str, ast.expr]
    empty_container_assignments: set[str]
    loop_sources: dict[str, ast.expr]
    imported_names: set[str]
    module_annotations: dict[str, str]
    class_member_annotations: dict[str, str]
    known_receiver_names: set[str]
    inferred_receiver_types: dict[str, str]
    calls_by_receiver: dict[str, list[ast.Call]] = field(default_factory=dict)


@dataclass(slots=True)
class ReadinessReport:
    source_root: str
    files_scanned: int
    python_files_scanned: int
    finding_count: int
    severity_counts: dict[str, int]
    finding_type_counts: dict[str, int]
    grouping: str
    findings: list[Finding]


@dataclass(slots=True)
class Config:
    exclude_dirs: set[str] = field(default_factory=lambda: {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        "site-packages",
        "node_modules",
    })
    include_info: bool = True
    max_findings_per_type: int | None = None


def load_config(path: Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    readiness = data.get("readiness", {}) if isinstance(data, dict) else {}

    exclude_dirs = readiness.get("exclude_dirs")
    if isinstance(exclude_dirs, list):
        cfg.exclude_dirs = {str(item) for item in exclude_dirs}

    include_info = readiness.get("include_info")
    if isinstance(include_info, bool):
        cfg.include_info = include_info

    max_findings = readiness.get("max_findings_per_type")
    if isinstance(max_findings, int) and max_findings > 0:
        cfg.max_findings_per_type = max_findings

    return cfg


def iter_python_files(source_root: Path, exclude_dirs: set[str]) -> Iterable[Path]:
    for path in source_root.rglob("*.py"):
        if set(path.parts) & exclude_dirs:
            continue
        yield path


def safe_unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node).strip()
    except Exception:
        return None


def annotation_to_string(annotation: ast.AST | None) -> str | None:
    return safe_unparse(annotation)


def is_any_annotation(annotation: str | None) -> bool:
    if not annotation:
        return False
    normalized = annotation.replace("typing.", "")
    return normalized == "Any" or normalized.endswith(".Any")


def split_top_level_comma(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def strip_final_annotation(annotation: str) -> str:
    text = annotation.strip()
    if "[" not in text or not text.endswith("]"):
        return text

    root, inner = text.split("[", 1)
    if root.split(".")[-1] != "Final":
        return text

    return inner[:-1].strip()


def iterable_element_type(annotation: str | None) -> str | None:
    if not annotation:
        return None

    annotation = strip_final_annotation(annotation)
    if "[" not in annotation or "]" not in annotation:
        return None

    root, rest = annotation.split("[", 1)
    root = root.split(".")[-1]
    inner = rest.rsplit("]", 1)[0].strip()
    if not inner:
        return None

    if root in {"list", "List", "Sequence", "Iterable", "Iterator", "Collection", "set", "Set"}:
        return inner

    if root in {"tuple", "Tuple"}:
        parts = split_top_level_comma(inner)
        if len(parts) == 2 and parts[1] == "...":
            return parts[0]
        if len(parts) == 1:
            return parts[0]

    if root in {"dict", "Dict", "Mapping", "MutableMapping"}:
        parts = split_top_level_comma(inner)
        if len(parts) >= 2:
            return parts[0].strip()

    return None


def mapping_item_tuple_type(annotation: str | None) -> str | None:
    if not annotation:
        return None

    annotation = strip_final_annotation(annotation)
    if "[" not in annotation or "]" not in annotation:
        return None

    root, rest = annotation.split("[", 1)
    root = root.split(".")[-1]
    inner = rest.rsplit("]", 1)[0].strip()
    if root not in {"dict", "Dict", "Mapping", "MutableMapping"}:
        return None

    parts = split_top_level_comma(inner)
    if len(parts) < 2:
        return None

    return f"tuple[{parts[0].strip()}, {parts[1].strip()}]"


def tuple_item_types(annotation: str | None) -> list[str]:
    if not annotation:
        return []

    annotation = strip_final_annotation(annotation)
    if "[" not in annotation or "]" not in annotation:
        return []

    root, rest = annotation.split("[", 1)
    root = root.split(".")[-1]
    if root not in {"tuple", "Tuple"}:
        return []

    inner = rest.rsplit("]", 1)[0].strip()
    parts = split_top_level_comma(inner)
    if len(parts) == 2 and parts[1] == "...":
        return [parts[0]]
    return parts


def receiver_name(call: ast.Call) -> tuple[str | None, str | None]:
    if isinstance(call.func, ast.Attribute):
        return safe_unparse(call.func.value), call.func.attr
    if isinstance(call.func, ast.Name):
        return None, call.func.id
    return None, safe_unparse(call.func)


def simple_name(expr: ast.AST | None) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    return None


def self_attribute_name(expr: ast.AST | None) -> str | None:
    if (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "self"
    ):
        return expr.attr
    return None


def is_empty_container_expr(expr: ast.expr) -> bool:
    if isinstance(expr, (ast.List, ast.Set, ast.Tuple)):
        return len(expr.elts) == 0
    if isinstance(expr, ast.Dict):
        return len(expr.keys) == 0
    if isinstance(expr, ast.Call):
        return (
                isinstance(expr.func, ast.Name)
                and expr.func.id in {"list", "dict", "set", "tuple"}
                and not expr.args
                and not expr.keywords
        )
    return False


def is_internal_or_builtin_receiver(name: str) -> bool:
    return (
            name in {"self", "cls", "super"}
            or name in BUILTIN_RECEIVERS
            or name.startswith("typing.")
    )


def decorator_names(node: CallableNode) -> set[str]:
    names: set[str] = set()
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            names.add(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            names.add(decorator.attr)
            full = safe_unparse(decorator)
            if full:
                names.add(full)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                names.add(decorator.func.id)
            elif isinstance(decorator.func, ast.Attribute):
                names.add(decorator.func.attr)
                full = safe_unparse(decorator.func)
                if full:
                    names.add(full)
    return names


def has_marker_comment(lines: list[str], node: ast.AST, marker_names: set[str], window: int = 4) -> bool:
    lineno = getattr(node, "lineno", 1)
    start = max(0, lineno - window - 1)
    end = max(0, lineno - 1)
    for line in lines[start:end]:
        if "::" not in line:
            continue
        if any(marker in line for marker in marker_names):
            return True
    return False


def imported_names_from_unit_types(unit_types: Any) -> set[str]:
    import_map = getattr(unit_types, "import_map", {}) or {}
    return {str(name) for name in import_map.keys()}


def module_annotations_from_unit_index(unit_index: Any) -> dict[str, str]:
    annotations: dict[str, str] = {}

    for binding in getattr(unit_index, "bindings", []) or []:
        annotation = getattr(binding, "annotation", None)
        name = getattr(binding, "name", None)
        if name and annotation:
            annotations[str(name)] = str(annotation)

    return annotations


def known_names_from_unit_index(unit_index: Any) -> set[str]:
    names: set[str] = set()

    for entry in getattr(unit_index, "entries", []) or []:
        name = getattr(entry, "name", None)
        if name:
            names.add(str(name))

    for binding in getattr(unit_index, "bindings", []) or []:
        name = getattr(binding, "name", None)
        if name:
            names.add(str(name))

    return names


def annotation_from_type_ref_dict(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    name = payload.get("name")
    if not name:
        return None

    args = payload.get("args") or []
    if not args:
        return str(name)

    rendered_args = [annotation_from_type_ref_dict(arg) or str(arg) for arg in args]
    return f"{name}[{', '.join(rendered_args)}]"


def field_annotations_from_registry(
        field_registry: dict[str, dict[str, dict[str, object]]],
        class_fqn: str,
) -> dict[str, str]:
    fields = field_registry.get(class_fqn, {}) or {}
    annotations: dict[str, str] = {}

    for field_name, payload in fields.items():
        annotation = annotation_from_type_ref_dict(payload)
        if annotation:
            annotations[str(field_name)] = annotation

    return annotations


def property_return_annotations_from_class(node: ast.ClassDef) -> dict[str, str]:
    annotations: dict[str, str] = {}

    for child in node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "property" not in decorator_names(child):
            continue

        annotation = annotation_to_string(child.returns)
        if annotation:
            annotations[child.name] = annotation

    return annotations


def type_for_iterable_expr(
        expr: ast.AST,
        *,
        local_annotations: dict[str, str],
        param_annotations: dict[str, str | None],
        module_annotations: dict[str, str],
        class_member_annotations: dict[str, str],
) -> str | None:
    name = simple_name(expr)
    if name:
        return (
                local_annotations.get(name)
                or param_annotations.get(name)
                or module_annotations.get(name)
        )

    attr_name = self_attribute_name(expr)
    if attr_name:
        return class_member_annotations.get(attr_name)

    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        method = expr.func.attr
        value = expr.func.value

        if method == "items":
            base_type = type_for_iterable_expr(
                value,
                local_annotations=local_annotations,
                param_annotations=param_annotations,
                module_annotations=module_annotations,
                class_member_annotations=class_member_annotations,
            )
            return mapping_item_tuple_type(base_type)

        if method in {"keys", "values"}:
            base_type = type_for_iterable_expr(
                value,
                local_annotations=local_annotations,
                param_annotations=param_annotations,
                module_annotations=module_annotations,
                class_member_annotations=class_member_annotations,
            )
            tuple_type = mapping_item_tuple_type(base_type)
            parts = tuple_item_types(tuple_type)
            if method == "keys" and parts:
                return parts[0]
            if method == "values" and len(parts) >= 2:
                return parts[1]

    return None


def iteration_value_type_for_expr(
        expr: ast.AST,
        *,
        local_annotations: dict[str, str],
        param_annotations: dict[str, str | None],
        module_annotations: dict[str, str],
        class_member_annotations: dict[str, str],
) -> str | None:
    iterable_type = type_for_iterable_expr(
        expr,
        local_annotations=local_annotations,
        param_annotations=param_annotations,
        module_annotations=module_annotations,
        class_member_annotations=class_member_annotations,
    )

    if iterable_type is None:
        return None

    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        method = expr.func.attr
        if method in {"items", "keys", "values"}:
            return iterable_type

    return iterable_element_type(iterable_type)


def bind_target_type(
        target: ast.AST,
        value_type: str | None,
        inferred_receiver_types: dict[str, str],
) -> None:
    if not value_type:
        return

    if isinstance(target, ast.Name):
        inferred_receiver_types[target.id] = value_type
        return

    if isinstance(target, (ast.Tuple, ast.List)):
        item_types = tuple_item_types(value_type)
        if not item_types:
            return

        if len(item_types) == 1:
            item_types = item_types * len(target.elts)

        for elt, item_type in zip(target.elts, item_types):
            if isinstance(elt, ast.Name):
                inferred_receiver_types[elt.id] = item_type


def bind_comprehension_target_types(
        node: CallableNode,
        *,
        local_annotations: dict[str, str],
        param_annotations: dict[str, str | None],
        module_annotations: dict[str, str],
        class_member_annotations: dict[str, str],
        inferred_receiver_types: dict[str, str],
) -> None:
    for child in ast.walk(node):
        if not isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            continue

        for generator in child.generators:
            value_type = iteration_value_type_for_expr(
                generator.iter,
                local_annotations=local_annotations,
                param_annotations=param_annotations,
                module_annotations=module_annotations,
                class_member_annotations=class_member_annotations,
            )
            bind_target_type(generator.target, value_type, inferred_receiver_types)


def collect_context(
        node: CallableNode,
        class_name: str | None,
        imported_names: set[str],
        module_annotations: dict[str, str],
        class_member_annotations: dict[str, str],
        known_receiver_names: set[str],
) -> FunctionContext:
    param_annotations: dict[str, str | None] = {}
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg:
        args.append(node.args.vararg)
    if node.args.kwarg:
        args.append(node.args.kwarg)

    for arg in args:
        param_annotations[arg.arg] = annotation_to_string(arg.annotation)

    local_annotations: dict[str, str] = {}
    local_assignments: dict[str, ast.expr] = {}
    empty_container_assignments: set[str] = set()
    loop_sources: dict[str, ast.expr] = {}
    inferred_receiver_types: dict[str, str] = {}
    calls_by_receiver: dict[str, list[ast.Call]] = {}

    for child in ast.walk(node):
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            annotation = annotation_to_string(child.annotation)
            if annotation:
                local_annotations[child.target.id] = annotation
            if child.value is not None:
                local_assignments[child.target.id] = child.value
            continue

        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    local_assignments[target.id] = child.value
                    if is_empty_container_expr(child.value):
                        empty_container_assignments.add(target.id)
            continue

        if isinstance(child, ast.For):
            value_type = iteration_value_type_for_expr(
                child.iter,
                local_annotations=local_annotations,
                param_annotations=param_annotations,
                module_annotations=module_annotations,
                class_member_annotations=class_member_annotations,
            )
            bind_target_type(child.target, value_type, inferred_receiver_types)

            if isinstance(child.target, ast.Name):
                loop_sources[child.target.id] = child.iter
            elif isinstance(child.target, (ast.Tuple, ast.List)):
                for elt in child.target.elts:
                    if isinstance(elt, ast.Name):
                        loop_sources[elt.id] = child.iter
            continue

        if isinstance(child, ast.Call):
            receiver, _method = receiver_name(child)
            if receiver and re.fullmatch(r"(?:self\.)?[A-Za-z_]\w*", receiver):
                calls_by_receiver.setdefault(receiver, []).append(child)

    bind_comprehension_target_types(
        node,
        local_annotations=local_annotations,
        param_annotations=param_annotations,
        module_annotations=module_annotations,
        class_member_annotations=class_member_annotations,
        inferred_receiver_types=inferred_receiver_types,
    )

    qualified_name = f"{class_name}.{node.name}" if class_name else node.name
    return FunctionContext(
        node=node,
        class_name=class_name,
        qualified_name=qualified_name,
        param_annotations=param_annotations,
        return_annotation=annotation_to_string(node.returns),
        local_annotations=local_annotations,
        local_assignments=local_assignments,
        empty_container_assignments=empty_container_assignments,
        loop_sources=loop_sources,
        imported_names=imported_names,
        module_annotations=module_annotations,
        class_member_annotations=class_member_annotations,
        known_receiver_names=known_receiver_names,
        inferred_receiver_types=inferred_receiver_types,
        calls_by_receiver=calls_by_receiver,
    )


class SourceReadinessScanner:
    def __init__(self, source_root: Path, config: Config):
        self.source_root = source_root.resolve()
        self.config = config
        self.findings: list[Finding] = []
        self.function_return_annotations: dict[str, str | None] = {}
        self.project_fqns: set[str] = set()

    def scan(self, grouping: str) -> ReadinessReport:
        python_files = list(iter_python_files(self.source_root, self.config.exclude_dirs))
        self.project_fqns = {derive_fqn(path, self.source_root) for path in python_files}
        parsed_modules: list[tuple[Path, ast.Module, list[str], Any, dict[str, dict[str, dict[str, object]]]]] = []

        for path in python_files:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                self.add_finding(
                    finding_type="syntax_error_blocks_analysis",
                    severity="blocking",
                    file=path,
                    line=exc.lineno or 1,
                    message="Python syntax error prevents AST analysis for this file.",
                    suggestion="Fix the syntax error before running PyBastion on this source tree.",
                    confidence="high",
                )
                continue
            except OSError as exc:
                self.add_finding(
                    finding_type="source_file_unreadable",
                    severity="blocking",
                    file=path,
                    line=1,
                    message=f"Could not read source file: {exc}",
                    suggestion="Check file permissions and encoding.",
                    confidence="high",
                )
                continue

            stage1_result = process_file(
                path,
                self.source_root,
                project_symbol_fqns=self.project_fqns,
            )
            if stage1_result is None:
                self.add_finding(
                    finding_type="stage1_unit_index_failed",
                    severity="blocking",
                    file=path,
                    line=1,
                    message="Stage 1 unit indexing failed for this file.",
                    suggestion="Fix the source or Stage 1 inspection failure before relying on readiness results for this unit.",
                    confidence="high",
                )
                continue

            unit_index, field_registry = stage1_result
            parsed_modules.append((path, tree, source.splitlines(), unit_index, field_registry))
            self.index_function_definitions(tree)

        for path, tree, lines, unit_index, field_registry in parsed_modules:
            self.scan_module(path, tree, lines, unit_index, field_registry)

        findings = self.apply_limits(self.findings)
        severity_counts = count_by(
            findings,
            lambda finding: finding.severity,
            fixed_keys=("blocking", "high_impact", "advisory", "info"),
        )
        finding_type_counts = count_by(findings, lambda finding: finding.finding_type)

        return ReadinessReport(
            source_root=str(self.source_root),
            files_scanned=len(list(self.source_root.rglob("*"))) if self.source_root.exists() else 0,
            python_files_scanned=len(python_files),
            finding_count=len(findings),
            severity_counts=severity_counts,
            finding_type_counts=finding_type_counts,
            grouping=grouping,
            findings=findings,
        )

    def index_function_definitions(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function_return_annotations[node.name] = annotation_to_string(node.returns)

    def scan_module(
            self,
            path: Path,
            tree: ast.Module,
            lines: list[str],
            unit_index: Any,
            field_registry: dict[str, dict[str, dict[str, object]]],
    ) -> None:
        module_fqn = derive_fqn(path, self.source_root)
        module_annotations = module_annotations_from_unit_index(unit_index)
        known_receiver_names = known_names_from_unit_index(unit_index)

        try:
            unit_types = inspect_unit_types(
                unit_fqn=module_fqn,
                source_path=path,
                tree=tree,
                project_fqns=self.project_fqns,
            )
            imported_names = imported_names_from_unit_types(unit_types)
        except Exception as exc:
            imported_names = set()
            self.add_finding(
                finding_type="unit_type_inspection_failed",
                severity="blocking",
                file=path,
                line=1,
                message=f"Stage 1 type inspection failed for this unit: {exc}",
                suggestion="Fix the inspection failure before relying on readiness results for this unit.",
                confidence="high",
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_fqn = f"{module_fqn}.{node.name}"
                class_member_annotations = {
                    **field_annotations_from_registry(field_registry, class_fqn),
                    **property_return_annotations_from_class(node),
                }

                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.scan_callable(
                            path,
                            lines,
                            collect_context(
                                child,
                                node.name,
                                imported_names,
                                module_annotations,
                                class_member_annotations,
                                known_receiver_names,
                            ),
                        )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self.find_parent_class_name(tree, node) is not None:
                    continue
                self.scan_callable(
                    path,
                    lines,
                    collect_context(
                        node,
                        None,
                        imported_names,
                        module_annotations,
                        {},
                        known_receiver_names,
                    ),
                )

    def find_parent_class_name(self, tree: ast.Module, target: CallableNode) -> str | None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and target in node.body:
                return node.name
        return None

    def scan_callable(self, path: Path, lines: list[str], ctx: FunctionContext) -> None:
        self.check_unannotated_receivers(path, ctx)
        self.check_loop_receivers(path, ctx)
        self.check_empty_container_receivers(path, ctx)
        self.check_factory_return_receivers(path, ctx)
        self.check_any_receivers(path, ctx)
        self.check_self_attribute_receivers(path, ctx)
        self.check_dynamic_dispatch(path, ctx)
        self.check_marker_opportunities(path, lines, ctx)
        self.check_opaque_conditions(path, ctx)

    def has_receiver_type_evidence(self, receiver: str, ctx: FunctionContext) -> bool:
        if is_internal_or_builtin_receiver(receiver):
            return True
        if receiver in ctx.imported_names:
            return True
        if receiver in ctx.module_annotations:
            return True
        if receiver in ctx.known_receiver_names:
            return True
        if receiver in ctx.local_annotations:
            return True
        if receiver in ctx.inferred_receiver_types:
            return True
        if receiver in ctx.param_annotations and ctx.param_annotations[receiver]:
            return True
        if receiver in ctx.local_assignments:
            return True
        return False

    def check_unannotated_receivers(self, path: Path, ctx: FunctionContext) -> None:
        for receiver, calls in ctx.calls_by_receiver.items():
            if receiver.startswith("self."):
                continue
            if receiver in ctx.loop_sources:
                continue
            if self.has_receiver_type_evidence(receiver, ctx):
                continue

            call = calls[0]
            self.add_finding(
                finding_type="unresolved_call_receiver_type",
                severity="high_impact",
                file=path,
                line=call.lineno,
                callable_name=ctx.qualified_name,
                symbol=receiver,
                observed=safe_unparse(call),
                message=f"Receiver '{receiver}' is used for a method call but has no visible type source.",
                suggestion=f"Add a parameter, local, or assignment-source annotation for '{receiver}'.",
                example=f"{receiver}: SomeProtocolOrClass",
                confidence="medium",
            )

    def check_loop_receivers(self, path: Path, ctx: FunctionContext) -> None:
        for receiver, iter_expr in ctx.loop_sources.items():
            if receiver not in ctx.calls_by_receiver:
                continue
            if receiver in ctx.local_annotations or receiver in ctx.inferred_receiver_types:
                continue

            iterable_name = simple_name(iter_expr)
            iterable_annotation = None
            if iterable_name:
                iterable_annotation = (
                        ctx.local_annotations.get(iterable_name)
                        or ctx.param_annotations.get(iterable_name)
                        or ctx.module_annotations.get(iterable_name)
                )

            if iterable_element_type(iterable_annotation):
                continue

            call = ctx.calls_by_receiver[receiver][0]
            iter_text = safe_unparse(iter_expr) or "<unknown>"
            if iterable_name:
                suggestion = f"Annotate '{iterable_name}' with an element type so '{receiver}' can be inferred."
                example = f"{iterable_name}: Sequence[SomeProtocolOrClass]"
            else:
                suggestion = f"Add a local annotation for loop variable '{receiver}' before it is used as a receiver."
                example = f"{receiver}: SomeProtocolOrClass"

            self.add_finding(
                finding_type="unannotated_loop_receiver",
                severity="high_impact",
                file=path,
                line=call.lineno,
                callable_name=ctx.qualified_name,
                symbol=receiver,
                observed=f"for {receiver} in {iter_text}: ... {safe_unparse(call)}",
                message=f"Loop variable '{receiver}' is used as a call receiver, but its element type is not evident.",
                suggestion=suggestion,
                example=example,
                confidence="high",
            )

    def check_empty_container_receivers(self, path: Path, ctx: FunctionContext) -> None:
        for name in ctx.empty_container_assignments:
            if name in ctx.local_annotations:
                continue
            for loop_var, iter_expr in ctx.loop_sources.items():
                if simple_name(iter_expr) != name or loop_var not in ctx.calls_by_receiver:
                    continue
                if loop_var in ctx.inferred_receiver_types:
                    continue
                call = ctx.calls_by_receiver[loop_var][0]
                self.add_finding(
                    finding_type="untyped_empty_container_iteration",
                    severity="advisory",
                    file=path,
                    line=call.lineno,
                    callable_name=ctx.qualified_name,
                    symbol=name,
                    observed=safe_unparse(call),
                    message=f"Container '{name}' is initialized empty and later iterated into receiver calls.",
                    suggestion=f"Annotate '{name}' with an element type.",
                    example=f"{name}: list[SomeProtocolOrClass] = []",
                    confidence="medium",
                )

    def check_factory_return_receivers(self, path: Path, ctx: FunctionContext) -> None:
        for receiver, assigned_expr in ctx.local_assignments.items():
            if receiver not in ctx.calls_by_receiver:
                continue
            if receiver in ctx.local_annotations or receiver in ctx.inferred_receiver_types:
                continue
            if not isinstance(assigned_expr, ast.Call):
                continue
            if not isinstance(assigned_expr.func, ast.Name):
                continue

            callee_name = assigned_expr.func.id
            if callee_name[:1].isupper():
                continue
            if self.function_return_annotations.get(callee_name):
                continue

            call = ctx.calls_by_receiver[receiver][0]
            self.add_finding(
                finding_type="missing_factory_return_annotation_for_receiver",
                severity="advisory",
                file=path,
                line=assigned_expr.lineno,
                callable_name=ctx.qualified_name,
                symbol=receiver,
                observed=f"{receiver} = {safe_unparse(assigned_expr)}; {safe_unparse(call)}",
                message=f"Variable '{receiver}' is assigned from '{callee_name}(...)' and later used as a receiver, but that function has no known return annotation.",
                suggestion=f"Add a return annotation to '{callee_name}'.",
                example=f"def {callee_name}(...) -> SomeProtocolOrClass: ...",
                confidence="medium",
            )

    def check_any_receivers(self, path: Path, ctx: FunctionContext) -> None:
        for receiver, calls in ctx.calls_by_receiver.items():
            annotation = (
                    ctx.local_annotations.get(receiver)
                    or ctx.param_annotations.get(receiver)
                    or ctx.module_annotations.get(receiver)
                    or ctx.inferred_receiver_types.get(receiver)
            )
            if not is_any_annotation(annotation):
                continue
            call = calls[0]
            self.add_finding(
                finding_type="broad_any_receiver_annotation",
                severity="advisory",
                file=path,
                line=call.lineno,
                callable_name=ctx.qualified_name,
                symbol=receiver,
                observed=safe_unparse(call),
                message=f"Receiver '{receiver}' is annotated as Any and used for method calls.",
                suggestion="Use a Protocol, ABC, or concrete type when the receiver participates in integration behavior.",
                example=f"{receiver}: SomeProtocol",
                confidence="high",
            )

    def check_self_attribute_receivers(self, path: Path, ctx: FunctionContext) -> None:
        if ctx.class_name is None:
            return
        for receiver, calls in ctx.calls_by_receiver.items():
            if not receiver.startswith("self."):
                continue
            attr = receiver.split(".", 1)[1]
            if not attr or attr in ctx.class_member_annotations:
                continue
            call = calls[0]
            self.add_finding(
                finding_type="self_attribute_receiver_needs_type_evidence",
                severity="advisory",
                file=path,
                line=call.lineno,
                callable_name=ctx.qualified_name,
                symbol=receiver,
                observed=safe_unparse(call),
                message=f"Instance attribute receiver '{receiver}' is used for a method call.",
                suggestion=f"Ensure '{attr}' has a class attribute annotation, property return annotation, or is assigned from an annotated __init__ parameter.",
                example=f"{attr}: SomeProtocolOrClass",
                confidence="low",
            )

    @staticmethod
    def build_parent_map(node: ast.AST) -> dict[ast.AST, ast.AST]:
        return {
            child: parent
            for parent in ast.walk(node)
            for child in ast.iter_child_nodes(parent)
        }

    @staticmethod
    def call_result_is_invoked(
            call: ast.Call,
            parent_map: dict[ast.AST, ast.AST],
    ) -> bool:
        parent = parent_map.get(call)
        return isinstance(parent, ast.Call) and parent.func is call

    def check_dynamic_dispatch(self, path: Path, ctx: FunctionContext) -> None:
        parent_map = SourceReadinessScanner.build_parent_map(ctx.node)

        for child in ast.walk(ctx.node):
            if not isinstance(child, ast.Call):
                continue

            func_name = None
            if isinstance(child.func, ast.Name):
                func_name = child.func.id
            elif isinstance(child.func, ast.Attribute):
                func_name = child.func.attr

            if func_name == "getattr" and not SourceReadinessScanner.call_result_is_invoked(child, parent_map):
                continue

            if func_name in {"hasattr", "setattr"}:
                continue

            if func_name in DYNAMIC_DISPATCH_NAMES:
                self.add_finding(
                    finding_type="dynamic_dispatch_limits_static_resolution",
                    severity="info",
                    file=path,
                    line=child.lineno,
                    callable_name=ctx.qualified_name,
                    observed=safe_unparse(child),
                    message=f"Dynamic dispatch via '{func_name}' may not be statically resolvable.",
                    suggestion="Use an explicit typed dispatch table or a PyBastion marker if this dispatch is important for seam analysis.",
                    example="handlers: Mapping[str, Callable[[Request], Response]] = {...}",
                    confidence="high",
                )

    def check_marker_opportunities(self, path: Path, lines: list[str], ctx: FunctionContext) -> None:
        names = decorator_names(ctx.node)
        if names & {"MechanicalOperation", "UtilityOperation"}:
            return
        if has_marker_comment(lines, ctx.node, {"MechanicalOperation", "UtilityOperation"}):
            return

        lname = ctx.node.name.lower()
        if any(hint in lname for hint in MECHANICAL_NAME_HINTS):
            self.add_finding(
                finding_type="possible_mechanical_operation_marker",
                severity="info",
                file=path,
                line=ctx.node.lineno,
                callable_name=ctx.qualified_name,
                symbol=ctx.node.name,
                message=f"Callable '{ctx.qualified_name}' looks like serialization/deserialization or mechanical conversion.",
                suggestion="Consider marking it as MechanicalOperation if it should not be treated as an integration seam.",
                example="# :: MechanicalOperation | type=deserialization",
                confidence="low",
            )
            return

        if any(hint in lname for hint in UTILITY_NAME_HINTS):
            self.add_finding(
                finding_type="possible_utility_operation_marker",
                severity="info",
                file=path,
                line=ctx.node.lineno,
                callable_name=ctx.qualified_name,
                symbol=ctx.node.name,
                message=f"Callable '{ctx.qualified_name}' looks like a utility transformation or normalization helper.",
                suggestion="Consider marking it as UtilityOperation if it should not become an integration seam.",
                example="# :: UtilityOperation | type=normalization",
                confidence="low",
            )

    def check_opaque_conditions(self, path: Path, ctx: FunctionContext) -> None:
        for child in ast.walk(ctx.node):
            if not isinstance(child, ast.If):
                continue
            opaque_call = self.opaque_condition_call(child.test, ctx)
            if opaque_call is None:
                continue
            self.add_finding(
                finding_type="opaque_branch_condition_for_path_modeling",
                severity="info",
                file=path,
                line=child.lineno,
                callable_name=ctx.qualified_name,
                observed=safe_unparse(child.test),
                message="Branch condition contains a call that may require controllable fixture or target behavior to exercise.",
                suggestion="No source change is required. If this branch matters for integration tests, make the receiver type explicit and keep the behavior controllable.",
                confidence="low",
            )

    def opaque_condition_call(self, expr: ast.expr, ctx: FunctionContext) -> ast.Call | None:
        for node in ast.walk(expr):
            if not isinstance(node, ast.Call):
                continue
            if self.is_modelable_condition_call(node, ctx):
                continue
            return node
        return None

    def is_modelable_condition_call(self, node: ast.Call, ctx: FunctionContext) -> bool:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in MODELABLE_PREDICATE_FUNCTIONS:
                return True
            if name[:1].isupper():
                return True
            if name in self.function_return_annotations:
                return True
            if name in ctx.known_receiver_names:
                return True
            if name in ctx.imported_names:
                return True
            return False

        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            receiver = safe_unparse(node.func.value) or ""
            root = receiver.split(".", 1)[0]

            if method in MODELABLE_PREDICATE_METHODS:
                return True
            if receiver == "self" and method.startswith("_"):
                return True
            if root in ctx.imported_names:
                return True

        return False

    def add_finding(
            self,
            *,
            finding_type: str,
            severity: Severity,
            file: Path,
            line: int,
            message: str,
            suggestion: str | None = None,
            example: str | None = None,
            callable_name: str | None = None,
            symbol: str | None = None,
            observed: str | None = None,
            confidence: str = "medium",
    ) -> None:
        if severity == "info" and not self.config.include_info:
            return
        self.findings.append(
            Finding(
                finding_type=finding_type,
                severity=severity,
                file=self.rel(file),
                line=line,
                message=message,
                suggestion=suggestion,
                example=example,
                callable_name=callable_name,
                symbol=symbol,
                observed=observed,
                confidence=confidence,
            )
        )

    def rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.source_root))
        except Exception:
            return str(path)

    def apply_limits(self, findings: list[Finding]) -> list[Finding]:
        findings = dedupe_findings(findings)
        limit = self.config.max_findings_per_type
        if not limit:
            return sorted(findings, key=lambda item: (item.file, item.line, item.finding_type))

        counts: dict[str, int] = {}
        limited: list[Finding] = []
        for finding in sorted(findings, key=lambda item: (item.file, item.line, item.finding_type)):
            current = counts.get(finding.finding_type, 0)
            if current >= limit:
                continue
            limited.append(finding)
            counts[finding.finding_type] = current + 1
        return limited


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[Any, ...]] = set()
    result: list[Finding] = []
    for finding in findings:
        key = (
            finding.finding_type,
            finding.file,
            finding.line,
            finding.callable_name,
            finding.symbol,
            finding.observed,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


def count_by(findings: list[Finding], key_fn, fixed_keys: tuple[str, ...] = ()) -> dict[str, int]:
    counts: dict[str, int] = {key: 0 for key in fixed_keys}
    for finding in findings:
        key = str(key_fn(finding))
        counts[key] = counts.get(key, 0) + 1
    if fixed_keys:
        return counts
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def finding_to_report_dict(finding: Finding) -> dict[str, Any]:
    payload = {
        "type": finding.finding_type,
        "severity": finding.severity,
        "file": finding.file,
        "line": finding.line,
        "message": finding.message,
        "confidence": finding.confidence,
    }
    if finding.callable_name:
        payload["callable"] = finding.callable_name
    if finding.symbol:
        payload["symbol"] = finding.symbol
    if finding.observed:
        payload["observed"] = finding.observed
    if finding.suggestion:
        payload["suggestion"] = finding.suggestion
    if finding.example:
        payload["example"] = finding.example
    return payload


def module_for_finding(finding: Finding) -> str:
    if not finding.file:
        return "<unknown>"
    path = Path(finding.file)
    if path.name == "__init__.py":
        parts = path.parent.parts
    else:
        parts = path.with_suffix("").parts
    return ".".join(parts) if parts else "<root>"


def grouping_value(finding: Finding, code: str) -> str:
    match code:
        case "s":
            return finding.severity
        case "m":
            return module_for_finding(finding)
        case "t":
            return finding.finding_type
        case _:
            raise ValueError(f"Unsupported grouping code: {code}")


def validate_grouping(value: str) -> str:
    allowed = {"m", "s", "t"}
    codes = list(value)
    if len(codes) != 3 or set(codes) != allowed:
        raise ValueError("--grouping must contain each of m, s, and t exactly once, such as smt or mst.")
    return value


def sorted_group_keys(keys: Iterable[str], code: str) -> list[str]:
    if code == "s":
        return sorted(keys, key=lambda item: (SEVERITY_ORDER.get(item, 99), item))
    return sorted(keys)


def grouped_findings_as_dict(findings: list[Finding], grouping: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for finding in findings:
        current = root
        for code in grouping:
            current = current.setdefault(grouping_value(finding, code), {})
        current.setdefault("_findings", []).append(finding_to_report_dict(finding))
    return sort_grouped_findings(root, grouping, 0)


def sort_grouped_findings(node: dict[str, Any], grouping: str, depth: int) -> dict[str, Any]:
    if "_findings" in node:
        findings = sorted(
            node["_findings"],
            key=lambda item: (str(item.get("file", "")), int(item.get("line", 0)), str(item.get("type", ""))),
        )
        return {"findings": findings}

    code = grouping[depth]
    ordered: dict[str, Any] = {}
    for key in sorted_group_keys(node.keys(), code):
        ordered[key] = sort_grouped_findings(node[key], grouping, depth + 1)
    return ordered


def report_to_dict(report: ReadinessReport) -> dict[str, Any]:
    return {
        "source_root": report.source_root,
        "python_files_scanned": report.python_files_scanned,
        "finding_count": report.finding_count,
        "severity_counts": report.severity_counts,
        "finding_type_counts": report.finding_type_counts,
        "grouping": report.grouping,
        "findings": grouped_findings_as_dict(report.findings, report.grouping),
    }


def render_markdown_grouped(node: dict[str, Any], level: int = 2) -> list[str]:
    lines: list[str] = []
    if "findings" in node:
        for finding in node["findings"]:
            location = f"{finding.get('file')}:{finding.get('line')}"
            lines.append(f"- `{location}` `{finding.get('type')}`")
            if finding.get("callable"):
                lines.append(f"  - Callable: `{finding['callable']}`")
            if finding.get("symbol"):
                lines.append(f"  - Symbol: `{finding['symbol']}`")
            if finding.get("observed"):
                lines.append(f"  - Observed: `{finding['observed']}`")
            lines.append(f"  - {finding.get('message')}")
            if finding.get("suggestion"):
                lines.append(f"  - Suggestion: {finding['suggestion']}")
            if finding.get("example"):
                lines.append(f"  - Example: `{finding['example']}`")
        return lines

    for key, child in node.items():
        lines.append("")
        lines.append(f"{'#' * level} {key}")
        lines.extend(render_markdown_grouped(child, min(level + 1, 6)))
    return lines


def render_markdown(report: ReadinessReport) -> str:
    grouped = grouped_findings_as_dict(report.findings, report.grouping)
    lines = [
        "# PyBastion Readiness Findings",
        "",
        f"Source root: `{report.source_root}`",
        f"Python files scanned: **{report.python_files_scanned}**",
        f"Findings: **{report.finding_count}**",
        f"Grouping: `{report.grouping}`",
        "",
        "## Severity counts",
        "",
    ]
    for severity, count in report.severity_counts.items():
        if count:
            lines.append(f"- {severity}: {count}")

    lines.extend(["", "## Finding category counts", ""])
    for finding_type, count in report.finding_type_counts.items():
        lines.append(f"- {finding_type}: {count}")

    lines.extend(["", "## Findings"])
    if not report.findings:
        lines.append("No findings.")
    else:
        lines.extend(render_markdown_grouped(grouped))

    return "\n".join(lines)


def write_report(report: ReadinessReport, output: Path, fmt: str) -> None:
    if fmt == "json":
        text = json.dumps(report_to_dict(report), indent=2, sort_keys=False)
    elif fmt in {"yaml", "yml"}:
        text = yaml.safe_dump(report_to_dict(report), sort_keys=False, allow_unicode=True, width=120)
    elif fmt in {"markdown", "md"}:
        text = render_markdown(report)
    else:
        raise ValueError(f"Unsupported report format: {fmt}")

    output.write_text(text, encoding="utf-8")


def print_console_summary(report: ReadinessReport, output_path: Path) -> None:
    print()
    print("PyBastion readiness summary")
    print("=" * 32)
    print(f"Source root:  {report.source_root}")
    print(f"Python files: {report.python_files_scanned}")
    print(f"Findings:     {report.finding_count}")

    if report.severity_counts:
        print()
        print("By severity:")
        for severity in ("blocking", "high_impact", "advisory", "info"):
            count = report.severity_counts.get(severity, 0)
            if count:
                print(f"  {severity:12} {count}")

    if report.finding_type_counts:
        print()
        print("Top finding categories:")
        for finding_type, count in sorted(
                report.finding_type_counts.items(),
                key=lambda item: (-item[1], item[0]),
        )[:10]:
            print(f"  {finding_type:52} {count}")

    print()
    print(f"Report: {output_path}")
    print()


def resolve_output_path(
        output_arg: Path | None,
        project_root: Path,
        output_format: str,
) -> tuple[Path, str]:
    match output_format:
        case "json":
            default_filename = "readiness.json"
        case "markdown" | "md":
            default_filename = "readiness.md"
        case "yaml" | "yml":
            default_filename = "readiness.yaml"
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")

    if output_arg is None:
        output_dir = project_root / "dist" / "pybastion" / "inspect"
        output_filename = default_filename
    else:
        resolved = (
            output_arg
            if output_arg.is_absolute()
            else project_root / output_arg
        ).resolve()

        if resolved.exists() and resolved.is_dir():
            output_dir = resolved
            output_filename = default_filename
        elif not resolved.exists() and resolved.suffix == "":
            output_dir = resolved
            output_filename = default_filename
        else:
            output_dir = resolved.parent
            output_filename = resolved.name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve(), output_filename


def resolve_path(value: Path, project_root: Path) -> Path:
    if value.is_absolute():
        return value.resolve()
    return (project_root / value).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-only PyBastion analysis readiness scanner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=Path("."),
        help="Project root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Python source root to inspect. Defaults to <project-path>/src. "
            "Relative paths are resolved against --project-path."
        ),
    )
    parser.add_argument(
        "--grouping",
        metavar="CODES",
        default="smt",
        help=(
            "Grouping strategy for findings. Codes: "
            "s=severity, "
            "m=module, "
            "t=type. "
            "If omitted, the default is 'smt' (severity, module, type)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Readiness report output path. Defaults to "
            "<project-path>/dist/pybastion/inspect/readiness.yaml. "
            "If this is a directory, readiness.<format> is written inside it."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("yaml", "yml", "json", "markdown", "md"),
        default="yaml",
        help="Report output format.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional TOML config.",
    )
    parser.add_argument(
        "--max-findings-per-type",
        type=int,
        help="Limit noisy finding categories.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        grouping = validate_grouping(args.grouping)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    project_root = args.project_path.resolve()
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None and args.source_root.is_absolute()
        else (project_root / (args.source_root or "src")).resolve()
    )

    if not project_root.exists():
        print(f"ERROR: project root not found: {project_root}", file=sys.stderr)
        return 2
    if not source_root.exists() or not source_root.is_dir():
        print(f"ERROR: source root not found: {source_root}", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
        if args.max_findings_per_type:
            cfg.max_findings_per_type = args.max_findings_per_type

        output_dir, output_filename = resolve_output_path(
            args.output,
            project_root,
            args.format,
        )
        output_path = output_dir / output_filename

        scanner = SourceReadinessScanner(source_root, cfg)
        report = scanner.scan(grouping)
        write_report(report, output_path, args.format)
        print_console_summary(report, output_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
