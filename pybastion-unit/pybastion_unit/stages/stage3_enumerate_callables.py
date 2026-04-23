#!/usr/bin/env python3
"""Stage 3 callable inventory builder for structured stage 1 and stage 2 output.

This module consumes:
  - stage 1 structured unit index JSON
  - stage 2 EI YAML for a unit
  - source files referenced by the unit index
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pybastion_common.models import (
    Branch,
    CallableAnalysisInfo,
    CallableEntry,
    CallableHierarchyInfo,
    CallableSignatureInfo,
    ParamSpec,
    ProjectIndex,
    TypeRef,
    UnitIndex,
    UnitIndexEntry,
)
from pybastion_common.smt_path_checker import filter_feasible_paths
from pybastion_unit.helpers.chain_resolution import (
    ResolutionContext,
    resolve_target_chain,
)
from pybastion_unit.helpers.decorator_processing import (
    extract_callable_decorators,
    has_effect,
    validate_feature_co_occurrences,
)
from pybastion_unit.helpers.integration_analysis import build_integration_entries, default_signature
from pybastion_unit.helpers.type_indexing import (
    build_module_index,
    inspect_unit_types, self_attribute_name,
)
from pybastion_unit.helpers.type_inference import (
    build_param_type_map_with_defaults,
    infer_literal_type,
)

ENUM_BASES: set[str] = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
CALLABLE_ENTRY_KINDS: set[str] = {
    "unit_function",
    "method",
    "nested_function",
    "module_assignment",
}
CLASS_ENTRY_KINDS: set[str] = {"class", "nested_class"}


@dataclass(frozen=True)
class TargetResolution:
    original_target: str
    resolved_target: str
    resolution_kind: str
    resolved_receiver_type: str | None = None
    resolution_basis: str | None = None
    candidate_targets: list[str] | None = None


# ============================================================================
# Stage 1 model
# ============================================================================


def load_class_field_registry(filepath: Path) -> dict[str, dict[str, TypeRef]]:
    payload = json.loads(filepath.read_text(encoding="utf-8"))
    raw_registry: dict[str, dict[str, dict[str, Any]]] = payload.get("class_field_registry", {}) or {}

    return {
        class_fqn: {
            field_name: TypeRef.from_dict(type_payload)
            for field_name, type_payload in field_map.items()
        }
        for class_fqn, field_map in raw_registry.items()
    }


def load_project_index(filepath: Path) -> ProjectIndex:
    payload = json.loads(filepath.read_text(encoding="utf-8"))
    units: list[UnitIndex] = []
    for unit_payload in payload.get("units", []):
        units.append(
            UnitIndex(
                unit_id=unit_payload["unit_id"],
                fully_qualified_name=unit_payload["fully_qualified_name"],
                filepath=unit_payload["filepath"],
                language=unit_payload["language"],
                source_hash=unit_payload["source_hash"],
                entries=[UnitIndexEntry(**entry) for entry in unit_payload.get("entries", [])],
            )
        )
    return ProjectIndex(source_root=payload["source_root"], units=units)


# ============================================================================
# Stage 2 model helpers
# ============================================================================


def load_stage2_yaml(filepath: Path | None) -> dict[str, Any]:
    if filepath is None or not filepath.exists():
        return {}
    payload = yaml.safe_load(filepath.read_text(encoding="utf-8"))
    return payload or {}


def build_stage2_lookup(stage2_payload: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in stage2_payload.get("functions", []) or []:
        key = (item.get("name", ""), item.get("line_start", 0), item.get("line_end", 0))
        lookup[key] = item
    return lookup


# ============================================================================
# AST analysis helpers
# ============================================================================


def collect_project_contract_classes(project_index: ProjectIndex) -> set[str]:
    contract_classes: set[str] = set()

    for unit in project_index.units:
        source_path = Path(unit.filepath)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        ast_index = build_module_index(tree, unit.fully_qualified_name)

        for entry in unit.entries:
            if entry.kind not in CLASS_ENTRY_KINDS:
                continue

            node = ast_index.nodes_by_fqn_and_line.get((entry.fully_qualified_name, entry.lineno))
            if not isinstance(node, ast.ClassDef):
                continue

            class_is_contract = is_abstract_base_class(node)

            if not class_is_contract:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and has_abstractmethod_decorator(child):
                        class_is_contract = True
                        break

            if class_is_contract:
                contract_classes.add(entry.fully_qualified_name)

    return contract_classes


def is_enum_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in ENUM_BASES:
            return True
        if isinstance(base, ast.Attribute) and base.attr in ENUM_BASES:
            return True
    return False


def has_abstractmethod_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "abstractmethod":
            return True
    return False


def _base_expr_lookup_name(base_expr: ast.expr) -> str:
    match base_expr:
        case ast.Subscript(value=value):
            return ast.unparse(value).strip()
        case _:
            return ast.unparse(base_expr).strip()


def _strip_generic_suffix(type_name: str) -> str:
    return type_name.split("[", 1)[0].strip()


def resolve_base_class_fqn(
        base_expr: ast.expr,
        unit_fqn: str,
        import_map: dict[str, str],
        callable_inventory: dict[str, str],
) -> str:
    raw = ast.unparse(base_expr).strip()
    lookup = _strip_generic_suffix(_base_expr_lookup_name(base_expr))

    if lookup in import_map:
        return _strip_generic_suffix(import_map[lookup])

    if "." not in lookup:
        same_unit = f"{unit_fqn}.{lookup}"
        if same_unit in callable_inventory:
            return same_unit

        matches = [fqn for fqn in callable_inventory if fqn.endswith(f".{lookup}")]
        if len(matches) == 1:
            return matches[0]

    return lookup if lookup else raw


def is_contract_base_name(base_fqn: str) -> bool:
    short = base_fqn.rsplit(".", 1)[-1]
    return short in {
        "ABC",
        "Protocol",
    }


def is_abstract_base_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "ABC":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "ABC":
            return True
    return False


def signature_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.setdefault("signature_info", {})


def hierarchy_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.setdefault("hierarchy_info", {})


def analysis_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.setdefault("analysis_info", {})


def mark_contract_methods(entries: list[dict[str, Any]], in_contract_class: bool = False) -> None:
    for entry in entries:
        hinfo = hierarchy_info(entry)

        entry_is_contract_class = (
                in_contract_class
                or bool(hinfo.get("is_abstract", False))
                or bool(hinfo.get("contract_base_classes", []))
        )

        if entry.get("kind") == "method" and entry_is_contract_class:
            hinfo["is_contract_method"] = True

        children = entry.get("children", []) or []
        if children:
            mark_contract_methods(children, entry_is_contract_class)


def is_non_executable_callable_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """
    Return True for declarative/contract callables that should not participate
    in executable flow graphs.

    Cases:
    - body is just `pass`
    - body is just `...`
    - body is just `raise NotImplementedError(...)`
    - same as above, but preceded by a docstring
    """
    body = list(node.body)

    if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) != 1:
        return False

    stmt = body[0]

    if isinstance(stmt, ast.Pass):
        return True

    if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
    ):
        return True

    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        exc = stmt.exc

        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True

        if isinstance(exc, ast.Call):
            func = exc.func
            if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                return True

    return False


def annotate_method_owners(
        entries: list[dict[str, Any]],
        unit_fqn: str,
        ancestors: list[str] | None = None,
) -> None:
    if ancestors is None:
        ancestors = []

    for entry in entries:
        kind = entry.get("kind")
        name = entry.get("name", "")

        current_ancestors = [*ancestors]
        if kind in {"class", "enum"}:
            current_ancestors.append(name)

        if kind == "method" and ancestors:
            hierarchy_info(entry)["owner_class_fqn"] = ".".join([unit_fqn, *ancestors])

        children = entry.get("children", []) or []
        if children:
            annotate_method_owners(children, unit_fqn, current_ancestors)


def annotate_entry_fqns(
        entries: list[dict[str, Any]],
        unit_fqn: str,
        ancestors: list[str] | None = None,
) -> None:
    if ancestors is None:
        ancestors = []

    for entry in entries:
        name = entry.get("name", "")
        kind = entry.get("kind")
        entry["_fqn"] = ".".join([unit_fqn, *ancestors, name])

        child_ancestors = [*ancestors]
        if kind in {"class", "enum", "method", "function", "assignment"}:
            child_ancestors.append(name)

        children = entry.get("children", []) or []
        if children:
            annotate_entry_fqns(children, unit_fqn, child_ancestors)


class AstAnalyzer:
    def __init__(
            self,
            *,
            source_lines: list[str],
            unit_id: str,
            unit_fqn: str,
            callable_inventory: dict[str, str],
            project_fqns: set[str],
            import_map: dict[str, str],
            local_return_types: dict[str, TypeRef],
            field_types_by_class: dict[str, dict[str, TypeRef]],
            project_contract_classes: set[str],
    ):
        self.source_lines = source_lines
        self.unit_id = unit_id
        self.unit_fqn = unit_fqn
        self.callable_inventory = callable_inventory
        self.project_fqns = project_fqns
        self.import_map = import_map
        self.local_return_types = local_return_types
        self.field_types_by_class = field_types_by_class
        self.project_contract_classes = project_contract_classes

    def _normalize_type_name(self, type_name: str | TypeRef) -> str:
        if isinstance(type_name, TypeRef):
            return type_name.name
        return TypeRef.from_annotation_string(type_name).name

    def analyze_entry(self, entry: UnitIndexEntry, node: ast.AST) -> dict[str, Any]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._analyze_callable(entry, node)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            return self._analyze_assignment(entry, node)
        if isinstance(node, ast.ClassDef):
            return self._analyze_class(entry, node)
        return {
            "id": entry.id,
            "kind": entry.kind,
            "name": entry.name,
            "line_start": entry.lineno,
            "line_end": entry.end_lineno,
        }

    def _analyze_class(self, entry: UnitIndexEntry, node: ast.ClassDef) -> dict[str, Any]:
        kind = "enum" if is_enum_class(node) else "class"

        raw_bases = [ast.unparse(base) for base in node.bases]
        resolved_bases = [
            resolve_base_class_fqn(
                base,
                self.unit_fqn,
                self.import_map,
                self.callable_inventory,
            )
            for base in node.bases
        ]
        contract_bases = [
            base for base in resolved_bases
            if is_contract_base_name(base) or base in self.project_contract_classes
        ]

        return CallableEntry(
            id=entry.id,
            kind=kind,
            name=entry.name,
            line_start=entry.lineno,
            line_end=entry.end_lineno,
            signature_info=CallableSignatureInfo(
                decorators=self._extract_decorators(node.decorator_list),
            ),
            hierarchy_info=CallableHierarchyInfo(
                base_classes=raw_bases,
                resolved_base_classes=resolved_bases,
                contract_base_classes=contract_bases,
                is_abstract=is_abstract_base_class(node),
            ),
        ).to_dict()

    def _analyze_callable(self, entry: UnitIndexEntry, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
        decorators = self._extract_decorators(node.decorator_list) + extract_callable_decorators(
            node,
            self.source_lines,
        )
        param_types = self._build_param_type_map(node)
        local_types = self.build_local_type_map(node)
        known_types = {**param_types, **local_types}
        kind = "method" if entry.kind == "method" else "function"

        payload = CallableEntry(
            id=entry.id,
            kind=kind,
            name=entry.name,
            line_start=entry.lineno,
            line_end=entry.end_lineno,
            signature_info=CallableSignatureInfo(
                signature=self._build_signature(node),
                visibility=self._extract_visibility(entry.name),
                decorators=decorators,
                modifiers=self._extract_modifiers(node),
                params=self._extract_params(node),
                return_type=self._extract_type_ref(node.returns),
            ),
            hierarchy_info=CallableHierarchyInfo(),
            analysis_info=CallableAnalysisInfo(
                integration_candidates=[],
                needs_callable_analysis=True,
            ),
        ).to_dict()

        if is_non_executable_callable_body(node):
            payload["is_executable"] = False
            hierarchy_info(payload)["is_contract_method"] = True

        if has_abstractmethod_decorator(node):
            hierarchy_info(payload)["is_contract_method"] = True

        payload["_known_types"] = known_types
        return payload

    def _analyze_assignment(
            self,
            entry: UnitIndexEntry,
            node: ast.Assign | ast.AnnAssign | ast.AugAssign,
    ) -> dict[str, Any]:
        local_types = self.build_local_type_map(node)

        payload = CallableEntry(
            id=entry.id,
            kind="assignment",
            name=entry.name,
            line_start=entry.lineno,
            line_end=entry.end_lineno,
            signature_info=CallableSignatureInfo(
                visibility=self._extract_visibility(entry.name),
            ),
            hierarchy_info=CallableHierarchyInfo(),
            analysis_info=CallableAnalysisInfo(
                integration_candidates=[],
                needs_callable_analysis=True,
            ),
        ).to_dict()

        payload["_known_types"] = local_types
        return payload

    def build_callable_local_type_map(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, TypeRef]:
        local_types: dict[str, TypeRef] = {}

        owner_class_fqn = (
            self._owner_class_fqn_for_callable(self.current_callable_fqn)
            if hasattr(self, "current_callable_fqn")
            else None
        )
        owner_fields = self.field_types_by_class.get(owner_class_fqn or "", {})

        def infer_expr_type(expr: ast.AST) -> TypeRef | None:
            if isinstance(expr, ast.Name):
                if expr.id == "self":
                    if owner_class_fqn:
                        return TypeRef(name=owner_class_fqn)
                    return None
                return local_types.get(expr.id)

            literal_type = infer_literal_type(expr)
            if literal_type is not None:
                return literal_type

            if isinstance(expr, ast.Attribute):
                attr_name = self_attribute_name(expr)
                if attr_name and attr_name in owner_fields:
                    return owner_fields[attr_name]

            if isinstance(expr, ast.Call):
                func = expr.func
                if isinstance(func, ast.Name) and func.id in self.local_return_types:
                    return self.local_return_types[func.id]
                if isinstance(func, ast.Attribute) and func.attr in self.local_return_types:
                    return self.local_return_types[func.attr]

            return None

        def unpack_tuple_like_type(type_ref: TypeRef) -> list[TypeRef]:
            if type_ref.name in {"tuple", "Tuple"}:
                return list(type_ref.args)

            if type_ref.name in {"list", "List", "set", "Set", "Sequence", "Iterable"} and len(type_ref.args) == 1:
                inner = type_ref.args[0]
                if inner.name in {"tuple", "Tuple"}:
                    return list(inner.args)

            return []

        def infer_iter_value_type(expr: ast.AST) -> TypeRef | None:
            iter_type = infer_expr_type(expr)
            if iter_type is None:
                return None

            if iter_type.name in {"list", "List", "set", "Set", "Iterable", "Iterator", "Sequence"} and iter_type.args:
                return iter_type.args[0]

            if iter_type.name in {"dict", "Dict", "Mapping"} and len(iter_type.args) >= 2:
                return iter_type.args[0]

            return None

        def infer_items_pair_type(expr: ast.AST) -> TypeRef | None:
            if not isinstance(expr, ast.Call):
                return None

            func = expr.func
            if not isinstance(func, ast.Attribute):
                return None
            if func.attr != "items":
                return None

            receiver_type = infer_expr_type(func.value)
            if receiver_type is None:
                return None

            if receiver_type.name in {"dict", "Dict", "Mapping"} and len(receiver_type.args) >= 2:
                return TypeRef(
                    name="tuple",
                    args=[receiver_type.args[0], receiver_type.args[1]],
                )

            return None

        def bind_target_types(target: ast.AST, value_type: TypeRef | None) -> None:
            if value_type is None:
                return

            if isinstance(target, ast.Name):
                local_types[target.id] = value_type
                return

            if isinstance(target, (ast.Tuple, ast.List)):
                item_types = unpack_tuple_like_type(value_type)
                if not item_types:
                    return

                for elt, elt_type in zip(target.elts, item_types):
                    if isinstance(elt, ast.Name):
                        local_types[elt.id] = elt_type

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                type_ref = TypeRef.from_annotation_ast(stmt.annotation)
                if type_ref is not None:
                    local_types[stmt.target.id] = type_ref
                continue

            if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
            ):
                target_name = stmt.targets[0].id
                inferred = infer_expr_type(stmt.value)
                if inferred is not None:
                    local_types[target_name] = inferred
                continue

            if isinstance(stmt, ast.For):
                pair_type = infer_items_pair_type(stmt.iter)
                if pair_type is not None:
                    bind_target_types(stmt.target, pair_type)
                    continue

                iter_value_type = infer_iter_value_type(stmt.iter)
                if iter_value_type is not None:
                    bind_target_types(stmt.target, iter_value_type)
                    continue

        return local_types

    def build_assignment_local_type_map(
            self,
            node: ast.Assign | ast.AnnAssign | ast.AugAssign,
    ) -> dict[str, TypeRef]:
        local_types: dict[str, TypeRef] = {}

        match node:
            case ast.AnnAssign(target=ast.Name() as target, annotation=annotation):
                type_ref = TypeRef.from_annotation_ast(annotation)
                if type_ref is not None:
                    local_types[target.id] = type_ref

            case ast.Assign(targets=[ast.Name() as target], value=value):
                if isinstance(value, ast.Call):
                    func = value.func

                    if isinstance(func, ast.Name):
                        inferred = self.local_return_types.get(func.id)
                        if inferred is not None:
                            local_types[target.id] = inferred

                    elif isinstance(func, ast.Attribute):
                        inferred = self.local_return_types.get(func.attr)
                        if inferred is not None:
                            local_types[target.id] = inferred

            case ast.AugAssign(target=ast.Name()):
                pass

        return {k: v for k, v in local_types.items() if v is not None}

    def build_local_type_map(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Assign | ast.AnnAssign | ast.AugAssign,
    ) -> dict[str, TypeRef]:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                return self.build_callable_local_type_map(node)

            case ast.Assign() | ast.AnnAssign() | ast.AugAssign():
                return self.build_assignment_local_type_map(node)

            case _:
                return {}

    def _build_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        try:
            start_line = node.lineno - 1
            sig_line = self.source_lines[start_line].strip()
            if not sig_line.endswith(":"):
                for i in range(start_line + 1, min(start_line + 20, len(self.source_lines))):
                    sig_line += " " + self.source_lines[i].strip()
                    if self.source_lines[i].strip().endswith(":"):
                        break
            sig_line = sig_line.replace("async def ", "").replace("def ", "")
            if sig_line.endswith(":"):
                sig_line = sig_line[:-1].strip()
            sig_line = re.sub(r"\s+", " ", sig_line)
            return sig_line.replace("( ", "(").replace(" )", ")").replace(" ,", ",")
        except Exception:
            return f"{node.name}(...)"

    @staticmethod
    def _extract_decorators(decorator_list: list[ast.expr]) -> list[dict[str, Any]]:
        decorators: list[dict[str, Any]] = []
        for dec in decorator_list:
            info: dict[str, Any] = {}
            if isinstance(dec, ast.Name):
                info["name"] = dec.id
            elif isinstance(dec, ast.Call):
                info["name"] = ast.unparse(dec.func)
                if dec.args:
                    info["args"] = [ast.unparse(arg) for arg in dec.args]
                if dec.keywords:
                    info["kwargs"] = {kw.arg: ast.unparse(kw.value) for kw in dec.keywords if kw.arg}
            else:
                info["name"] = ast.unparse(dec)
            decorators.append(info)
        return decorators

    @staticmethod
    def _extract_modifiers(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        return ["async"] if isinstance(node, ast.AsyncFunctionDef) else []

    @staticmethod
    def _extract_visibility(name: str) -> str:
        if name.startswith("__") and not name.endswith("__"):
            return "private"
        if name.startswith("_"):
            return "protected"
        return "public"

    def _extract_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ParamSpec]:
        params: list[ParamSpec] = []
        for arg in node.args.args:
            params.append(ParamSpec(name=arg.arg, type=self._extract_type_ref(arg.annotation), default=None))
        defaults = node.args.defaults
        if defaults:
            start_idx = len(params) - len(defaults)
            for i, default in enumerate(defaults):
                params[start_idx + i].default = ast.unparse(default)
        return params

    def _build_param_type_map(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, TypeRef]:
        return build_param_type_map_with_defaults(node)

    def _extract_type_ref(self, annotation: ast.expr | None) -> TypeRef | None:
        return TypeRef.from_annotation_ast(annotation)

    def find_integration_candidates(
            self,
            *,
            branches: list[Branch],
            callable_id: str,
            callable_fqn: str,
            known_types: dict[str, TypeRef],
    ) -> list[dict[str, Any]]:
        integrations = build_integration_entries(
            branches=branches,
            callable_id=callable_id,
            callable_fqn=callable_fqn,
            unit_fqn=self.unit_fqn,
            project_fqns=self.project_fqns,
            callable_inventory=self.callable_inventory,
            known_types=known_types,
            resolve_target=lambda target, param_types: self._resolve_target(
                target,
                param_types,
                callable_fqn=callable_fqn,
            ),
            signature_for_branch=default_signature,
        )

        enriched: list[dict[str, Any]] = []
        for integration in integrations:
            payload = integration.to_dict()

            classification = payload.get("classification", {}) or {}
            resolved_target = classification.get("resolved_target") or payload.get("target")

            payload["resolved_target"] = resolved_target
            payload["resolution_kind"] = payload.get("kind", "unknown")

            enriched.append(payload)

        return enriched

    def _owner_class_fqn_for_callable(self, callable_fqn: str) -> str | None:
        parts = callable_fqn.split(".")
        if len(parts) < 2:
            return None
        return ".".join(parts[:-1]) if parts[-2][:1].isupper() else None

    def _resolve_target_details(
            self,
            target: str,
            param_types: dict[str, TypeRef],
            callable_fqn: str | None = None,
    ) -> TargetResolution:
        owner_class_fqn = None
        if callable_fqn:
            owner_class_fqn = self._owner_class_fqn_for_callable(callable_fqn)

        ctx = ResolutionContext(
            unit_fqn=self.unit_fqn,
            import_map=self.import_map,
            callable_inventory=self.callable_inventory,
            local_return_types=self.local_return_types,
            field_types_by_class=self.field_types_by_class,
            callable_fqn=callable_fqn or self.unit_fqn,
            owner_class_fqn=owner_class_fqn,
            known_types=param_types,
        )

        result = resolve_target_chain(ctx, target)
        return TargetResolution(
            original_target=result.original_target,
            resolved_target=result.resolved_target,
            resolution_kind=result.resolution_kind,
            resolved_receiver_type=result.resolved_receiver_type,
            resolution_basis=result.resolution_basis,
            candidate_targets=result.candidate_targets,
        )

    def _resolve_target(
            self,
            target: str,
            param_types: dict[str, TypeRef],
            callable_fqn: str | None = None,
    ) -> tuple[str, str]:
        resolution = self._resolve_target_details(
            target,
            param_types,
            callable_fqn=callable_fqn,
        )
        return resolution.resolved_target, resolution.resolved_receiver_type or ""

    @staticmethod
    def _get_call_target(call_node: ast.Call) -> str | None:
        try:
            return ast.unparse(call_node.func)
        except Exception:
            return None


# ============================================================================
# CFG / path analysis
# ============================================================================


def iter_successor_outcomes(branch: Branch):
    if branch.statement_outcome is not None:
        yield branch.statement_outcome
    for target in branch.conditional_targets or []:
        yield target
    for outcome in branch.disruptive_outcomes or []:
        yield outcome


def build_cfg(branches: list[Branch]) -> dict[str, list[str]]:
    by_id = {branch.id: branch for branch in branches}
    graph: dict[str, list[str]] = {}

    for branch in branches:
        successors: list[str] = []
        for outcome in iter_successor_outcomes(branch):
            if outcome.is_terminal:
                continue
            if outcome.target_ei:
                successors.append(outcome.target_ei)

        seen: set[str] = set()
        graph[branch.id] = [ei for ei in successors if ei in by_id and not (ei in seen or seen.add(ei))]

    return graph


def enumerate_paths(graph: dict[str, list[str]], start_ei: str, target_ei: str) -> list[list[str]]:
    def dfs(current: str, target: str, path: list[str], visited: set[str]) -> list[list[str]]:
        if current == target:
            return [path + [current]]
        if current in visited:
            return []
        visited_copy = visited | {current}
        all_paths: list[list[str]] = []
        for next_ei in graph.get(current, []):
            all_paths.extend(dfs(next_ei, target, path + [current], visited_copy))
        return all_paths

    return [[target_ei]] if start_ei == target_ei else dfs(start_ei, target_ei, [], set())


def add_execution_paths(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        sinfo = signature_info(entry)
        ainfo = analysis_info(entry)

        if sinfo.get("decorators"):
            for decorator in sinfo["decorators"]:
                if has_effect(decorator, "exclude_from_flow"):
                    for integration in ainfo.get("integration_candidates", []):
                        integration["execution_paths"] = []
                        integration["suppressed_by"] = decorator.get("name")
                    break

        if (
                ainfo.get("needs_callable_analysis")
                and ainfo.get("branches")
                and ainfo.get("integration_candidates")
        ):
            branches = [Branch.from_dict(item) for item in ainfo["branches"]]
            graph = build_cfg(branches)

            predecessors: dict[str, list[str]] = {branch.id: [] for branch in branches}
            for src, targets in graph.items():
                for target in targets:
                    predecessors.setdefault(target, []).append(src)

            branch_by_id = {branch.id: branch for branch in branches}

            explicit_entry_eis = [
                branch.id
                for branch in branches
                if (
                        branch.statement_outcome is not None
                        and branch.statement_outcome.synthetic
                        and branch.stmt_type == "FunctionInvocation"
                        and branch.description == "function start"
                )
            ]

            if explicit_entry_eis:
                entry_eis = explicit_entry_eis
            else:
                entry_eis = [
                    ei_id
                    for ei_id, preds in predecessors.items()
                    if not preds
                       and not (
                            branch_by_id[ei_id].owner_info is not None
                            and branch_by_id[ei_id].owner_info.stmt_type == "Try"
                            and branch_by_id[ei_id].owner_info.region == "except"
                    )
                ]

                if not entry_eis and branches:
                    first_line = min(branch.line for branch in branches)
                    entry_eis = [branch.id for branch in branches if branch.line == first_line]

            for integration in ainfo["integration_candidates"]:
                target_ei = integration.get("ei_id")
                if not target_ei:
                    integration["execution_paths"] = []
                    continue

                all_paths: list[list[str]] = []
                for start_ei in entry_eis:
                    all_paths.extend(enumerate_paths(graph, start_ei, target_ei))

                unique_paths: list[list[str]] = []
                for path in all_paths:
                    if path not in unique_paths:
                        unique_paths.append(path)

                feasible_paths, feasibility_results = filter_feasible_paths(unique_paths, branches, timeout_ms=5000)
                integration["execution_paths"] = feasible_paths
                integration["path_analysis"] = {
                    "total_syntactic_paths": len(unique_paths),
                    "feasible_paths": len(feasible_paths),
                    "infeasible_paths": len(unique_paths) - len(feasible_paths),
                    "filter_effectiveness": (
                        100 * (1 - len(feasible_paths) / len(unique_paths)) if unique_paths else 0
                    ),
                }

                witnesses: list[dict[str, Any]] = []
                for path in feasible_paths:
                    path_id = "->".join(path)
                    result = feasibility_results.get(path_id)
                    if result and result.witness_values:
                        witnesses.append({"path": path, "witness": result.witness_values})
                if witnesses:
                    integration["test_witnesses"] = witnesses

        if entry.get("children"):
            add_execution_paths(entry["children"])


def build_class_hierarchy(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    classes: dict[str, dict[str, Any]] = {}

    def recurse(items: list[dict[str, Any]]) -> None:
        for entry in items:
            if entry.get("kind") in {"class", "enum"}:
                class_fqn = entry.get("_fqn")
                hinfo = hierarchy_info(entry)
                if class_fqn:
                    classes[class_fqn] = {
                        "bases": hinfo.get("resolved_base_classes", []),
                        "is_abstract": hinfo.get("is_abstract", False),
                        "contract_bases": hinfo.get("contract_base_classes", []),
                    }
            recurse(entry.get("children", []) or [])

    recurse(entries)
    return classes


def find_methods_by_owner(entries: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    methods_by_owner: dict[str, dict[str, dict[str, Any]]] = {}

    def recurse(items: list[dict[str, Any]]) -> None:
        for entry in items:
            if entry.get("kind") == "method":
                owner = hierarchy_info(entry).get("owner_class_fqn")
                name = entry.get("name")
                if owner and name:
                    methods_by_owner.setdefault(owner, {})[name] = entry
            recurse(entry.get("children", []) or [])

    recurse(entries)
    return methods_by_owner


def annotate_method_overrides(
        entries: list[dict[str, Any]],
        class_hierarchy: dict[str, dict[str, Any]],
        project_method_fqns: set[str],
) -> dict[str, list[str]]:
    methods_by_owner = find_methods_by_owner(entries)
    contract_methods: dict[str, list[str]] = {}

    def recurse(items: list[dict[str, Any]]) -> None:
        for entry in items:
            if entry.get("kind") == "method":
                hinfo = hierarchy_info(entry)
                owner = hinfo.get("owner_class_fqn")
                name = entry.get("name")
                if owner and name:
                    overrides: list[str] = []
                    implements_contract = False

                    for base in class_hierarchy.get(owner, {}).get("bases", []):
                        base_fqn = f"{base}.{name}"

                        base_method = methods_by_owner.get(base, {}).get(name)
                        if base_method is not None:
                            overrides.append(base_fqn)
                            if hierarchy_info(base_method).get("is_contract_method", False):
                                implements_contract = True
                                contract_methods.setdefault(base_fqn, []).append(entry.get("_fqn", ""))
                            continue

                        if base_fqn in project_method_fqns:
                            overrides.append(base_fqn)
                            if base in (class_hierarchy.get(owner, {}).get("contract_bases", []) or []):
                                implements_contract = True
                                contract_methods.setdefault(base_fqn, []).append(entry.get("_fqn", ""))

                    if overrides:
                        hinfo["overrides"] = overrides
                    if implements_contract:
                        hinfo["implements_contract_method"] = True

            recurse(entry.get("children", []) or [])

    recurse(entries)
    return {k: v for k, v in contract_methods.items()}


# ============================================================================
# Unit processing
# ============================================================================


def attach_children(entries_by_id: dict[str, dict[str, Any]], unit_entries: list[UnitIndexEntry]) -> list[
    dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for entry in sorted(unit_entries, key=lambda item: item.ordinal_within_parent):
        payload = entries_by_id[entry.id]
        payload.setdefault("children", [])
        if (
                entry.parent_id
                and entry.parent_id in entries_by_id
                and entry.parent_id != entry.owner_id
                and entry.parent_id != entry.id
        ):
            pass
        if entry.parent_id and entry.parent_id in entries_by_id and entry.parent_id != entry.id:
            entries_by_id[entry.parent_id].setdefault("children", []).append(payload)
        else:
            roots.append(payload)
    return roots


def merge_stage2(
        entries_by_id: dict[str, dict[str, Any]],
        unit_entries: list[UnitIndexEntry],
        stage2_lookup: dict[tuple[str, int, int], dict[str, Any]],
) -> None:
    for entry in unit_entries:
        payload = entries_by_id[entry.id]
        key = (entry.name, entry.lineno, entry.end_lineno)
        stage2_item = stage2_lookup.get(key)
        if stage2_item is None:
            continue

        ainfo = analysis_info(payload)
        ainfo["branches"] = stage2_item.get("branches", [])
        ainfo["total_eis"] = stage2_item.get("total_eis", len(stage2_item.get("branches", [])))


def count_all_entries(entries: list[dict[str, Any]]) -> int:
    total = 0
    for entry in entries:
        total += 1
        total += count_all_entries(entry.get("children", []))
    return total


def count_by_kind(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        kind = entry.get("kind", "unknown")
        counts[kind] = counts.get(kind, 0) + 1
        for child_kind, child_count in count_by_kind(entry.get("children", [])).items():
            counts[child_kind] = counts.get(child_kind, 0) + child_count
    return counts


def resolve_stage2_file(unit: UnitIndex, ei_file: Path | None, ei_root: Path | None) -> Path | None:
    if ei_file is not None:
        return ei_file
    if ei_root is None:
        return None
    candidate = ei_root / (unit.fully_qualified_name.replace(".", "/") + "_eis.yaml")
    return candidate if candidate.exists() else None


def process_unit(
        unit: UnitIndex,
        project_symbol_fqns: set[str],
        project_method_fqns: set[str],
        project_contract_classes: set[str],
        project_field_types_by_class: dict[str, dict[str, TypeRef]],
        stage2_file: Path | None,
        output_root: Path,
) -> dict[str, Any]:
    source_path = Path(unit.filepath)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    source_lines = source.splitlines()
    ast_index = build_module_index(tree, unit.fully_qualified_name)
    callable_inventory = {entry.fully_qualified_name: entry.id for entry in unit.entries}

    unit_types = inspect_unit_types(
        unit_fqn=unit.fully_qualified_name,
        source_path=source_path,
        tree=tree,
        ast_index=ast_index,
        project_fqns=project_symbol_fqns,
    )

    import_map = unit_types.import_map
    local_return_types = unit_types.local_return_types
    field_types_by_class: dict[str, dict[str, TypeRef]] = {
        **project_field_types_by_class,
        **unit_types.field_types_by_class,
    }
    analyzer = AstAnalyzer(
        source_lines=source_lines,
        unit_id=unit.unit_id,
        unit_fqn=unit.fully_qualified_name,
        callable_inventory=callable_inventory,
        project_fqns=project_symbol_fqns,
        import_map=import_map,
        local_return_types=local_return_types,
        field_types_by_class=field_types_by_class,
        project_contract_classes=project_contract_classes,
    )

    entries_by_id: dict[str, dict[str, Any]] = {}
    for entry in unit.entries:
        if entry.kind in CLASS_ENTRY_KINDS:
            node = ast_index.nodes_by_fqn_and_line.get((entry.fully_qualified_name, entry.lineno))
        elif entry.kind == "module_assignment":
            node = ast_index.assignment_nodes_by_fqn_and_line.get((entry.fully_qualified_name, entry.lineno))
        else:
            node = ast_index.nodes_by_fqn_and_line.get((entry.fully_qualified_name, entry.lineno))

        if entry.kind in {"unit_function", "method", "nested_function", "module_assignment"}:
            analyzer.current_callable_fqn = entry.fully_qualified_name
        else:
            analyzer.current_callable_fqn = None

        if node is None:
            payload = {
                "id": entry.id,
                "kind": entry.kind,
                "name": entry.name,
                "line_start": entry.lineno,
                "line_end": entry.end_lineno,
            }
        else:
            payload = analyzer.analyze_entry(entry, node)
        entries_by_id[entry.id] = payload

    merge_stage2(entries_by_id, unit.entries, build_stage2_lookup(load_stage2_yaml(stage2_file)))

    for entry in unit.entries:
        payload = entries_by_id[entry.id]
        ainfo = analysis_info(payload)
        branches_payload = ainfo.get("branches")
        if not branches_payload:
            continue

        branches = [Branch.from_dict(item) for item in branches_payload]
        known_types: dict[str, TypeRef] = payload.pop("_known_types", {}) or {}

        if entry.kind in CLASS_ENTRY_KINDS:
            node = ast_index.nodes_by_fqn_and_line.get((entry.fully_qualified_name, entry.lineno))
        elif entry.kind == "module_assignment":
            node = ast_index.assignment_nodes_by_fqn_and_line.get((entry.fully_qualified_name, entry.lineno))
        else:
            node = ast_index.nodes_by_fqn_and_line.get((entry.fully_qualified_name, entry.lineno))

        analyzer.current_callable_fqn = entry.fully_qualified_name

        if node is None:
            local_types: dict[str, TypeRef] = {}
        else:
            local_types = analyzer.build_local_type_map(node)
        merged_known_types: dict[str, TypeRef] = {**known_types, **local_types}

        if entry.fully_qualified_name == "project_resolution_engine.internal.util.multiformat.MultiformatSerializableMixin.flat_summary":
            print("KNOWN TYPES FOR", entry.fully_qualified_name)
            for name, type_ref in sorted(merged_known_types.items()):
                print(" ", name, "->", type_ref.to_dict())

        ainfo["integration_candidates"] = analyzer.find_integration_candidates(
            branches=branches,
            callable_id=payload["id"],
            callable_fqn=entry.fully_qualified_name,
            known_types=merged_known_types,
        )

    for payload in entries_by_id.values():
        payload.pop("_known_types", None)

    roots = attach_children(entries_by_id, unit.entries)
    annotate_entry_fqns(roots, unit.fully_qualified_name)
    annotate_method_owners(roots, unit.fully_qualified_name)
    mark_contract_methods(roots)

    class_hierarchy = build_class_hierarchy(roots)
    contract_methods = annotate_method_overrides(
        roots,
        class_hierarchy,
        project_method_fqns,
    )

    add_execution_paths(roots)

    for err in validate_feature_co_occurrences(roots):
        print(f"Warning: {err}")

    for entry in unit.entries:
        payload = entries_by_id[entry.id]
        analysis_info(payload).pop("needs_callable_analysis", None)

    kind_counts = count_by_kind(roots)
    inventory = {
        "unit": source_path.stem,
        "fully_qualified_name": unit.fully_qualified_name,
        "unit_id": unit.unit_id,
        "filepath": str(source_path),
        "language": unit.language,
        "type_hierarchy": class_hierarchy,
        "contract_methods": contract_methods,
        "entries": roots,
        "summary": {
            "total_entries": count_all_entries(roots),
            "classes": kind_counts.get("class", 0),
            "enums": kind_counts.get("enum", 0),
            "methods": kind_counts.get("method", 0),
            "functions": kind_counts.get("function", 0),
            "assignments": kind_counts.get("assignment", 0),
        },
    }

    output_path = output_root / (unit.fully_qualified_name.replace(".", "/") + ".inventory.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(inventory, sort_keys=False, allow_unicode=True, width=float("inf")),
        encoding="utf-8",
    )

    print(f"  → Unit ID: {unit.unit_id}")
    print(f"  → {inventory['summary']['total_entries']} entries")
    print(f"  → Saved: {output_path}")
    return inventory


# ============================================================================
# CLI
# ============================================================================


def select_unit(project_index: ProjectIndex, file_arg: Path | None, fqn_arg: str | None) -> UnitIndex | None:
    if file_arg is not None:
        resolved = file_arg.resolve()
        for unit in project_index.units:
            if Path(unit.filepath).resolve() == resolved:
                return unit
    if fqn_arg is not None:
        for unit in project_index.units:
            if unit.fully_qualified_name == fqn_arg:
                return unit
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stage 3 inventory from structured stage 1 and stage 2 outputs")
    parser.add_argument("--unit-index", type=Path, required=True, help="Structured stage 1 project index JSON")
    parser.add_argument("--file", type=Path, help="Python source file for the unit")
    parser.add_argument("--fqn", type=str, help="Fully qualified module name for the unit")
    parser.add_argument("--ei-file", type=Path, help="Stage 2 EI YAML for this unit")
    parser.add_argument("--ei-root", type=Path, help="Root directory containing per-unit stage 2 YAML files")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dist/inventory"),
        help="Root directory for inventory output",
    )
    args = parser.parse_args()

    if args.file is None and args.fqn is None:
        print("Error: provide --file or --fqn", file=sys.stderr)
        return 1
    if not args.unit_index.exists():
        print(f"Error: Unit index not found: {args.unit_index}", file=sys.stderr)
        return 1

    project_index = load_project_index(args.unit_index)
    project_contract_classes = collect_project_contract_classes(project_index)
    project_field_types_by_class = load_class_field_registry(args.unit_index)
    unit = select_unit(project_index, args.file, args.fqn)
    if unit is None:
        print("Error: could not resolve unit from --file/--fqn", file=sys.stderr)
        return 1

    stage2_file = resolve_stage2_file(unit, args.ei_file, args.ei_root)
    print(f"Processing: {unit.filepath}")
    print(f"  → FQN: {unit.fully_qualified_name}")
    if stage2_file is not None:
        print(f"  → Stage 2: {stage2_file}")
    else:
        print("  → Stage 2: none (inventory will be built without EI merge)")

    project_symbol_fqns: set[str] = set()
    project_method_fqns: set[str] = set()
    for indexed_unit in project_index.units:
        project_symbol_fqns.add(indexed_unit.fully_qualified_name)
        for entry in indexed_unit.entries:
            project_symbol_fqns.add(entry.fully_qualified_name)
            if entry.kind == "method":
                project_method_fqns.add(entry.fully_qualified_name)

    process_unit(
        unit,
        project_symbol_fqns,
        project_method_fqns,
        project_contract_classes,
        project_field_types_by_class,
        stage2_file,
        args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
