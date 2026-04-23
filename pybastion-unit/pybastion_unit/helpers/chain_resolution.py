#!/usr/bin/env python3
"""
Experimental chain-resolution forensics.

Purpose:
- inspect how far a dotted call target can be resolved statically
- extract imports, class field types, param/local types, and owner metadata
- explain where resolution stops and why
- optionally inspect all integration candidates for a callable from an inventory yaml

Examples:
    python chain_forensics.py \
        --source project_resolution_engine/internal/orchestration.py \
        --unit-fqn project_resolution_engine.internal.orchestration \
        --callable-fqn project_resolution_engine.internal.orchestration.ArtifactCoordinator.resolve \
        --target self.repo.get

    python chain_forensics.py \
        --source project_resolution_engine/internal/orchestration.py \
        --unit-fqn project_resolution_engine.internal.orchestration \
        --inventory dist/inventory/project_resolution_engine/internal/orchestration.inventory.yaml \
        --callable-fqn project_resolution_engine.internal.orchestration.ArtifactCoordinator.resolve \
        --all-targets

Assumptions:
- bounded static reasoning only
- no fake runtime inference
- stop at ambiguity
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pybastion_common.models import TypeRef
from pybastion_unit.helpers.debug_tracing import (
    debug_print_resolution_state,
    debug_should_trace_target,
)
from pybastion_unit.helpers.type_indexing import (
    build_import_map,
    build_local_return_type_map,
    build_module_index,
    serialize_nested_type_map,
    serialize_type_map,
    self_attribute_name, build_field_types_by_class,
)

GENERIC_CONTAINER_TYPES: set[str] = {
    "list",
    "set",
    "tuple",
    "dict",
    "frozenset",
    "Optional",
    "type",
    "Sequence",
    "Mapping",
    "MutableMapping",
}


# ============================================================================
# Result model
# ============================================================================


@dataclass(frozen=True)
class ChainStep:
    index: int
    part: str
    action: str
    resolved_to: str | None = None
    basis: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class ChainResolutionResult:
    original_target: str
    parts: list[str]

    resolved_target: str
    resolved_receiver_type: str | None

    resolution_kind: str
    resolution_basis: str | None = None

    chain_steps: list[ChainStep] = field(default_factory=list)
    candidate_targets: list[str] = field(default_factory=list)

    stopped_at_index: int | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class CallableForensics:
    callable_fqn: str
    callable_name: str
    owner_class_fqn: str | None
    param_types: dict[str, TypeRef]
    local_types: dict[str, TypeRef]
    known_types: dict[str, TypeRef]


@dataclass(frozen=True)
class ModuleForensics:
    source_path: str
    unit_fqn: str
    import_map: dict[str, str]
    callable_inventory: dict[str, str]
    local_return_types: dict[str, TypeRef]
    field_types_by_class: dict[str, dict[str, TypeRef]]
    callable_forensics: dict[str, CallableForensics]


@dataclass(frozen=True)
class ResolutionContext:
    unit_fqn: str
    import_map: dict[str, str]
    callable_inventory: dict[str, str]
    local_return_types: dict[str, TypeRef]
    field_types_by_class: dict[str, dict[str, TypeRef]]
    callable_fqn: str
    owner_class_fqn: str | None
    known_types: dict[str, TypeRef]


# ============================================================================
# Generic helpers
# ============================================================================


def split_target_parts(target: str) -> list[str]:
    return [part.strip() for part in target.split(".") if part.strip()]


def class_object_target_type(type_ref: TypeRef, ctx: ResolutionContext) -> str | None:
    if type_ref.name != "type" or not type_ref.args:
        return None

    target = type_ref.args[0]
    target_name = target.name

    if target_name in ctx.import_map:
        target_name = ctx.import_map[target_name]

    matches = candidate_type_fqns(target_name, ctx)
    if len(matches) == 1:
        return matches[0]

    if target_name and target_name != "Any":
        return target_name

    return None


def normalize_type_name(type_name: str | TypeRef) -> str:
    if isinstance(type_name, TypeRef):
        return type_name.name
    return TypeRef.from_annotation_string(type_name).name


# ============================================================================
# Callable-local type extraction
# ============================================================================


def owner_class_fqn_for_callable(callable_fqn: str) -> str | None:
    parts = callable_fqn.split(".")
    if len(parts) < 2:
        return None
    return ".".join(parts[:-1]) if parts[-2][:1].isupper() else None


def build_param_type_map(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, TypeRef]:
    type_map: dict[str, TypeRef] = {}
    for arg in node.args.args + node.args.kwonlyargs:
        type_ref = TypeRef.from_annotation_ast(arg.annotation)
        if type_ref is not None:
            type_map[arg.arg] = type_ref
    return type_map


def build_local_type_map(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    local_return_types: dict[str, TypeRef],
    field_types_by_class: dict[str, dict[str, TypeRef]],
    owner_class_fqn: str | None,
) -> dict[str, TypeRef]:
    local_types: dict[str, TypeRef] = {}
    owner_fields = field_types_by_class.get(owner_class_fqn or "", {})

    def infer_expr_type(expr: ast.AST) -> TypeRef | None:
        if isinstance(expr, ast.Name):
            return local_types.get(expr.id)

        if isinstance(expr, ast.Attribute):
            attr_name = self_attribute_name(expr)
            if attr_name and attr_name in owner_fields:
                return owner_fields[attr_name]

        if isinstance(expr, ast.Call):
            func = expr.func
            if isinstance(func, ast.Name) and func.id in local_return_types:
                return local_return_types[func.id]
            if isinstance(func, ast.Attribute) and func.attr in local_return_types:
                return local_return_types[func.attr]

        return None

    for stmt in ast.walk(node):
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            type_ref = TypeRef.from_annotation_ast(stmt.annotation)
            if type_ref is not None:
                local_types[stmt.target.id] = type_ref
            continue

        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target_name = stmt.targets[0].id
            value = stmt.value

            if isinstance(value, ast.Name) and value.id == "self":
                if owner_class_fqn:
                    local_types[target_name] = TypeRef(name=owner_class_fqn)
                continue

            inferred = infer_expr_type(value)
            if inferred is not None:
                local_types[target_name] = inferred
                continue

        if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
            loop_var = stmt.target.id
            iter_type = infer_expr_type(stmt.iter)

            if iter_type is not None:
                element_type = extract_element_type(iter_type)
                if element_type is not None:
                    local_types[loop_var] = element_type

    return local_types


def build_callable_forensics(
        callable_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
        field_types_by_class: dict[str, dict[str, TypeRef]],
        local_return_types: dict[str, TypeRef],
) -> dict[str, CallableForensics]:
    result: dict[str, CallableForensics] = {}

    for callable_fqn, node in callable_nodes.items():
        owner_class_fqn = owner_class_fqn_for_callable(callable_fqn)
        param_types = build_param_type_map(node)
        local_types = build_local_type_map(
            node=node,
            local_return_types=local_return_types,
            field_types_by_class=field_types_by_class,
            owner_class_fqn=owner_class_fqn,
        )
        known_types = {**param_types, **local_types}

        result[callable_fqn] = CallableForensics(
            callable_fqn=callable_fqn,
            callable_name=node.name,
            owner_class_fqn=owner_class_fqn,
            param_types=param_types,
            local_types=local_types,
            known_types=known_types,
        )

    return result


# ============================================================================
# Element type extraction
# ============================================================================


def extract_element_type(type_ref: TypeRef) -> TypeRef | None:
    if type_ref.name in {"list", "set", "tuple", "frozenset", "Sequence"} and type_ref.args:
        return type_ref.args[0]
    return None


# ============================================================================
# Chain resolution
# ============================================================================


def candidate_type_fqns(type_name: str | TypeRef, ctx: ResolutionContext) -> list[str]:
    normalized = normalize_type_name(type_name)
    if not normalized or normalized == "Any":
        return []

    if normalized in ctx.import_map:
        normalized = ctx.import_map[normalized]

    matches: list[str] = []

    if normalized in ctx.callable_inventory:
        matches.append(normalized)

    for fqn in ctx.callable_inventory:
        if fqn.endswith(f".{normalized}") and fqn not in matches:
            matches.append(fqn)

    seen: set[str] = set()
    return [m for m in matches if not (m in seen or seen.add(m))]


def resolve_root_symbol(
        root: str,
        ctx: ResolutionContext,
) -> tuple[str | None, str | None, str | None, list[str], str | None]:
    if root in {"self", "cls"}:
        if ctx.owner_class_fqn:
            return None, ctx.owner_class_fqn, "owner_class", [], None
        return None, None, None, [], "owner_class_unknown"

    if root in ctx.known_types:
        type_ref = ctx.known_types[root]

        class_target = class_object_target_type(type_ref, ctx)
        if class_target is not None:
            return None, class_target, "known_class_object_type", [], None

        receiver_type = type_ref.name

        if receiver_type in ctx.import_map:
            receiver_type = ctx.import_map[receiver_type]

        if receiver_type in GENERIC_CONTAINER_TYPES:
            return None, receiver_type, "known_type_fallback", [], None

        matches = candidate_type_fqns(receiver_type, ctx)
        if len(matches) == 1:
            return None, matches[0], "known_type", [], None
        if len(matches) > 1:
            return None, None, None, matches, "ambiguous_known_type"

        if receiver_type and receiver_type != "Any":
            return None, receiver_type, "known_type_fallback", [], None
        return None, None, None, [], "known_type_unknown"

    if root in ctx.import_map:
        imported = ctx.import_map[root]
        return imported, imported, "import_map", [], None

    same_unit = f"{ctx.unit_fqn}.{root}"
    if same_unit in ctx.callable_inventory:
        return same_unit, same_unit, "same_unit_local", [], None

    if root == "sort_dict":
        print("DEBUG root:", root)
        print("DEBUG unit_fqn:", ctx.unit_fqn)
        print("DEBUG exact same_unit:", f"{ctx.unit_fqn}.{root}")
        print(
            "DEBUG suffix_matches:",
            [fqn for fqn in ctx.callable_inventory if fqn.endswith(f".{root}")]
        )

    suffix_matches = [
        fqn
        for fqn in ctx.callable_inventory
        if fqn.endswith(f".{root}")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0], suffix_matches[0], "same_unit_suffix", [], None
    if len(suffix_matches) > 1:
        return None, None, None, suffix_matches, "ambiguous_same_unit_suffix"

    return None, None, None, [], "root_unresolved"


def field_type_for_owner(owner_fqn: str | None, field_name: str, ctx: ResolutionContext) -> TypeRef | None:
    if owner_fqn is None:
        return None
    return ctx.field_types_by_class.get(owner_fqn, {}).get(field_name)


def resolve_attribute_hop(
        receiver_type: str | None,
        attr_name: str,
        ctx: ResolutionContext,
) -> tuple[str | None, str | None, str | None, list[str], str | None]:
    if receiver_type is None:
        return None, None, None, [], "receiver_type_unknown"

    field_type = field_type_for_owner(receiver_type, attr_name, ctx)
    if field_type:
        receiver_name = field_type.name

        if receiver_name in ctx.import_map:
            receiver_name = ctx.import_map[receiver_name]

        matches = candidate_type_fqns(receiver_name, ctx)
        if len(matches) == 1:
            return None, matches[0], "field_type", [], None
        if len(matches) > 1:
            return None, None, None, matches, "ambiguous_field_type"

        if receiver_name and receiver_name != "Any":
            return None, receiver_name, "field_type_fallback", [], None

    return None, None, None, [], "attribute_type_unknown"


def resolve_target_chain(ctx: ResolutionContext, target: str) -> ChainResolutionResult:
    debug = debug_should_trace_target(target)

    parts = split_target_parts(target)
    if debug:
        debug_print_resolution_state(
            "begin",
            target=target,
            parts=parts,
            known_types={k: v.to_dict() for k, v in ctx.known_types.items()},
        )

    if not parts:
        return ChainResolutionResult(
            original_target=target,
            parts=[],
            resolved_target=target,
            resolved_receiver_type=None,
            resolution_kind="unresolved",
            resolution_basis="empty_target",
            stop_reason="empty_target",
        )

    steps: list[ChainStep] = []

    root = parts[0]
    resolved_symbol_fqn, receiver_type, basis, candidates, stop_reason = resolve_root_symbol(root, ctx)

    if debug:
        debug_print_resolution_state(
            "after_root",
            root=root,
            resolved_symbol_fqn=resolved_symbol_fqn,
            receiver_type=receiver_type,
            basis=basis,
            candidates=candidates,
            stop_reason=stop_reason,
        )

    steps.append(
        ChainStep(
            index=0,
            part=root,
            action="resolve_root",
            resolved_to=resolved_symbol_fqn or receiver_type,
            basis=basis or stop_reason,
        )
    )

    if len(parts) == 1:
        if resolved_symbol_fqn:
            kind = "exact" if resolved_symbol_fqn in ctx.callable_inventory else "normalized"

            if debug:
                debug_print_resolution_state(
                    "return_single_resolved_symbol",
                    resolved_target=resolved_symbol_fqn,
                    resolved_receiver_type=receiver_type,
                    resolution_kind=kind,
                    resolution_basis=basis,
                )

            return ChainResolutionResult(
                original_target=target,
                parts=parts,
                resolved_target=resolved_symbol_fqn,
                resolved_receiver_type=receiver_type,
                resolution_kind=kind,
                resolution_basis=basis,
                chain_steps=steps,
            )

        if debug:
            debug_print_resolution_state(
                "return_single_unresolved",
                resolved_target=target,
                resolved_receiver_type=receiver_type,
                resolution_kind="ambiguous" if candidates else "unresolved",
                resolution_basis=basis,
                candidates=candidates,
                stop_reason=stop_reason,
            )

        return ChainResolutionResult(
            original_target=target,
            parts=parts,
            resolved_target=target,
            resolved_receiver_type=receiver_type,
            resolution_kind="ambiguous" if candidates else "unresolved",
            resolution_basis=basis,
            chain_steps=steps,
            candidate_targets=candidates,
            stopped_at_index=0,
            stop_reason=stop_reason,
        )

    if resolved_symbol_fqn and len(parts) == 2:
        candidate = f"{resolved_symbol_fqn}.{parts[1]}"
        kind = "exact" if candidate in ctx.callable_inventory else "normalized"

        if debug:
            debug_print_resolution_state(
                "resolved_symbol_short_circuit",
                candidate=candidate,
                resolved_receiver_type=resolved_symbol_fqn,
                resolution_kind=kind,
                resolution_basis=basis,
            )

        steps.append(
            ChainStep(
                index=1,
                part=parts[1],
                action="append_from_resolved_root",
                resolved_to=candidate,
                basis="resolved_symbol_fqn",
            )
        )
        return ChainResolutionResult(
            original_target=target,
            parts=parts,
            resolved_target=candidate,
            resolved_receiver_type=resolved_symbol_fqn,
            resolution_kind=kind,
            resolution_basis=basis,
            chain_steps=steps,
        )

    if receiver_type is None:
        if debug:
            debug_print_resolution_state(
                "return_no_receiver_type",
                resolved_target=target,
                resolution_kind="ambiguous" if candidates else "unresolved",
                resolution_basis=basis,
                candidates=candidates,
                stop_reason=stop_reason,
            )

        return ChainResolutionResult(
            original_target=target,
            parts=parts,
            resolved_target=target,
            resolved_receiver_type=None,
            resolution_kind="ambiguous" if candidates else "unresolved",
            resolution_basis=basis,
            chain_steps=steps,
            candidate_targets=candidates,
            stopped_at_index=0,
            stop_reason=stop_reason,
        )

    for index, part in enumerate(parts[1:-1], start=1):
        if debug:
            debug_print_resolution_state(
                "before_attribute_hop",
                index=index,
                part=part,
                incoming_receiver_type=receiver_type,
            )

        _, next_receiver_type, hop_basis, hop_candidates, hop_stop = resolve_attribute_hop(
            receiver_type,
            part,
            ctx,
        )

        if debug:
            debug_print_resolution_state(
                "after_attribute_hop",
                index=index,
                part=part,
                next_receiver_type=next_receiver_type,
                hop_basis=hop_basis,
                hop_candidates=hop_candidates,
                hop_stop=hop_stop,
            )

        steps.append(
            ChainStep(
                index=index,
                part=part,
                action="resolve_attribute_hop",
                resolved_to=next_receiver_type,
                basis=hop_basis or hop_stop,
            )
        )

        if next_receiver_type is None:
            if debug:
                debug_print_resolution_state(
                    "return_hop_failed",
                    index=index,
                    failed_part=part,
                    last_receiver_type=receiver_type,
                    resolution_kind="ambiguous" if hop_candidates else "unresolved",
                    resolution_basis=hop_basis or basis,
                    hop_candidates=hop_candidates,
                    hop_stop=hop_stop,
                )

            return ChainResolutionResult(
                original_target=target,
                parts=parts,
                resolved_target=target,
                resolved_receiver_type=receiver_type,
                resolution_kind="ambiguous" if hop_candidates else "unresolved",
                resolution_basis=hop_basis or basis,
                chain_steps=steps,
                candidate_targets=hop_candidates,
                stopped_at_index=index,
                stop_reason=hop_stop,
            )

        receiver_type = next_receiver_type

    final_part = parts[-1]
    resolved_target = f"{receiver_type}.{final_part}"
    kind = "exact" if resolved_target in ctx.callable_inventory else "contract"

    if debug:
        debug_print_resolution_state(
            "before_final_return",
            final_receiver_type=receiver_type,
            final_part=final_part,
            resolved_target=resolved_target,
            resolution_kind=kind,
            resolution_basis=basis or "receiver_type",
        )

    steps.append(
        ChainStep(
            index=len(parts) - 1,
            part=final_part,
            action="append_final_callable",
            resolved_to=resolved_target,
            basis="receiver_type",
        )
    )

    return ChainResolutionResult(
        original_target=target,
        parts=parts,
        resolved_target=resolved_target,
        resolved_receiver_type=receiver_type,
        resolution_kind=kind,
        resolution_basis=basis or "receiver_type",
        chain_steps=steps,
    )


# ============================================================================
# Inventory support
# ============================================================================


def load_inventory_targets(inventory_path: Path, callable_fqn: str) -> list[str]:
    payload = yaml.safe_load(inventory_path.read_text(encoding="utf-8")) or {}
    unit_fqn = payload.get("fully_qualified_name") or payload.get("unit")

    targets: list[str] = []
    seen_callables: list[str] = []

    def recurse(entries: list[dict[str, Any]], ancestors: list[str]) -> None:
        for entry in entries:
            name = entry.get("name", "")
            kind = entry.get("kind", "")
            fq = ".".join([unit_fqn, *ancestors, name])
            seen_callables.append(fq)

            if fq == callable_fqn:
                ainfo = entry.get("analysis_info", {}) or {}
                candidates = ainfo.get("integration_candidates", []) or []
                print(f"[debug] matched callable: {fq}")
                print(f"[debug] integration candidate count: {len(candidates)}")
                for candidate in candidates:
                    target = candidate.get("target")
                    if target:
                        targets.append(target)

            next_ancestors = [*ancestors]
            if kind in {"class", "enum", "method", "function", "assignment"}:
                next_ancestors.append(name)

            recurse(entry.get("children", []) or [], next_ancestors)

    recurse(payload.get("entries", []) or [], [])

    if not targets:
        print(f"[debug] callable not found or had no targets: {callable_fqn}")
        print("[debug] first few reconstructed callables:")
        for item in seen_callables[:25]:
            print(f"  {item}")

    return targets


# ============================================================================
# Module assembly
# ============================================================================


def inspect_module(source_path: Path, unit_fqn: str) -> ModuleForensics:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    import_map, _ = build_import_map(tree)
    local_return_types = build_local_return_type_map(tree)

    ast_index = build_module_index(tree, unit_fqn)

    callable_inventory = {
        fqn: fqn
        for fqn in sorted(ast_index.module_symbol_fqns)
    }

    field_types_by_class = build_field_types_by_class(
        class_nodes=ast_index.class_nodes,
        import_map=import_map,
        local_return_types=local_return_types,
    )

    callable_forensics = build_callable_forensics(
        callable_nodes=ast_index.callable_nodes,
        field_types_by_class=field_types_by_class,
        local_return_types=local_return_types,
    )

    return ModuleForensics(
        source_path=str(source_path),
        unit_fqn=unit_fqn,
        import_map=import_map,
        callable_inventory=callable_inventory,
        local_return_types=local_return_types,
        field_types_by_class=field_types_by_class,
        callable_forensics=callable_forensics,
    )


def build_resolution_context(module: ModuleForensics, callable_fqn: str) -> ResolutionContext:
    callable_meta = module.callable_forensics.get(callable_fqn)
    if callable_meta is None:
        raise KeyError(f"Callable not found in source AST: {callable_fqn}")

    return ResolutionContext(
        unit_fqn=module.unit_fqn,
        import_map=module.import_map,
        callable_inventory=module.callable_inventory,
        local_return_types=module.local_return_types,
        field_types_by_class=module.field_types_by_class,
        callable_fqn=callable_fqn,
        owner_class_fqn=callable_meta.owner_class_fqn,
        known_types=callable_meta.known_types,
    )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental chain-resolution forensics")
    parser.add_argument("--source", type=Path, required=True, help="Python source file")
    parser.add_argument("--unit-fqn", type=str, required=True, help="Module FQN for the source file")
    parser.add_argument("--callable-fqn", type=str, required=True, help="Callable FQN to analyze")

    parser.add_argument("--target", type=str, help="Single target expression to resolve")
    parser.add_argument("--inventory", type=Path, help="Inventory yaml to pull integration targets from")
    parser.add_argument("--all-targets", action="store_true", help="Resolve all inventory targets for the callable")

    parser.add_argument("--dump-module", action="store_true", help="Dump extracted module forensics")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of YAML-ish text")
    return parser.parse_args()


def render_result(result: ChainResolutionResult) -> dict[str, Any]:
    return {
        "original_target": result.original_target,
        "resolved_target": result.resolved_target,
        "resolved_receiver_type": result.resolved_receiver_type,
        "resolution_kind": result.resolution_kind,
        "resolution_basis": result.resolution_basis,
        "stopped_at_index": result.stopped_at_index,
        "stop_reason": result.stop_reason,
        "candidate_targets": result.candidate_targets,
        "chain_steps": [asdict(step) for step in result.chain_steps],
    }


def main() -> int:
    args = parse_args()

    module = inspect_module(args.source, args.unit_fqn)

    if args.dump_module:
        payload = {
            "source_path": module.source_path,
            "unit_fqn": module.unit_fqn,
            "import_map": module.import_map,
            "local_return_types": serialize_type_map(module.local_return_types),
            "field_types_by_class": serialize_nested_type_map(module.field_types_by_class),
            "callable_forensics": {
                fq: asdict(meta) for fq, meta in module.callable_forensics.items()
            },
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=False))
        else:
            print(yaml.dump(payload, sort_keys=False, allow_unicode=True, width=float("inf")))
        return 0

    ctx = build_resolution_context(module, args.callable_fqn)

    targets: list[str] = []

    if args.target:
        targets.append(args.target)

    if args.all_targets:
        if not args.inventory:
            raise SystemExit("--all-targets requires --inventory")
        targets.extend(load_inventory_targets(args.inventory, args.callable_fqn))

    if not targets:
        if args.all_targets:
            raise SystemExit(
                f"No inventory targets found for callable {args.callable_fqn} "
                f"in inventory {args.inventory}"
            )
        raise SystemExit("Provide --target or --all-targets")

    rendered = []
    seen: set[str] = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        rendered.append(render_result(resolve_target_chain(ctx, target)))

    payload = {
        "callable_fqn": args.callable_fqn,
        "owner_class_fqn": ctx.owner_class_fqn,
        "known_types": serialize_type_map(ctx.known_types),
        "field_types_for_owner": serialize_type_map(
            module.field_types_by_class.get(ctx.owner_class_fqn or "", {})
        ),
        "results": rendered,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print(yaml.dump(payload, sort_keys=False, allow_unicode=True, width=float("inf")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())