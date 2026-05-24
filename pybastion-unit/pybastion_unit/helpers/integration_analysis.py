from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable

from pybastion_common.models import ExecutionItem, TypeRef
from pybastion_common.knowledge_base import (
    BUILTIN_METHODS,
    BUILTIN_RECEIVER_METHODS,
    COMMON_EXTLIB_MODULES,
    PYTHON_BUILTINS,
    is_stdlib_module,
)


@dataclass(frozen=True)
class IntegrationEntry:
    id: str
    ei_id: str
    line: int
    kind: str
    target: str
    signature: str
    stmt_type: str | None
    owner_stmt_type: str | None
    owner_region: str | None
    classification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ResolverFn = Callable[[str, dict[str, TypeRef]], tuple[str | None, str | None]]
SignatureFn = Callable[[ExecutionItem], str]


def _integration_id(
    *,
    callable_id: str,
    ei_id: str,
    target: str,
    kind: str,
) -> str:
    seed = f"{callable_id}|{ei_id}|{kind}|{target}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10].upper()
    return f"{callable_id}_I{digest}"


def default_signature(ei: ExecutionItem) -> str:
    constraint = ei.constraint
    if constraint is not None and constraint.expr:
        return constraint.expr.strip()

    if ei.statement_outcome is not None and ei.statement_outcome.outcome:
        return ei.statement_outcome.outcome.strip()

    return (ei.description or "").strip()


def _base_name(target: str) -> str:
    return target.split(".")[-1].split("(")[0].strip()


def _root_name(target: str) -> str:
    return target.split(".", 1)[0].strip()


def _normalize_receiver_type(receiver_type: str | None) -> str | None:
    if not receiver_type:
        return None

    normalized = receiver_type.strip()

    if "::" in normalized:
        container, _inner = normalized.split("::", 1)
        return container.strip()

    if "[" in normalized:
        normalized = normalized.split("[", 1)[0].strip()

    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]

    return normalized or None


def is_builtin_receiver_method(receiver_type: str | None, method_name: str) -> bool:
    normalized = _normalize_receiver_type(receiver_type)
    if not normalized:
        return False
    return f"{normalized}::{method_name}" in BUILTIN_RECEIVER_METHODS


def _is_builtin_target(target: str) -> bool:
    base_name = _base_name(target)

    # Only plain builtin-like symbols are filtered here.
    # Do not treat dotted receiver calls as builtin solely by basename.
    if "." not in target:
        return base_name in PYTHON_BUILTINS or base_name in BUILTIN_METHODS

    return base_name in PYTHON_BUILTINS


def _is_stdlib_target(target: str) -> bool:
    if target.startswith("collections.abc."):
        return True

    root_name = _root_name(target)
    return is_stdlib_module(root_name)


def _is_extlib_target(target: str) -> bool:
    root_name = _root_name(target)
    return root_name in COMMON_EXTLIB_MODULES


def _is_collapsible_operation_target(
    resolved_target: str | None,
    collapsible_operation_fqns: set[str],
) -> bool:
    if not resolved_target:
        return False

    return resolved_target in collapsible_operation_fqns


def _is_forbidden_integration_ei(ei: ExecutionItem) -> bool:
    owner = ei.owner_info
    if owner is not None and owner.stmt_type == "Try" and owner.region == "except":
        return True

    for decorator in ei.decorators:
        if decorator.get("name", "") in ("MechanicalOperation", "UtilityOperation"):
            return True

    constraint = ei.constraint
    if constraint is None:
        return False

    target = (constraint.operation_target or "").strip()
    if not target:
        return False

    if _is_builtin_target(target):
        return True

    return False


def _is_operation_ei(ei: ExecutionItem) -> bool:
    constraint = ei.constraint
    return (
        constraint is not None
        and constraint.constraint_type == "operation"
        and bool((constraint.operation_target or "").strip())
    )


def _classify_integration_target(
    *,
    raw_target: str,
    resolved_target: str | None,
    resolved_type: str | None,
    callable_fqn: str,
    unit_fqn: str,
    project_fqns: set[str],
    callable_inventory: dict[str, str],
) -> dict[str, Any]:
    candidate = (resolved_target or raw_target or "").strip()
    receiver_type = (resolved_type or "").strip()
    method_name = _base_name(raw_target)

    if resolved_target is None or resolved_target == "":
        return {
            "is_integration": True,
            "kind": "unknown",
            "resolved_target": raw_target,
        }

    if not candidate:
        return {
            "is_integration": False,
            "kind": "unknown",
            "resolved_target": raw_target,
        }

    inventory_fqns = set(callable_inventory.keys())

    if candidate == callable_fqn:
        return {
            "is_integration": False,
            "kind": "same_callable",
            "resolved_target": candidate,
        }

    if candidate in inventory_fqns or candidate in project_fqns:
        if candidate.startswith(f"{unit_fqn}."):
            return {
                "is_integration": False,
                "kind": "same_unit",
                "resolved_target": candidate,
            }
        return {
            "is_integration": True,
            "kind": "interunit",
            "resolved_target": candidate,
        }

    if is_builtin_receiver_method(receiver_type, method_name):
        return {
            "is_integration": False,
            "kind": "builtin",
            "resolved_target": candidate,
        }

    builtin_receiver_types = {
        "str",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "bytes",
        "bytearray",
    }
    if receiver_type in builtin_receiver_types:
        return {
            "is_integration": False,
            "kind": "builtin",
            "resolved_target": candidate,
        }

    if (
        receiver_type
        and "." in raw_target
        and (
            receiver_type == "Path"
            or receiver_type.startswith("pathlib.")
            or is_stdlib_module(receiver_type.split(".", 1)[0])
        )
    ):
        method_name = raw_target.split(".")[-1]
        stdlib_target = f"{receiver_type}.{method_name}"
        return {
            "is_integration": True,
            "kind": "stdlib",
            "resolved_target": stdlib_target,
        }

    if _is_builtin_target(candidate):
        return {
            "is_integration": False,
            "kind": "builtin",
            "resolved_target": candidate,
        }

    if _is_stdlib_target(candidate):
        return {
            "is_integration": True,
            "kind": "stdlib",
            "resolved_target": candidate,
        }

    if _is_extlib_target(candidate):
        return {
            "is_integration": True,
            "kind": "extlib",
            "resolved_target": candidate,
        }

    return {
        "is_integration": True,
        "kind": "unknown",
        "resolved_target": candidate,
    }


def build_integration_entries(
    *,
    execution_items: list[ExecutionItem],
    callable_id: str,
    callable_fqn: str,
    unit_fqn: str,
    project_fqns: set[str],
    callable_inventory: dict[str, str],
    collapsible_operation_fqns: set[str],
    known_types: dict[str, TypeRef],
    resolve_target: ResolverFn,
    signature_for_ei: SignatureFn | None = None,
) -> list[IntegrationEntry]:
    signature_fn = signature_for_ei or default_signature
    entries: list[IntegrationEntry] = []

    for ei in execution_items:
        if not _is_operation_ei(ei):
            continue

        if _is_forbidden_integration_ei(ei):
            continue

        constraint = ei.constraint
        raw_target = (constraint.operation_target or "").strip()
        if not raw_target:
            continue

        signature = signature_fn(ei)
        if not signature:
            continue

        resolved_target, resolved_type = resolve_target(raw_target, known_types)
        if _is_collapsible_operation_target(
            resolved_target,
            collapsible_operation_fqns,
        ):
            continue

        classification = _classify_integration_target(
            raw_target=raw_target,
            resolved_target=resolved_target,
            resolved_type=resolved_type,
            callable_fqn=callable_fqn,
            unit_fqn=unit_fqn,
            project_fqns=project_fqns,
            callable_inventory=callable_inventory,
        )

        if not classification["is_integration"]:
            continue

        kind = str(classification["kind"])
        target = str(classification["resolved_target"])

        entries.append(
            IntegrationEntry(
                id=_integration_id(
                    callable_id=callable_id,
                    ei_id=ei.id,
                    target=target,
                    kind=kind,
                ),
                ei_id=ei.id,
                line=ei.line,
                kind=kind,
                target=target,
                signature=signature,
                stmt_type=ei.stmt_type,
                owner_stmt_type=(
                    ei.owner_info.stmt_type if ei.owner_info is not None else None
                ),
                owner_region=(
                    ei.owner_info.region if ei.owner_info is not None else None
                ),
                classification=classification,
            )
        )

    return entries
