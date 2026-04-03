from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable

from pybastion_common.models import Branch
from pybastion_unit.shared.knowledge_base import (
    BUILTIN_METHODS,
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


ResolverFn = Callable[[str, dict[str, str]], tuple[str | None, str | None]]
SignatureFn = Callable[[Branch], str]


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


def _default_signature(branch: Branch) -> str:
    constraint = branch.constraint
    if constraint is not None and constraint.expr:
        return constraint.expr.strip()

    if branch.statement_outcome is not None and branch.statement_outcome.outcome:
        return branch.statement_outcome.outcome.strip()

    return (branch.description or "").strip()


def _base_name(target: str) -> str:
    return target.split(".")[-1].split("(")[0].strip()


def _root_name(target: str) -> str:
    return target.split(".", 1)[0].strip()


def _is_builtin_target(target: str) -> bool:
    base_name = _base_name(target)
    root_name = _root_name(target)

    return (
            base_name in PYTHON_BUILTINS
            or base_name in BUILTIN_METHODS
            or root_name in {"self", "cls", "object"}
    )


def _is_stdlib_target(target: str) -> bool:
    if target.startswith("collections.abc."):
        return True

    root_name = _root_name(target)
    return is_stdlib_module(root_name)


def _is_extlib_target(target: str) -> bool:
    root_name = _root_name(target)
    return root_name in COMMON_EXTLIB_MODULES


def _is_forbidden_integration_branch(branch: Branch) -> bool:
    owner = branch.owner_info
    if (
            owner is not None
            and owner.stmt_type == "Try"
            and owner.region == "except"
    ):
        return True

    constraint = branch.constraint
    if constraint is None:
        return False

    target = (constraint.operation_target or "").strip()
    if not target:
        return False

    if target.startswith("self.") or target.startswith("cls.") or target.startswith("object."):
        return True

    if _is_builtin_target(target):
        return True

    return False


def _is_operation_branch(branch: Branch) -> bool:
    constraint = branch.constraint
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

    if receiver_type and "." in raw_target and (
            receiver_type == "Path"
            or receiver_type.startswith("pathlib.")
            or is_stdlib_module(receiver_type.split(".", 1)[0])
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
        branches: list[Branch],
        callable_id: str,
        callable_fqn: str,
        unit_fqn: str,
        project_fqns: set[str],
        callable_inventory: dict[str, str],
        known_types: dict[str, str],
        resolve_target: ResolverFn,
        signature_for_branch: SignatureFn | None = None,
) -> list[IntegrationEntry]:
    signature_fn = signature_for_branch or _default_signature
    entries: list[IntegrationEntry] = []

    for branch in branches:
        if not _is_operation_branch(branch):
            continue

        if _is_forbidden_integration_branch(branch):
            continue

        constraint = branch.constraint
        raw_target = (constraint.operation_target or "").strip()
        if not raw_target:
            continue

        signature = signature_fn(branch)
        if not signature:
            continue

        resolved_target, resolved_type = resolve_target(raw_target, known_types)
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
                    ei_id=branch.id,
                    target=target,
                    kind=kind,
                ),
                ei_id=branch.id,
                line=branch.line,
                kind=kind,
                target=target,
                signature=signature,
                stmt_type=branch.stmt_type,
                owner_stmt_type=(
                    branch.owner_info.stmt_type
                    if branch.owner_info is not None
                    else None
                ),
                owner_region=(
                    branch.owner_info.region
                    if branch.owner_info is not None
                    else None
                ),
                classification=classification,
            )
        )

    return entries
