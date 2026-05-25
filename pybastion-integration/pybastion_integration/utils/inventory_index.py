from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CALLABLE_KINDS: set[str] = {"function", "method"}


def signature_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("signature_info", {}) or {}


def hierarchy_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("hierarchy_info", {}) or {}


def analysis_info(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("analysis_info", {}) or {}


@dataclass(slots=True)
class CallableContext:
    callable_id: str
    callable_name: str
    callable_kind: str
    callable_fqn: str
    unit_name: str
    unit_fqn: str
    entry: dict[str, Any]
    execution_items: list[dict[str, Any]]
    integration_by_ei: dict[str, list[dict[str, Any]]]


@dataclass(slots=True)
class InventoryIndex:
    callable_contexts: dict[str, CallableContext]
    fqn_to_callable_id: dict[str, str]
    ei_to_callable_id: dict[str, str]
    type_hierarchy: dict[str, dict[str, Any]]
    contract_methods: dict[str, list[str]]

    @staticmethod
    def empty() -> InventoryIndex:
        return InventoryIndex(
            callable_contexts={},
            fqn_to_callable_id={},
            ei_to_callable_id={},
            type_hierarchy={},
            contract_methods={},
        )

    def build_local_callables_by_unit(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = defaultdict(dict)

        for callable_id, context in self.callable_contexts.items():
            result[context.unit_fqn][context.callable_name] = callable_id

        return {unit_fqn: dict(name_map) for unit_fqn, name_map in result.items()}

    def build_contract_impl_index(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)

        for contract_method_fqn, impl_fqns in self.contract_methods.items():
            for impl_fqn in impl_fqns:
                impl_id = self.fqn_to_callable_id.get(impl_fqn)
                if impl_id is not None and impl_id not in result[contract_method_fqn]:
                    result[contract_method_fqn].append(impl_id)

        for callable_id, context in self.callable_contexts.items():
            hinfo = hierarchy_info(context.entry)
            if not hinfo.get("implements_contract_method", False):
                continue

            for overridden_fqn in hinfo.get("overrides", []) or []:
                if callable_id not in result[overridden_fqn]:
                    result[overridden_fqn].append(callable_id)

        return {contract_fqn: impl_ids for contract_fqn, impl_ids in result.items()}


def discover_inventory_files(inventories_root: Path) -> list[Path]:
    inventory_files = list(inventories_root.rglob("*.inventory.yaml"))
    inventory_files.extend(inventories_root.rglob("*_inventory.yaml"))
    return sorted(set(inventory_files))


def load_inventory(path: Path) -> dict[str, Any] | None:
    with open(path, "r", encoding="utf-8") as f:
        inventory = yaml.safe_load(f)

    if not inventory:
        return None

    if not isinstance(inventory, dict):
        raise TypeError(f"Inventory must be a mapping: {path}")

    return inventory


def build_callable_fqn(
        unit_fqn: str,
        ancestor_names: list[str],
        entry_name: str,
) -> str:
    parts = [unit_fqn, *ancestor_names, entry_name]
    return ".".join(part for part in parts if part)


def index_inventory(inventory: dict[str, Any]) -> InventoryIndex:
    unit_name = inventory["unit"]
    unit_fqn = inventory.get("fully_qualified_name", unit_name)

    callable_contexts: dict[str, CallableContext] = {}
    fqn_to_callable_id: dict[str, str] = {}
    ei_to_callable_id: dict[str, str] = {}

    type_hierarchy = inventory.get("type_hierarchy", {}) or {}
    contract_methods = inventory.get("contract_methods", {}) or {}

    def recurse(entries: list[dict[str, Any]], ancestors: list[str]) -> None:
        for entry in entries:
            kind = entry.get("kind", "unknown")
            name = entry.get("name", "unknown")
            entry_id = entry["id"]
            children = entry.get("children", []) or []

            ainfo = analysis_info(entry)
            execution_items = ainfo.get("execution_items", []) or []
            integration_candidates = ainfo.get("integration_candidates", []) or []

            callable_fqn = build_callable_fqn(
                unit_fqn=unit_fqn,
                ancestor_names=ancestors,
                entry_name=name,
            )

            if kind in CALLABLE_KINDS:
                if entry.get("is_executable") is False:
                    recurse(children, [*ancestors, name])
                    continue

                integration_by_ei: dict[str, list[dict[str, Any]]] = defaultdict(list)

                for candidate in integration_candidates:
                    ei_id = candidate.get("ei_id")
                    if ei_id:
                        integration_by_ei[ei_id].append(candidate)

                context = CallableContext(
                    callable_id=entry_id,
                    callable_name=name,
                    callable_kind=kind,
                    callable_fqn=callable_fqn,
                    unit_name=unit_name,
                    unit_fqn=unit_fqn,
                    entry=entry,
                    execution_items=execution_items,
                    integration_by_ei=dict(integration_by_ei),
                )

                callable_contexts[entry_id] = context
                fqn_to_callable_id[callable_fqn] = entry_id

                for ei in execution_items:
                    ei_id = ei.get("id")
                    if ei_id:
                        ei_to_callable_id[ei_id] = entry_id

            recurse(children, [*ancestors, name])

    recurse(inventory.get("entries", []) or [], [])

    return InventoryIndex(
        callable_contexts=callable_contexts,
        fqn_to_callable_id=fqn_to_callable_id,
        ei_to_callable_id=ei_to_callable_id,
        type_hierarchy=type_hierarchy,
        contract_methods=contract_methods,
    )


def merge_inventory_indexes(indexes: list[InventoryIndex]) -> InventoryIndex:
    merged = InventoryIndex.empty()

    for index in indexes:
        merged.callable_contexts.update(index.callable_contexts)
        merged.fqn_to_callable_id.update(index.fqn_to_callable_id)
        merged.ei_to_callable_id.update(index.ei_to_callable_id)
        merged.type_hierarchy.update(index.type_hierarchy)

        for contract_method_fqn, impl_fqns in index.contract_methods.items():
            bucket = merged.contract_methods.setdefault(contract_method_fqn, [])
            for impl_fqn in impl_fqns:
                if impl_fqn not in bucket:
                    bucket.append(impl_fqn)

    return merged


def load_all_inventories(inventory_paths: list[Path]) -> InventoryIndex:
    indexes: list[InventoryIndex] = []

    for path in inventory_paths:
        inventory = load_inventory(path)
        if inventory is None:
            continue

        indexes.append(index_inventory(inventory))

    return merge_inventory_indexes(indexes)


def load_inventory_index(inventories_root: Path) -> InventoryIndex:
    inventory_paths = discover_inventory_files(inventories_root)
    return load_all_inventories(inventory_paths)


def build_ei_details_index(
        inventory_index: InventoryIndex,
) -> dict[str, dict[str, Any]]:
    ei_index: dict[str, dict[str, Any]] = {}

    for context in inventory_index.callable_contexts.values():
        for ei in context.execution_items:
            ei_id = ei.get("id")
            if ei_id:
                ei_index[ei_id] = ei

    return ei_index


def build_callable_entry_index(
        inventory_index: InventoryIndex,
) -> dict[str, dict[str, Any]]:
    return {
        callable_id: context.entry
        for callable_id, context in inventory_index.callable_contexts.items()
    }


def execution_items_description(execution_items: dict[str, Any]) -> str:
    return str(
        execution_items.get("description")
        or execution_items.get("outcome")
        or execution_items.get("condition")
        or ""
    )


def execution_items_statement_outcome(execution_items: dict[str, Any]) -> dict[str, Any]:
    return execution_items.get("statement_outcome") or {}


def execution_items_constraint(execution_items: dict[str, Any]) -> dict[str, Any]:
    return execution_items.get("constraint") or {}
