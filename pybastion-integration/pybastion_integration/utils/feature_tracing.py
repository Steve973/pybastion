#!/usr/bin/env python3
"""
Feature tracing helpers.

This module inventories feature-flow markers from PyBastion unit inventory YAML
files.

Feature markers provide feature intent. Execution and control-flow facts come
from inventory EI metadata. When a marker is attached to a StatementAnchor, this
helper resolves the full same-line EI decomposition for the marked statement.

The current exported flow path generation is still a first-pass branch-forced
path probe. The newer FeatureFlowCase / FeaturePathSegment model is the intended
foundation for full branch-state traversal and segment assembly.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

# =============================================================================
# Marker and EI classification constants
# =============================================================================

FEATURE_MARKER_NAMES: set[str] = {
    'FeatureStart',
    'FeatureTrace',
    'FeatureBranch',
    'FeatureConverge',
    'FeatureEnd',
    'FeatureEndConditional',
}

CONTROL_STMT_TYPES: set[str] = {
    'If',
    'For',
    'AsyncFor',
    'While',
    'Match',
}

CONTROL_CONSTRAINT_TYPES: set[str] = {
    'condition',
    'iteration',
    'match_case',
}

TERMINAL_VIA_VALUES: set[str] = {
    'return',
    'implicit-return',
    'raise',
    'exception',
}


# =============================================================================
# Feature flow enums
# =============================================================================

class FeatureFlowCaseStatus(StrEnum):
    ACTIVE = 'active'
    COMPLETED = 'completed'
    UNRESOLVED = 'unresolved'


class FeatureFlowEndKind(StrEnum):
    FEATURE_END = 'feature_end'
    FEATURE_END_CONDITIONAL = 'feature_end_conditional'
    UNRESOLVED = 'unresolved'


class FeatureFlowOutcomeKind(StrEnum):
    SUCCESS = 'success'
    FAILURE = 'failure'
    CONDITIONAL = 'conditional'
    UNKNOWN = 'unknown'


class FeaturePathSegmentReason(StrEnum):
    START_TO_END = 'start_to_end'
    START_TO_BRANCH = 'start_to_branch'
    BRANCH_TARGET = 'branch_target'
    BRANCH_TO_CONVERGE = 'branch_to_converge'
    CONVERGE_TO_BRANCH = 'converge_to_branch'
    SHARED_TAIL = 'shared_tail'
    TO_END = 'to_end'
    TO_CONDITIONAL_END = 'to_conditional_end'


# =============================================================================
# Marker inventory models
# =============================================================================

@dataclass(frozen=True)
class EiExecutionMetadata:
    ei_id: str
    line: int | None
    stmt_type: str | None
    description: str | None
    condition: str | None
    statement_outcome: dict[str, Any] | None
    conditional_targets: list[dict[str, Any]]
    disruptive_outcomes: list[dict[str, Any]]
    constraint: dict[str, Any] | None
    is_terminal: bool
    terminates_via: str | None


@dataclass(frozen=True)
class MarkedStatementMetadata:
    entry_ei: EiExecutionMetadata
    eis: list[EiExecutionMetadata]
    terminal_eis: list[EiExecutionMetadata]
    control_eis: list[EiExecutionMetadata]


@dataclass(frozen=True)
class FeatureBranchSelection:
    branch: str | None
    control_polarity: bool | None
    selected_target_eis: list[str]
    selected_conditions: list[str]


@dataclass(frozen=True)
class FeatureMarkerRecord:
    feature_name: str
    marker_name: str
    inventory_path: str
    unit: str | None
    unit_fqn: str | None
    callable_id: str
    callable_name: str | None
    callable_fqn: str | None
    node_id: str
    kwargs: dict[str, Any]
    line: int | None
    stmt_type: str | None
    description: str | None
    condition: str | None
    marker_ei: EiExecutionMetadata
    marked_statement: MarkedStatementMetadata | None
    branch_selection: FeatureBranchSelection | None = None


@dataclass
class FeatureMarkerInventory:
    feature_name: str
    starts: list[FeatureMarkerRecord] = field(default_factory=list)
    traces: list[FeatureMarkerRecord] = field(default_factory=list)
    branches: list[FeatureMarkerRecord] = field(default_factory=list)
    converges: list[FeatureMarkerRecord] = field(default_factory=list)
    ends: list[FeatureMarkerRecord] = field(default_factory=list)
    conditional_ends: list[FeatureMarkerRecord] = field(default_factory=list)

    def all_records(self) -> list[FeatureMarkerRecord]:
        return [
            *self.starts,
            *self.traces,
            *self.branches,
            *self.converges,
            *self.ends,
            *self.conditional_ends,
        ]


# =============================================================================
# Segment/case models for the real branch-aware expansion engine
# =============================================================================

@dataclass(frozen=True)
class FeaturePathSegment:
    feature_name: str
    segment_branch_path: tuple[str, ...]
    start_ei: str
    end_ei: str
    path: list[str]
    reason: FeaturePathSegmentReason


@dataclass(frozen=True)
class FeatureFlowCase:
    feature_name: str
    case_branch_path: tuple[str, ...]
    active_branch_path: tuple[str, ...]
    current_eis: tuple[str, ...]
    segments: tuple[FeaturePathSegment, ...]
    status: FeatureFlowCaseStatus = FeatureFlowCaseStatus.ACTIVE
    end_kind: FeatureFlowEndKind | None = None
    outcome_kind: FeatureFlowOutcomeKind | None = None


@dataclass(frozen=True)
class FeatureBranchPoint:
    feature_name: str
    branch_point_id: str
    marker_node_id: str
    control_ei_id: str
    branch_markers: tuple[FeatureMarkerRecord, ...]

# =============================================================================
# Current exported flow path models
# =============================================================================

@dataclass(frozen=True)
class FeatureFlowPath:
    feature_name: str
    end_marker_type: str
    start_marker: str
    end_marker: str
    variant: str
    required_branch_marker: str | None
    required_branch: str | None
    required_target_eis: list[str]
    path: list[dict[str, Any]]
    path_length: int
    reason: str


@dataclass(frozen=True)
class DiscardedFeatureFlowPath(FeatureFlowPath):
    discarded_reason: str
    duplicate_of_variant: str
    duplicate_of_required_branch: str | None


# =============================================================================
# Inventory loading and traversal
# =============================================================================

def load_inventory(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    return payload or {}


def find_inventory_files(inventory_root: Path) -> list[Path]:
    return sorted(inventory_root.rglob('*.inventory.yaml'))


def iter_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(items: list[dict[str, Any]]) -> None:
        for item in items:
            result.append(item)
            visit(item.get('children', []) or [])

    visit(entries)
    return result


# =============================================================================
# EI metadata extraction
# =============================================================================

def ei_execution_metadata(branch: dict[str, Any]) -> EiExecutionMetadata:
    statement_outcome = branch.get('statement_outcome')
    disruptive_outcomes = branch.get('disruptive_outcomes', []) or []

    terminates_via = branch.get('terminates_via')
    is_terminal = bool(branch.get('is_terminal', False))

    if statement_outcome:
        terminates_via = terminates_via or statement_outcome.get('terminates_via')
        is_terminal = is_terminal or bool(statement_outcome.get('is_terminal', False))

    for outcome in disruptive_outcomes:
        terminates_via = terminates_via or outcome.get('terminates_via')
        is_terminal = is_terminal or bool(outcome.get('is_terminal', False))

    return EiExecutionMetadata(
        ei_id=str(branch.get('id')),
        line=branch.get('line'),
        stmt_type=branch.get('stmt_type'),
        description=branch.get('description'),
        condition=branch.get('condition'),
        statement_outcome=statement_outcome,
        conditional_targets=branch.get('conditional_targets', []) or [],
        disruptive_outcomes=disruptive_outcomes,
        constraint=branch.get('constraint'),
        is_terminal=is_terminal,
        terminates_via=terminates_via,
    )


def parse_control_polarity(value: Any) -> bool | None:
    if value is True or value == 'true':
        return True
    if value is False or value == 'false':
        return False
    return None


def parse_ei_sort_key(ei_id: str) -> tuple[str, int, str]:
    if '_E' not in ei_id:
        return ei_id, 10 ** 9, ei_id

    prefix, suffix = ei_id.rsplit('_E', 1)
    digits: list[str] = []

    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break

    number = int(''.join(digits)) if digits else 10 ** 9
    return prefix, number, ei_id


def is_terminal_ei(ei: EiExecutionMetadata) -> bool:
    if ei.is_terminal:
        return True

    if ei.terminates_via in TERMINAL_VIA_VALUES:
        return True

    if ei.statement_outcome is not None:
        if ei.statement_outcome.get('terminates_via') in TERMINAL_VIA_VALUES:
            return True

    for outcome in ei.disruptive_outcomes:
        if outcome.get('terminates_via') in TERMINAL_VIA_VALUES:
            return True

    if ei.description and ei.description.startswith('raises '):
        return True

    return False


def is_control_ei(ei: EiExecutionMetadata) -> bool:
    if ei.conditional_targets:
        return True

    if ei.constraint is not None:
        if ei.constraint.get('constraint_type') in CONTROL_CONSTRAINT_TYPES:
            return True

    if ei.stmt_type in CONTROL_STMT_TYPES:
        return True

    return False


# =============================================================================
# Marker attachment and marked-statement resolution
# =============================================================================

def statement_anchor_target_ei_id(branch: dict[str, Any]) -> str | None:
    if branch.get('stmt_type') != 'StatementAnchor':
        return None

    statement_outcome = branch.get('statement_outcome') or {}
    target_ei = statement_outcome.get('target_ei')

    return str(target_ei) if target_ei else None


def marked_statement_metadata(
        *,
        marker_branch: dict[str, Any],
        branches_by_id: dict[str, dict[str, Any]],
        branches: list[dict[str, Any]],
) -> MarkedStatementMetadata | None:
    entry_ei_id = statement_anchor_target_ei_id(marker_branch)

    if entry_ei_id is None:
        return None

    entry_branch = branches_by_id.get(entry_ei_id)

    if entry_branch is None:
        return None

    entry_ei = ei_execution_metadata(entry_branch)
    line = entry_branch.get('line')

    statement_eis = [
        ei_execution_metadata(branch)
        for branch in branches
        if branch.get('line') == line
           and branch.get('stmt_type') != 'StatementAnchor'
    ]

    statement_eis.sort(key=lambda item: parse_ei_sort_key(item.ei_id))

    terminal_eis = [
        item
        for item in statement_eis
        if is_terminal_ei(item)
    ]

    control_eis = [
        item
        for item in statement_eis
        if is_control_ei(item)
    ]

    return MarkedStatementMetadata(
        entry_ei=entry_ei,
        eis=statement_eis,
        terminal_eis=terminal_eis,
        control_eis=control_eis,
    )


def add_unique_text(target: list[str], value: Any) -> None:
    if value is None:
        return

    text = str(value)
    if text not in target:
        target.append(text)


def derive_feature_branch_selection(
        *,
        marker_kwargs: dict[str, Any],
        marked_statement: MarkedStatementMetadata | None,
) -> FeatureBranchSelection | None:
    if marked_statement is None:
        return None

    branch = marker_kwargs.get('branch')
    control_polarity = parse_control_polarity(
        marker_kwargs.get('control_polarity')
    )

    selected_target_eis: list[str] = []
    selected_conditions: list[str] = []

    if control_polarity is not None:
        for control_ei in marked_statement.control_eis:
            for target in control_ei.conditional_targets:
                if target.get('condition_result') != control_polarity:
                    continue

                add_unique_text(selected_target_eis, target.get('target_ei'))
                add_unique_text(selected_conditions, target.get('target_condition'))

    return FeatureBranchSelection(
        branch=branch,
        control_polarity=control_polarity,
        selected_target_eis=selected_target_eis,
        selected_conditions=selected_conditions,
    )


# =============================================================================
# Feature marker inventory construction
# =============================================================================

def iter_feature_marker_records_for_callable(
        *,
        inventory_path: Path,
        inventory: dict[str, Any],
        entry: dict[str, Any],
) -> list[FeatureMarkerRecord]:
    analysis_info = entry.get('analysis_info', {}) or {}
    branches = analysis_info.get('branches', []) or []

    if not branches:
        return []

    branches_by_id = {
        str(branch.get('id')): branch
        for branch in branches
        if branch.get('id')
    }

    records: list[FeatureMarkerRecord] = []

    for branch in branches:
        decorators = branch.get('decorators', []) or []

        if not decorators:
            continue

        marker_ei = ei_execution_metadata(branch)
        marked_statement = marked_statement_metadata(
            marker_branch=branch,
            branches_by_id=branches_by_id,
            branches=branches,
        )

        for decorator in decorators:
            if not isinstance(decorator, dict):
                continue

            marker_name = decorator.get('name')
            if marker_name not in FEATURE_MARKER_NAMES:
                continue

            kwargs = decorator.get('kwargs', {}) or {}
            if not isinstance(kwargs, dict):
                kwargs = {}

            feature_name = kwargs.get('name')
            if not feature_name:
                continue

            branch_selection = (
                derive_feature_branch_selection(
                    marker_kwargs=kwargs,
                    marked_statement=marked_statement,
                )
                if marker_name == 'FeatureBranch'
                else None
            )

            records.append(
                FeatureMarkerRecord(
                    feature_name=feature_name,
                    marker_name=marker_name,
                    inventory_path=str(inventory_path),
                    unit=inventory.get('unit'),
                    unit_fqn=inventory.get('fully_qualified_name'),
                    callable_id=str(entry.get('id')),
                    callable_name=entry.get('name'),
                    callable_fqn=entry.get('_fqn'),
                    node_id=str(branch.get('id')),
                    kwargs=kwargs,
                    line=branch.get('line'),
                    stmt_type=branch.get('stmt_type'),
                    description=branch.get('description'),
                    condition=branch.get('condition'),
                    marker_ei=marker_ei,
                    marked_statement=marked_statement,
                    branch_selection=branch_selection,
                )
            )

    return records


def iter_feature_marker_records_from_inventory(
        inventory_path: Path,
) -> list[FeatureMarkerRecord]:
    inventory = load_inventory(inventory_path)
    entries = iter_entries(inventory.get('entries', []) or [])

    records: list[FeatureMarkerRecord] = []

    for entry in entries:
        records.extend(
            iter_feature_marker_records_for_callable(
                inventory_path=inventory_path,
                inventory=inventory,
                entry=entry,
            )
        )

    return records


def build_feature_marker_inventory_from_paths(
        inventory_paths: list[Path],
) -> dict[str, FeatureMarkerInventory]:
    inventories: dict[str, FeatureMarkerInventory] = {}

    for inventory_path in inventory_paths:
        for record in iter_feature_marker_records_from_inventory(inventory_path):
            inventory = inventories.setdefault(
                record.feature_name,
                FeatureMarkerInventory(feature_name=record.feature_name),
            )

            match record.marker_name:
                case 'FeatureStart':
                    inventory.starts.append(record)
                case 'FeatureTrace':
                    inventory.traces.append(record)
                case 'FeatureBranch':
                    inventory.branches.append(record)
                case 'FeatureConverge':
                    inventory.converges.append(record)
                case 'FeatureEnd':
                    inventory.ends.append(record)
                case 'FeatureEndConditional':
                    inventory.conditional_ends.append(record)

    return inventories


def feature_marker_inventory_to_dict(
        inventories: dict[str, FeatureMarkerInventory],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []

    for feature_name, inventory in sorted(inventories.items()):
        features.append(
            {
                'feature_name': feature_name,
                'summary': {
                    'starts': len(inventory.starts),
                    'traces': len(inventory.traces),
                    'branches': len(inventory.branches),
                    'converges': len(inventory.converges),
                    'ends': len(inventory.ends),
                    'conditional_ends': len(inventory.conditional_ends),
                    'total_markers': len(inventory.all_records()),
                },
                'markers': {
                    'starts': [asdict(record) for record in inventory.starts],
                    'traces': [asdict(record) for record in inventory.traces],
                    'branches': [asdict(record) for record in inventory.branches],
                    'converges': [asdict(record) for record in inventory.converges],
                    'ends': [asdict(record) for record in inventory.ends],
                    'conditional_ends': [
                        asdict(record)
                        for record in inventory.conditional_ends
                    ],
                },
            }
        )

    return {
        'feature_count': len(features),
        'features': features,
    }


def write_feature_marker_inventory(
        *,
        inventory_root: Path,
        output_path: Path,
) -> dict[str, Any]:
    inventory_paths = find_inventory_files(inventory_root)
    inventories = build_feature_marker_inventory_from_paths(inventory_paths)
    output = feature_marker_inventory_to_dict(inventories)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            width=float('inf'),
        ),
        encoding='utf-8',
    )

    return output


# =============================================================================
# Graph path helpers
# =============================================================================

def load_cfg(cfg_path: Path) -> nx.DiGraph:
    with open(cfg_path, 'rb') as f:
        return pickle.load(f)


def append_path(base: list[str], addition: list[str]) -> list[str]:
    if not base:
        return list(addition)
    if not addition:
        return base
    if base[-1] == addition[0]:
        return [*base, *addition[1:]]
    return [*base, *addition]


def shortest_path_or_none(
        cfg: nx.DiGraph,
        source: str,
        target: str,
) -> list[str] | None:
    if source not in cfg or target not in cfg:
        return None

    try:
        return nx.shortest_path(cfg, source, target)
    except nx.NetworkXNoPath:
        return None


def shortest_path_between_any(
        cfg: nx.DiGraph,
        sources: list[str],
        targets: list[str],
) -> list[str] | None:
    best: list[str] | None = None

    for source in sources:
        for target in targets:
            path = shortest_path_or_none(cfg, source, target)
            if path is None:
                continue
            if best is None or len(path) < len(best):
                best = path

    return best


def trace_feature_path_segment(
        cfg: nx.DiGraph,
        *,
        feature_name: str,
        segment_branch_path: tuple[str, ...],
        start_eis: list[str],
        end_eis: list[str],
        reason: FeaturePathSegmentReason,
) -> FeaturePathSegment | None:
    path = shortest_path_between_any(
        cfg,
        start_eis,
        end_eis,
    )

    if path is None:
        return None

    return FeaturePathSegment(
        feature_name=feature_name,
        segment_branch_path=segment_branch_path,
        start_ei=path[0],
        end_ei=path[-1],
        path=path,
        reason=reason,
    )


def build_path_through_required_targets(
        cfg: nx.DiGraph,
        *,
        start_sources: list[str],
        required_targets: list[str],
        end_targets: list[str],
) -> list[str] | None:
    best: list[str] | None = None

    for required_target in required_targets:
        path_to_required = shortest_path_between_any(
            cfg,
            start_sources,
            [required_target],
        )
        if path_to_required is None:
            continue

        path_to_end = shortest_path_between_any(
            cfg,
            [required_target],
            end_targets,
        )
        if path_to_end is None:
            continue

        candidate = append_path(path_to_required, path_to_end)

        if best is None or len(candidate) < len(best):
            best = candidate

    return best


# =============================================================================
# Marker endpoint helpers
# =============================================================================

def marker_exit_eis(record: FeatureMarkerRecord) -> list[str]:
    if record.marked_statement is not None:
        return [record.marked_statement.entry_ei.ei_id]

    return [record.node_id]


def marker_end_eis(record: FeatureMarkerRecord) -> list[str]:
    if record.marked_statement is None:
        return [record.node_id]

    terminal_eis = [
        item.ei_id
        for item in record.marked_statement.terminal_eis
    ]

    if terminal_eis:
        return terminal_eis

    return [record.marked_statement.entry_ei.ei_id]


# =============================================================================
# Segment/case helpers
# =============================================================================

def initial_feature_flow_case(
        *,
        feature_name: str,
        start_eis: list[str],
) -> FeatureFlowCase:
    return FeatureFlowCase(
        feature_name=feature_name,
        case_branch_path=('main',),
        active_branch_path=('main',),
        current_eis=tuple(start_eis),
        segments=(),
    )


def append_segment_to_case(
        case: FeatureFlowCase,
        segment: FeaturePathSegment,
        *,
        current_eis: list[str],
        active_branch_path: tuple[str, ...] | None = None,
) -> FeatureFlowCase:
    return FeatureFlowCase(
        feature_name=case.feature_name,
        case_branch_path=case.case_branch_path,
        active_branch_path=active_branch_path or case.active_branch_path,
        current_eis=tuple(current_eis),
        segments=(*case.segments, segment),
        status=case.status,
        end_kind=case.end_kind,
        outcome_kind=case.outcome_kind,
    )


def complete_feature_flow_case(
        case: FeatureFlowCase,
        *,
        end_kind: FeatureFlowEndKind,
        outcome_kind: FeatureFlowOutcomeKind,
        current_eis: list[str],
        segment: FeaturePathSegment,
) -> FeatureFlowCase:
    return FeatureFlowCase(
        feature_name=case.feature_name,
        case_branch_path=case.case_branch_path,
        active_branch_path=case.active_branch_path,
        current_eis=tuple(current_eis),
        segments=(*case.segments, segment),
        status=FeatureFlowCaseStatus.COMPLETED,
        end_kind=end_kind,
        outcome_kind=outcome_kind,
    )


def assemble_case_path(case: FeatureFlowCase) -> list[str]:
    path: list[str] = []

    for segment in case.segments:
        path = append_path(path, segment.path)

    return path


# =============================================================================
# Branch point discovery helpers
# =============================================================================

def branch_point_key(record: FeatureMarkerRecord) -> tuple[str, str]:
    if record.marked_statement is not None:
        return (
            record.callable_id,
            record.marked_statement.entry_ei.ei_id,
        )

    return (
        record.callable_id,
        record.node_id,
    )


def branch_point_control_ei_id(record: FeatureMarkerRecord) -> str:
    if record.marked_statement is not None:
        return record.marked_statement.entry_ei.ei_id

    return record.node_id


def build_feature_branch_points(
        feature: FeatureMarkerInventory,
) -> list[FeatureBranchPoint]:
    grouped: dict[tuple[str, str], list[FeatureMarkerRecord]] = {}

    for branch_marker in feature.branches:
        grouped.setdefault(
            branch_point_key(branch_marker),
            [],
        ).append(branch_marker)

    branch_points: list[FeatureBranchPoint] = []

    for index, records in enumerate(grouped.values(), start=1):
        first = records[0]
        control_ei_id = branch_point_control_ei_id(first)

        branch_points.append(
            FeatureBranchPoint(
                feature_name=feature.feature_name,
                branch_point_id=f'{feature.feature_name}::branch_point::{index}',
                marker_node_id=first.node_id,
                control_ei_id=control_ei_id,
                branch_markers=tuple(records),
            )
        )

    return branch_points


# =============================================================================
# Branch case expansion helpers
# =============================================================================

def selected_target_eis_for_branch_marker(
        branch_marker: FeatureMarkerRecord,
) -> list[str]:
    if branch_marker.branch_selection is None:
        return []

    return branch_marker.branch_selection.selected_target_eis


def extend_branch_path(
        branch_path: tuple[str, ...],
        branch_name: str,
) -> tuple[str, ...]:
    if branch_name == 'main':
        return branch_path

    return (*branch_path, branch_name)


def expand_case_at_branch_point(
        case: FeatureFlowCase,
        branch_point: FeatureBranchPoint,
) -> list[FeatureFlowCase]:
    expanded_cases: list[FeatureFlowCase] = []

    for branch_marker in branch_point.branch_markers:
        branch_name = branch_name_for_marker(branch_marker)
        selected_target_eis = selected_target_eis_for_branch_marker(
            branch_marker
        )

        if not selected_target_eis:
            continue

        next_branch_path = extend_branch_path(
            case.case_branch_path,
            branch_name,
        )

        expanded_cases.append(
            FeatureFlowCase(
                feature_name=case.feature_name,
                case_branch_path=next_branch_path,
                active_branch_path=next_branch_path,
                current_eis=tuple(selected_target_eis),
                segments=case.segments,
                status=case.status,
                end_kind=case.end_kind,
                outcome_kind=case.outcome_kind,
            )
        )

    return expanded_cases


def expand_case_at_branch_point_with_segments(
        cfg: nx.DiGraph,
        *,
        case: FeatureFlowCase,
        branch_point: FeatureBranchPoint,
) -> list[FeatureFlowCase]:
    expanded_cases: list[FeatureFlowCase] = []

    for branch_marker in branch_point.branch_markers:
        branch_name = branch_name_for_marker(branch_marker)
        selected_target_eis = selected_target_eis_for_branch_marker(
            branch_marker
        )

        if not selected_target_eis:
            continue

        branch_target_segment = trace_feature_path_segment(
            cfg,
            feature_name=case.feature_name,
            segment_branch_path=case.active_branch_path,
            start_eis=list(case.current_eis),
            end_eis=selected_target_eis,
            reason=FeaturePathSegmentReason.BRANCH_TARGET,
        )

        if branch_target_segment is None:
            continue

        next_branch_path = extend_branch_path(
            case.case_branch_path,
            branch_name,
        )

        expanded_cases.append(
            FeatureFlowCase(
                feature_name=case.feature_name,
                case_branch_path=next_branch_path,
                active_branch_path=next_branch_path,
                current_eis=(branch_target_segment.end_ei,),
                segments=(*case.segments, branch_target_segment),
                status=case.status,
                end_kind=case.end_kind,
                outcome_kind=case.outcome_kind,
            )
        )

    return expanded_cases


def expand_cases_at_branch_point_with_segments(
        cfg: nx.DiGraph,
        *,
        cases: list[FeatureFlowCase],
        branch_point: FeatureBranchPoint,
) -> list[FeatureFlowCase]:
    expanded_cases: list[FeatureFlowCase] = []

    for case in cases:
        expanded_cases.extend(
            expand_case_at_branch_point_with_segments(
                cfg,
                case=case,
                branch_point=branch_point,
            )
        )

    return expanded_cases


def expand_cases_at_branch_point(
        cases: list[FeatureFlowCase],
        branch_point: FeatureBranchPoint,
) -> list[FeatureFlowCase]:
    expanded_cases: list[FeatureFlowCase] = []

    for case in cases:
        expanded_cases.extend(
            expand_case_at_branch_point(
                case,
                branch_point,
            )
        )

    return expanded_cases


# =============================================================================
# Segment-based feature flow tracing
# =============================================================================

def trace_case_to_branch_point(
        cfg: nx.DiGraph,
        *,
        case: FeatureFlowCase,
        branch_point: FeatureBranchPoint,
) -> FeatureFlowCase | None:
    segment = trace_feature_path_segment(
        cfg,
        feature_name=case.feature_name,
        segment_branch_path=case.active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=[branch_point.control_ei_id],
        reason=FeaturePathSegmentReason.START_TO_BRANCH,
    )

    if segment is None:
        return None

    return append_segment_to_case(
        case,
        segment,
        current_eis=[segment.end_ei],
    )


def trace_case_to_end(
        cfg: nx.DiGraph,
        *,
        case: FeatureFlowCase,
        end: FeatureMarkerRecord,
) -> FeatureFlowCase | None:
    end_targets = marker_end_eis(end)

    reason = (
        FeaturePathSegmentReason.TO_CONDITIONAL_END
        if end.marker_name == 'FeatureEndConditional'
        else FeaturePathSegmentReason.TO_END
    )

    end_kind = (
        FeatureFlowEndKind.FEATURE_END_CONDITIONAL
        if end.marker_name == 'FeatureEndConditional'
        else FeatureFlowEndKind.FEATURE_END
    )

    outcome_kind = (
        FeatureFlowOutcomeKind.CONDITIONAL
        if end.marker_name == 'FeatureEndConditional'
        else FeatureFlowOutcomeKind.SUCCESS
    )

    segment = trace_feature_path_segment(
        cfg,
        feature_name=case.feature_name,
        segment_branch_path=case.active_branch_path,
        start_eis=list(case.current_eis),
        end_eis=end_targets,
        reason=reason,
    )

    if segment is None:
        return None

    return complete_feature_flow_case(
        case,
        end_kind=end_kind,
        outcome_kind=outcome_kind,
        current_eis=[segment.end_ei],
        segment=segment,
    )


def trace_feature_flow_cases_to_end(
        cfg: nx.DiGraph,
        *,
        feature: FeatureMarkerInventory,
        start: FeatureMarkerRecord,
        end: FeatureMarkerRecord,
) -> list[FeatureFlowCase]:
    cases = [
        initial_feature_flow_case(
            feature_name=feature.feature_name,
            start_eis=marker_exit_eis(start),
        )
    ]

    branch_points = build_feature_branch_points(feature)

    if not branch_points:
        completed_case = trace_case_to_end(
            cfg,
            case=cases[0],
            end=end,
        )

        return [completed_case] if completed_case is not None else []

    for branch_point in branch_points:
        cases_at_branch_point: list[FeatureFlowCase] = []

        for case in cases:
            case_at_branch_point = trace_case_to_branch_point(
                cfg,
                case=case,
                branch_point=branch_point,
            )

            if case_at_branch_point is not None:
                cases_at_branch_point.append(case_at_branch_point)

        cases = expand_cases_at_branch_point_with_segments(
            cfg,
            cases=cases_at_branch_point,
            branch_point=branch_point,
        )

        if not cases:
            return []

    completed_cases: list[FeatureFlowCase] = []

    for case in cases:
        completed_case = trace_case_to_end(
            cfg,
            case=case,
            end=end,
        )

        if completed_case is not None:
            completed_cases.append(completed_case)

    return completed_cases


def trace_feature_flow_cases(
        cfg: nx.DiGraph,
        inventories: dict[str, FeatureMarkerInventory],
) -> list[FeatureFlowCase]:
    completed_cases: list[FeatureFlowCase] = []

    for feature in inventories.values():
        if not feature.starts:
            continue

        start = feature.starts[0]
        ends = [
            *feature.ends,
            *feature.conditional_ends,
        ]

        for end in ends:
            completed_cases.extend(
                trace_feature_flow_cases_to_end(
                    cfg,
                    feature=feature,
                    start=start,
                    end=end,
                )
            )

    return completed_cases


# =============================================================================
# Feature flow case inspection helpers
# =============================================================================

def summarize_feature_flow_cases(
        inventories: dict[str, FeatureMarkerInventory],
) -> None:
    for feature in inventories.values():
        branch_points = build_feature_branch_points(feature)
        if not branch_points:
            continue

        if not feature.starts:
            continue

        start = feature.starts[0]
        cases = [
            initial_feature_flow_case(
                feature_name=feature.feature_name,
                start_eis=marker_exit_eis(start),
            )
        ]

        for branch_point in branch_points:
            cases = expand_cases_at_branch_point(
                cases,
                branch_point,
            )

        print(f'Feature cases for {feature.feature_name}: {len(cases)}')
        for case in cases:
            print(f'  branch_path={case.case_branch_path} current_eis={case.current_eis}')


# =============================================================================
# Current first-pass branch-forced path export
# =============================================================================

def dedupe_feature_flow_paths(
        flows: list[FeatureFlowPath],
) -> tuple[list[FeatureFlowPath], list[DiscardedFeatureFlowPath]]:
    result: list[FeatureFlowPath] = []
    discarded: list[DiscardedFeatureFlowPath] = []
    seen: dict[tuple[str, ...], FeatureFlowPath] = {}

    for flow in flows:
        key = tuple(
            str(path_item.get('id'))
            for path_item in flow.path
        )

        existing = seen.get(key)
        if existing is not None:
            base_data = {
                field.name: getattr(flow, field.name)
                for field in fields(FeatureFlowPath)
            }

            discarded.append(
                DiscardedFeatureFlowPath(
                    **base_data,
                    discarded_reason='duplicate_path',
                    duplicate_of_variant=existing.variant,
                    duplicate_of_required_branch=existing.required_branch,
                )
            )
            continue

        seen[key] = flow
        result.append(flow)

    return result, discarded


def branch_name_for_marker(branch_marker: FeatureMarkerRecord) -> str:
    if branch_marker.branch_selection is None:
        return 'main'

    return branch_marker.branch_selection.branch or 'main'


def is_explicit_main_branch_marker(branch_marker: FeatureMarkerRecord) -> bool:
    return (
            branch_marker.branch_selection is not None
            and branch_name_for_marker(branch_marker) == 'main'
            and bool(branch_marker.branch_selection.selected_target_eis)
    )


def trace_feature_to_end_path_probe(
        cfg: nx.DiGraph,
        *,
        feature: FeatureMarkerInventory,
        start: FeatureMarkerRecord,
        end: FeatureMarkerRecord,
) -> tuple[list[FeatureFlowPath], list[DiscardedFeatureFlowPath]]:
    flows: list[FeatureFlowPath] = []

    start_sources = marker_exit_eis(start)
    end_targets = marker_end_eis(end)

    explicit_main_branch_marker = next(
        (
            branch_marker
            for branch_marker in feature.branches
            if is_explicit_main_branch_marker(branch_marker)
        ),
        None,
    )

    if explicit_main_branch_marker is not None:
        branch_selection = explicit_main_branch_marker.branch_selection
        assert branch_selection is not None

        main_branch_path = build_path_through_required_targets(
            cfg,
            start_sources=start_sources,
            required_targets=branch_selection.selected_target_eis,
            end_targets=end_targets,
        )

        if main_branch_path is not None:
            flows.append(
                FeatureFlowPath(
                    feature_name=feature.feature_name,
                    end_marker_type=end.marker_name,
                    start_marker=start.node_id,
                    end_marker=end.node_id,
                    variant='main',
                    required_branch_marker=explicit_main_branch_marker.node_id,
                    required_branch='main',
                    required_target_eis=branch_selection.selected_target_eis,
                    path=describe_path(cfg, main_branch_path),
                    path_length=len(main_branch_path),
                    reason='main_branch_target_forced',
                )
            )
    else:
        main_branch_path = shortest_path_between_any(
            cfg,
            start_sources,
            end_targets,
        )

        if main_branch_path is not None:
            flows.append(
                FeatureFlowPath(
                    feature_name=feature.feature_name,
                    end_marker_type=end.marker_name,
                    start_marker=start.node_id,
                    end_marker=end.node_id,
                    variant='main',
                    required_branch_marker=None,
                    required_branch='main',
                    required_target_eis=[],
                    path=describe_path(cfg, main_branch_path),
                    path_length=len(main_branch_path),
                    reason='main_branch_path',
                )
            )

    for branch_marker in feature.branches:
        branch_selection = branch_marker.branch_selection
        if branch_selection is None:
            continue

        branch_name = branch_name_for_marker(branch_marker)

        if branch_name == 'main':
            continue

        required_targets = branch_selection.selected_target_eis
        if not required_targets:
            continue

        branch_path = build_path_through_required_targets(
            cfg,
            start_sources=start_sources,
            required_targets=required_targets,
            end_targets=end_targets,
        )

        if branch_path is None:
            continue

        flows.append(
            FeatureFlowPath(
                feature_name=feature.feature_name,
                end_marker_type=end.marker_name,
                start_marker=start.node_id,
                end_marker=end.node_id,
                variant=branch_name,
                required_branch=branch_name,
                required_branch_marker=branch_marker.node_id,
                required_target_eis=required_targets,
                path=describe_path(cfg, branch_path),
                path_length=len(branch_path),
                reason='branch_target_forced',
            )
        )

    return dedupe_feature_flow_paths(flows)


def trace_feature_flow_path_probes(
        cfg: nx.DiGraph,
        inventories: dict[str, FeatureMarkerInventory],
) -> tuple[list[FeatureFlowPath], list[DiscardedFeatureFlowPath]]:
    flows: list[FeatureFlowPath] = []
    discarded_flows: list[DiscardedFeatureFlowPath] = []

    for feature in inventories.values():
        if not feature.starts:
            continue

        start = feature.starts[0]
        ends = [
            *feature.ends,
            *feature.conditional_ends,
        ]

        for end in ends:
            kept, discarded = trace_feature_to_end_path_probe(
                cfg,
                feature=feature,
                start=start,
                end=end,
            )
            flows.extend(kept)
            discarded_flows.extend(discarded)

    return flows, discarded_flows


# =============================================================================
# Path output formatting
# =============================================================================

def compact_special_node_id(node_text: str, prefix: str) -> str:
    marker = f'{prefix}::'
    if node_text.startswith(marker):
        return node_text[len(marker):]
    return node_text


def describe_path_node(cfg: nx.DiGraph, node_id: str) -> dict[str, Any]:
    node_text = str(node_id)
    node_data = cfg.nodes.get(node_id, {})
    category = node_data.get('category')

    if node_text.startswith('collapsed::'):
        callable_name = node_data.get('callable_name')
        description = node_data.get('description') or 'collapsed call'

        label_parts = [
            item
            for item in [
                node_data.get('unit'),
                callable_name,
                description,
            ]
            if item
        ]

        return {
            'id': compact_special_node_id(node_text, 'collapsed'),
            'kind': 'collapsed',
            'label': '::'.join(label_parts) if label_parts else node_text,
        }

    if node_text.startswith('placeholder::'):
        operation_target = node_data.get('operation_target')
        description = node_data.get('description')

        label_parts = [
            item
            for item in [
                description,
                operation_target,
            ]
            if item
        ]

        return {
            'id': compact_special_node_id(node_text, 'placeholder'),
            'kind': 'placeholder',
            'label': '::'.join(label_parts) if label_parts else node_text,
        }

    if node_text.startswith('external::'):
        external_type = node_data.get('type')
        operation_target = node_data.get('operation_target')
        description = node_data.get('description')

        label_parts = [
            item
            for item in [
                'external',
                external_type,
                operation_target,
                description,
            ]
            if item
        ]

        return {
            'id': compact_special_node_id(node_text, 'external'),
            'kind': 'external',
            'label': '::'.join(label_parts) if label_parts else node_text,
        }

    description = (
            node_data.get('description')
            or node_data.get('condition')
            or node_data.get('stmt_type')
    )

    label_parts = [
        item
        for item in [
            node_data.get('unit'),
            node_data.get('callable_name'),
            description,
        ]
        if item
    ]

    result = {
        'id': node_text,
        'label': '::'.join(label_parts) if label_parts else node_text,
    }

    if category and category != 'execution_instance':
        result['kind'] = category

    return result


def feature_flow_case_to_dict(
        cfg: nx.DiGraph,
        case: FeatureFlowCase,
) -> dict[str, Any]:
    assembled_path = assemble_case_path(case)

    return {
        'feature_name': case.feature_name,
        'case_branch_path': list(case.case_branch_path),
        'active_branch_path': list(case.active_branch_path),
        'status': case.status.value,
        'end_kind': case.end_kind.value if case.end_kind is not None else None,
        'outcome_kind': (
            case.outcome_kind.value
            if case.outcome_kind is not None
            else None
        ),
        'segment_count': len(case.segments),
        'path_length': len(assembled_path),
        'segments': [
            {
                'segment_branch_path': list(segment.segment_branch_path),
                'start_ei': segment.start_ei,
                'end_ei': segment.end_ei,
                'reason': segment.reason.value,
                'path_length': len(segment.path),
                'path': describe_path(cfg, segment.path),
            }
            for segment in case.segments
        ],
        'path': describe_path(cfg, assembled_path),
    }


def describe_path(cfg: nx.DiGraph, path: list[str]) -> list[dict[str, Any]]:
    return [
        describe_path_node(cfg, node_id)
        for node_id in path
    ]


# =============================================================================
# YAML output writers
# =============================================================================


def feature_branch_points_to_dict(
        inventories: dict[str, FeatureMarkerInventory],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []

    for feature_name, feature in sorted(inventories.items()):
        branch_points = build_feature_branch_points(feature)

        features.append(
            {
                'feature_name': feature_name,
                'branch_point_count': len(branch_points),
                'branch_points': [
                    {
                        'branch_point_id': branch_point.branch_point_id,
                        'marker_node_id': branch_point.marker_node_id,
                        'control_ei_id': branch_point.control_ei_id,
                        'branches': [
                            {
                                'branch': branch_name_for_marker(marker),
                                'marker_node_id': marker.node_id,
                                'control_polarity': (
                                    marker.branch_selection.control_polarity
                                    if marker.branch_selection is not None
                                    else None
                                ),
                                'selected_target_eis': (
                                    marker.branch_selection.selected_target_eis
                                    if marker.branch_selection is not None
                                    else []
                                ),
                                'selected_conditions': (
                                    marker.branch_selection.selected_conditions
                                    if marker.branch_selection is not None
                                    else []
                                ),
                            }
                            for marker in branch_point.branch_markers
                        ],
                    }
                    for branch_point in branch_points
                ],
            }
        )

    return {
        'feature_count': len(features),
        'features': features,
    }


def write_feature_branch_points(
        *,
        inventories: dict[str, FeatureMarkerInventory],
        output_path: Path,
) -> dict[str, Any]:
    output = feature_branch_points_to_dict(inventories)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            width=float('inf'),
        ),
        encoding='utf-8',
    )

    return output


def feature_flow_cases_to_dict(
        cfg: nx.DiGraph,
        cases: list[FeatureFlowCase],
) -> dict[str, Any]:
    return {
        'case_count': len(cases),
        'cases': [
            feature_flow_case_to_dict(cfg, case)
            for case in cases
        ],
    }


def write_feature_flow_cases(
        *,
        cfg_path: Path,
        inventories: dict[str, FeatureMarkerInventory],
        output_path: Path,
) -> dict[str, Any]:
    cfg = load_cfg(cfg_path)
    cases = trace_feature_flow_cases(cfg, inventories)
    output = feature_flow_cases_to_dict(cfg, cases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            width=float('inf'),
        ),
        encoding='utf-8',
    )

    return output


def feature_flow_paths_to_dict(
        flows: list[FeatureFlowPath],
        discarded_flows: list[DiscardedFeatureFlowPath],
) -> dict[str, Any]:
    return {
        'flow_count': len(flows),
        'discarded_flow_count': len(discarded_flows),
        'flows': [
            asdict(flow)
            for flow in flows
        ],
        'discarded_flows': [
            asdict(flow)
            for flow in discarded_flows
        ],
    }


def write_feature_flow_paths(
        *,
        cfg_path: Path,
        inventories: dict[str, FeatureMarkerInventory],
        output_path: Path,
) -> dict[str, Any]:
    cfg = load_cfg(cfg_path)
    flows, discarded_flows = trace_feature_flow_path_probes(cfg, inventories)
    output = feature_flow_paths_to_dict(flows, discarded_flows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            width=float('inf'),
        ),
        encoding='utf-8',
    )

    return output


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Inspect feature-flow markers from PyBastion inventory YAML files.'
    )
    parser.add_argument(
        '--inventory-root',
        type=Path,
        required=True,
        help='Root directory containing *.inventory.yaml files.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output YAML path.',
    )
    parser.add_argument(
        '--cfg',
        type=Path,
        help='Optional stage1 EI CFG pickle for feature flow path probing.',
    )
    parser.add_argument(
        '--flow-output',
        type=Path,
        help='Optional feature flow path probe YAML output path.',
    )
    parser.add_argument(
        '--branch-points-output',
        type=Path,
        help='Optional feature branch point topology YAML output path.',
    )
    parser.add_argument(
        '--summarize-flow-cases',
        action='store_true',
        help='Print a console summary of expanded feature flow cases.',
    )
    parser.add_argument(
        '--case-output',
        type=Path,
        help='Optional segment-assembled feature flow case YAML output path.',
    )

    args = parser.parse_args()

    inventory_paths = find_inventory_files(args.inventory_root)
    inventories = build_feature_marker_inventory_from_paths(inventory_paths)

    output = feature_marker_inventory_to_dict(inventories)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.dump(
            output,
            sort_keys=False,
            allow_unicode=True,
            width=float('inf'),
        ),
        encoding='utf-8',
    )

    marker_count = sum(
        feature['summary']['total_markers']
        for feature in output['features']
    )

    print(f'Feature marker inventory written to {args.output}')
    print(f'Features: {output["feature_count"]}')
    print(f'Markers: {marker_count}')

    if args.branch_points_output:
        branch_points_output = write_feature_branch_points(
            inventories=inventories,
            output_path=args.branch_points_output,
        )

        branch_point_count = sum(
            feature['branch_point_count']
            for feature in branch_points_output['features']
        )

        print(f'Feature branch points written to {args.branch_points_output}')
        print(f'Branch points: {branch_point_count}')

    if args.summarize_flow_cases:
        summarize_feature_flow_cases(inventories)

    if args.cfg and args.case_output:
        case_output = write_feature_flow_cases(
            cfg_path=args.cfg,
            inventories=inventories,
            output_path=args.case_output,
        )

        print(f'Feature flow cases written to {args.case_output}')
        print(f'Feature flow cases: {case_output["case_count"]}')

    if args.cfg and args.flow_output:
        flow_output = write_feature_flow_paths(
            cfg_path=args.cfg,
            inventories=inventories,
            output_path=args.flow_output,
        )

        print(f'Feature flow paths written to {args.flow_output}')
        print(f'Feature flow paths: {flow_output["flow_count"]}')
        print(f'Discarded feature flow paths: {flow_output["discarded_flow_count"]}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
