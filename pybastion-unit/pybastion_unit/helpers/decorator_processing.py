#!/usr/bin/env python3
"""
Decorator Processing - Schema, Parsing, Validation, and Co-occurrence Checking

Handles all analysis decorator concerns:
- Schema definition (COMMON_FIELDS, ANALYSIS_DECORATORS)
- Comment-based decorator parsing
- Field validation against schema
- Codebase-wide co-occurrence validation for feature decorators
"""

from __future__ import annotations

import ast
from typing import Any

# =============================================================================
# Schema
# =============================================================================

COMMON_FIELDS: dict[str, dict[str, Any]] = {
    'name': {
        'required': True,
        'strict': False,
        'value_cardinality': '1..1',
    },
    'comment': {
        'required': False,
        'strict': False,
        'value_cardinality': '1..1',
    },
}

ANALYSIS_DECORATORS: dict[str, Any] = {
    'Operation': {
        'MechanicalOperation': {
            'effect': 'exclude_from_flow',
            'scope': 'immediate',
            'common_fields': ['comment'],
            'fields': {
                'type': {
                    'required': True,
                    'strict': True,
                    'value_cardinality': '1..1',
                    'values': [
                        'serialization',
                        'deserialization',
                        'validation',
                        'formatting',
                        'conversion',
                        'normalization',
                        'data_transform',
                        'presentation',
                        'construction',
                    ],
                },
                'alias': {
                    'required': False,
                    'strict': False,
                    'value_cardinality': '1..1',
                },
            },
        },
        'UtilityOperation': {
            'effect': 'exclude_from_flow',
            'scope': 'immediate',
            'common_fields': ['comment'],
            'fields': {
                'type': {
                    'required': True,
                    'strict': True,
                    'value_cardinality': '1..1',
                    'values': [
                        'logging',
                        'caching',
                        'config',
                        'observability',
                        'audit',
                        'data_structure',
                        'registry',
                    ],
                },
                'alias': {
                    'required': False,
                    'strict': False,
                    'value_cardinality': '1..1',
                },
            },
        },
    },
    'Feature': {
        'FeatureStart': {
            'effect': 'mark_flow_boundary',
            'scope': 'codebase',
            'common_fields': ['name', 'comment'],
            'fields': {
                'variants': {
                    'required': False,
                    'strict': False,
                    'value_cardinality': '1..N',
                    'exclusive': False,
                },
            },
            'co_occurrence': {
                'required': {
                    'one_of': {
                        'FeatureEnd': {
                            'correlation': {'field': 'name', 'operator': 'eq'},
                            'repeatable': True,
                        },
                        'FeatureEndConditional': {
                            'correlation': {'field': 'name', 'operator': 'eq'},
                            'repeatable': True,
                        },
                    },
                },
                'optional': {
                    'FeatureTrace': {
                        'correlation': {'field': 'name', 'operator': 'eq'},
                        'repeatable': True,
                    },
                    'FeatureBranch': {
                        'correlation': {'field': 'name', 'operator': 'eq'},
                        'repeatable': True,
                    },
                    'FeatureConverge': {
                        'correlation': {'field': 'name', 'operator': 'eq'},
                        'repeatable': True,
                    },
                    'FeatureEndConditional': {
                        'correlation': {'field': 'name', 'operator': 'eq'},
                        'repeatable': True,
                    },
                },
            },
        },
        'FeatureTrace': {
            'effect': 'mark_flow_waypoint',
            'scope': 'codebase',
            'common_fields': ['name', 'comment'],
            'fields': {
                'branch_name': {
                    'required': False,
                    'strict': False,
                    'value_cardinality': '1..1',
                },
            },
        },
        'FeatureBranch': {
            'effect': 'mark_flow_branch',
            'scope': 'codebase',
            'common_fields': ['name', 'comment'],
            'fields': {
                'branches': {
                    'required': True,
                    'strict': False,
                    'value_cardinality': '1..N',
                },
            },
        },
        'FeatureConverge': {
            'effect': 'mark_flow_convergence',
            'scope': 'codebase',
            'common_fields': ['name', 'comment'],
            'fields': {
                'branches': {
                    'required': True,
                    'strict': False,
                    'value_cardinality': '1..N',
                },
                'into': {
                    'required': False,
                    'strict': False,
                    'value_cardinality': '1..1',
                },
            },
        },
        'FeatureEndConditional': {
            'effect': 'mark_flow_boundary',
            'scope': 'codebase',
            'common_fields': ['name', 'comment'],
            'fields': {
                'on_condition': {
                    'required': True,
                    'strict': False,
                    'value_cardinality': '1..1',
                },
                'branch_name': {
                    'required': False,
                    'strict': False,
                    'value_cardinality': '1..1',
                },
            },
        },
        'FeatureEnd': {
            'effect': 'mark_flow_boundary',
            'scope': 'codebase',
            'common_fields': ['name', 'comment'],
            'fields': {
                'branch_name': {
                    'required': False,
                    'strict': False,
                    'value_cardinality': '1..1',
                },
            },
        },
    },
}


# =============================================================================
# Schema Resolution
# =============================================================================

def get_decorator_schema(decorator_name: str) -> dict[str, Any] | None:
    """
    Look up the schema for a decorator by name, resolving common_fields.

    Returns the full schema dict with common_fields merged into fields,
    or None if the decorator name is not known.
    """
    for group in ANALYSIS_DECORATORS.values():
        if decorator_name in group:
            schema = group[decorator_name]
            return _resolve_common_fields(schema)
    return None


def get_decorator_effect(decorator_name: str) -> str | None:
    """
    Return the effect string for a decorator name.

    Used by pipeline stages to act on decorators without hardcoding names.
    Returns None if decorator is not known.
    """
    schema = get_decorator_schema(decorator_name)
    if schema:
        return schema.get('effect')
    return None


def get_decorator_scope(decorator_name: str) -> str | None:
    """
    Return the scope ('immediate' or 'codebase') for a decorator name.

    Returns None if decorator is not known.
    """
    schema = get_decorator_schema(decorator_name)
    if schema:
        return schema.get('scope')
    return None


def is_known_decorator(decorator_name: str) -> bool:
    """Return True if the decorator name is in the schema."""
    return get_decorator_schema(decorator_name) is not None


def _resolve_common_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of the schema with common_fields merged into fields.
    """
    resolved = {k: v for k, v in schema.items() if k != 'common_fields'}
    common_keys = schema.get('common_fields', [])
    if common_keys:
        merged_fields = {}
        for key in common_keys:
            if key in COMMON_FIELDS:
                merged_fields[key] = COMMON_FIELDS[key]
        merged_fields.update(schema.get('fields', {}))
        resolved['fields'] = merged_fields
    return resolved


# =============================================================================
# Parsing
# =============================================================================

def extract_statement_decorators(
        stmt: ast.stmt,
        source_lines: list[str],
) -> list[dict[str, Any]]:
    """
    Extract analysis decorators from comments preceding a statement.

    Scans upward from the statement, collecting all # :: patterns
    until hitting a blank line or code.
    """
    decorators: list[dict[str, Any]] = []

    if stmt.lineno > 1:
        # Scan upward from the line before the statement
        for line_idx in range(stmt.lineno - 2, -1, -1):
            line = source_lines[line_idx].strip()
            if line.startswith('# :: '):
                decorator = parse_decorator_comment(line)
                if decorator:
                    decorators.append(decorator)
            elif line.startswith('#'):
                continue  # Regular comment, keep scanning
            elif not line:
                break  # Blank line, stop
            else:
                break  # Code line, stop

    return decorators


def extract_callable_decorators(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
) -> list[dict[str, Any]]:
    """
    Extract all analysis decorators from comments preceding a function node,
    and from the function's docstring if present.

    Returns a list of parsed decorator dicts with 'name' and 'kwargs'.
    """
    decorators: list[dict[str, Any]] = []

    # Scan upward from the line before the function
    if node.lineno > 1:
        for line_idx in range(node.lineno - 2, -1, -1):
            line = source_lines[line_idx].strip()
            if line.startswith('# :: '):
                decorator = parse_decorator_comment(line)
                if decorator:
                    decorators.append(decorator)
            elif line.startswith('@'):
                continue  # skip Python decorators, keep scanning
            else:
                break  # blank line or code — stop

    if not node.body:
        return decorators
    first_stmt = node.body[0]
    if not isinstance(first_stmt, ast.Expr):
        return decorators
    expr_value = first_stmt.value
    if not isinstance(expr_value, ast.Constant):
        return decorators
    if not isinstance(expr_value.value, str):
        return decorators

    # Check docstring
    docstring = expr_value.value
    for line in docstring.split('\n'):
        line = line.strip()
        if line.startswith(':: '):
            decorator = parse_decorator_comment(line)
            if decorator:
                decorators.append(decorator)

    return decorators


def parse_decorator_comment(line: str) -> dict[str, Any] | None:
    """
    Parse a single comment line into a decorator dict.

    Expected format: # :: DecoratorName | field=value | field=value
    Also handles docstring lines (no leading #).

    Returns dict with 'name' and 'kwargs', or None if line is not a decorator.
    """
    if '::' not in line:
        return None

    # Strip comment marker
    line = line.lstrip('#').strip()
    if not line.startswith('::'):
        return None

    # Strip :: marker
    line = line[2:].strip()

    parts = [p.strip() for p in line.split('|')]
    if not parts:
        return None

    decorator_name = parts[0].strip()
    if not decorator_name:
        return None

    kwargs: dict[str, str] = {}
    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            kwargs[key.strip()] = value.strip().strip('"').strip("'")

    return {
        'name': decorator_name,
        'kwargs': kwargs,
    }


# =============================================================================
# Field Validation
# =============================================================================

class DecoratorValidationError:
    """A single validation error on a decorator instance."""

    def __init__(self, decorator_name: str, message: str, field: str | None = None) -> None:
        self.decorator_name = decorator_name
        self.field = field
        self.message = message

    def __repr__(self) -> str:
        if self.field:
            return f"DecoratorValidationError({self.decorator_name}.{self.field}: {self.message})"
        return f"DecoratorValidationError({self.decorator_name}: {self.message})"


def validate_decorator(decorator: dict[str, Any]) -> list[DecoratorValidationError]:
    """
    Validate a parsed decorator dict against its schema.

    Checks:
    - Decorator name is known
    - All required fields are present
    - No unknown fields are present
    - Strict fields have values from the known values list
    - Value cardinality is respected for multi-value fields

    Returns a list of validation errors (empty if valid).
    """
    errors: list[DecoratorValidationError] = []
    name = decorator.get('name', '')
    kwargs = decorator.get('kwargs', {})

    schema = get_decorator_schema(name)
    if schema is None:
        errors.append(DecoratorValidationError(name, 'Unknown decorator name'))
        return errors

    fields = schema.get('fields', {})

    # Check required fields
    for field_name, field_spec in fields.items():
        if field_spec.get('required', False) and field_name not in kwargs:
            errors.append(DecoratorValidationError(name, f"Required field '{field_name}' is missing", field_name))

    # Check for unknown fields
    for field_name in kwargs:
        if field_name not in fields:
            errors.append(DecoratorValidationError(name, f"Unknown field '{field_name}'", field_name))

    # Check strict fields and value cardinality
    for field_name, value in kwargs.items():
        if field_name not in fields:
            continue

        field_spec = fields[field_name]

        # Strict value checking
        if field_spec.get('strict', False):
            valid_values = field_spec.get('values', [])
            if value not in valid_values:
                errors.append(
                    DecoratorValidationError(
                        name,
                        f"Field '{field_name}' has invalid value '{value}'. Must be one of: {valid_values}",
                        field_name
                    )
                )

        # Value cardinality checking for multi-value fields
        cardinality = field_spec.get('value_cardinality', '1..1')
        if cardinality not in ('1..1', '0..1'):
            # Multi-value field — parse comma-separated values
            values = [v.strip() for v in value.split(',') if v.strip()]
            min_count = int(cardinality.split('..')[0])
            if len(values) < min_count:
                errors.append(
                    DecoratorValidationError(
                        name,
                        f"Field '{field_name}' requires at least {min_count} value(s), got {len(values)}",
                        field_name
                    )
                )

    return errors


# =============================================================================
# Co-occurrence Validation
# =============================================================================

class CoOccurrenceValidationError:
    """A co-occurrence validation error for a feature flow."""

    def __init__(self, feature_name: str, message: str) -> None:
        self.feature_name = feature_name
        self.message = message

    def __repr__(self) -> str:
        return f"CoOccurrenceValidationError(feature={self.feature_name!r}: {self.message})"


def validate_feature_co_occurrences(
        inventory_entries: list[dict[str, Any]],
) -> list[CoOccurrenceValidationError]:
    """
    Validate codebase-wide co-occurrence rules for feature decorators.

    Collects all feature decorators from inventory entries, groups them by
    feature name, then checks co-occurrence rules defined on FeatureStart.

    Returns a list of co-occurrence errors (empty if valid).
    """
    errors: list[CoOccurrenceValidationError] = []

    # Collect all feature decorators grouped by feature name
    # by_flow: flow_name -> decorator_type -> list of decorator instances
    by_flow: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def collect(entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            for dec in entry.get('decorators', []):
                dec_name = dec.get('name', '')
                if get_decorator_scope(dec_name) == 'codebase':
                    feature_flow_name = dec.get('kwargs', {}).get('name')
                    if feature_flow_name:
                        if feature_flow_name not in by_flow:
                            by_flow[feature_flow_name] = {}
                        if dec_name not in by_flow[feature_flow_name]:
                            by_flow[feature_flow_name][dec_name] = []
                        by_flow[feature_flow_name][dec_name].append(dec)
            if 'children' in entry:
                collect(entry['children'])

    collect(inventory_entries)

    feature_schema = get_decorator_schema('FeatureStart')
    if not feature_schema:
        return errors

    co_occurrence = feature_schema.get('co_occurrence', {})
    required = co_occurrence.get('required', {})

    for flow_name, dec_types in by_flow.items():
        if 'FeatureStart' not in dec_types:
            continue

        # Check FeatureStart only appears once
        if len(dec_types['FeatureStart']) > 1:
            errors.append(
                CoOccurrenceValidationError(
                    flow_name,
                    f"FeatureStart appears {len(dec_types['FeatureStart'])} times — must appear exactly once"
                )
            )

        # Check required one_of
        one_of = required.get('one_of', {})
        if one_of:
            satisfied = any(dec_type in dec_types for dec_type in one_of)
            if not satisfied:
                errors.append(
                    CoOccurrenceValidationError(
                        flow_name,
                        f"FeatureStart requires at least one of: {list(one_of.keys())}"
                    )
                )

        # Check non-repeatable co-occurrences
        all_co_occurrence = {
            **required.get('one_of', {}),
            **co_occurrence.get('optional', {}),
        }
        for dec_type, rules in all_co_occurrence.items():
            if not rules.get('repeatable', True):
                count = len(dec_types.get(dec_type, []))
                if count > 1:
                    errors.append(
                        CoOccurrenceValidationError(
                            flow_name,
                            f"{dec_type} is not repeatable for flow '{flow_name}' but appears {count} times"
                        )
                    )

    return errors


# =============================================================================
# Convenience
# =============================================================================

def has_effect(decorator: dict[str, Any], effect: str) -> bool:
    """
    Return True if the decorator has the given effect.

    Replaces hardcoded decorator name checks in pipeline stages.

    Example:
        if has_effect(decorator, 'exclude_from_flow'):
            ...
    """
    return get_decorator_effect(decorator.get('name', '')) == effect


def collect_decorators_by_effect(
        decorators: list[dict[str, Any]],
        effect: str,
) -> list[dict[str, Any]]:
    """Return all decorators from the list that have the given effect."""
    return [d for d in decorators if has_effect(d, effect)]
