#!/usr/bin/env python3
"""
Shared data models for the unit ledger analysis pipeline.

These models define contracts between pipeline stages:
- enumerate_callables.py (Stage 1)
- enumerate_exec_items.py (Stage 2)  
- CFG path enumeration (Stage 3)
- Ledger transformation (Stage 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from callable_id_generation import ei_id_to_integration_id
from knowledge_base import (
    BOUNDARY_OPERATIONS,
    BUILTIN_METHODS,
    COMMON_EXTLIB_MODULES,
    PYTHON_BUILTINS,
    STDLIB_CLASSES,
    is_stdlib_module,
)


# =============================================================================
# Common Types
# =============================================================================

@dataclass
class TypeRef:
    """
    Type reference with optional generic arguments.

    Examples:
        int -> TypeRef(name='int')
        list[str] -> TypeRef(name='list', args=[TypeRef(name='str')])
        dict[str, Any] -> TypeRef(name='dict', args=[TypeRef(name='str'), TypeRef(name='Any')])
    """
    name: str
    args: list[TypeRef] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TypeRef | None:
        """Parse from inventory dict format."""
        if not data:
            return None
        return cls(
            name=data['name'],
            args=[cls.from_dict(arg) for arg in data.get('args', []) if arg is not None]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {'name': self.name}
        if self.args:
            result['args'] = [arg.to_dict() for arg in self.args]
        return result


@dataclass
class ParamSpec:
    """Parameter specification."""
    name: str
    type: TypeRef | None = None
    default: str | None = None  # Store as string representation

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParamSpec:
        """Parse from inventory dict format."""
        return cls(
            name=data['name'],
            type=TypeRef.from_dict(data['type']) if 'type' in data else None,
            default=data.get('default')
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {'name': self.name}
        if self.type:
            result['type'] = self.type.to_dict()
        if self.default is not None:
            result['default'] = self.default
        return result


# =============================================================================
# Execution Items (Branches)
# =============================================================================

@dataclass
class Branch:
    """
    Execution Item (EI) representation.

    Contract between enumerate_exec_items.py and downstream stages.
    Called "Branch" in current code but represents an Execution Item.
    """
    id: str
    line: int
    condition: str
    outcome: str

    def __post_init__(self) -> None:
        """Validate branch structure."""
        if not self.id:
            raise ValueError("Branch ID cannot be empty")
        if self.line <= 0:
            raise ValueError(f"Invalid line number: {self.line}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Branch:
        """Parse from inventory dict format."""
        return cls(
            id=data['id'],
            line=data['line'],
            condition=data['condition'],
            outcome=data['outcome']
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        return {
            'id': self.id,
            'line': self.line,
            'condition': self.condition,
            'outcome': self.outcome
        }

    def to_ledger_ei_spec(self) -> dict[str, str]:
        """
        Transform to ledger EiSpec format.

        Ledger format uses the same structure but may add additional fields.
        """
        return {
            'id': self.id,
            'condition': self.condition,
            'outcome': self.outcome
        }


# =============================================================================
# Integration Points
# =============================================================================

class IntegrationType(str, Enum):
    """Type of integration point."""
    CALL = 'call'
    CONSTRUCT = 'construct'
    IMPORT = 'import'
    DISPATCH = 'dispatch'
    IO = 'io'
    OTHER = 'other'


class IntegrationCategory(str, Enum):
    """Category of integration after classification."""
    INTERUNIT = 'interunit'
    STDLIB = 'stdlib'
    EXTLIB = 'extlib'
    BOUNDARY = 'boundaries'
    UNKNOWN = 'unknown'


@dataclass
class IntegrationCandidate:
    """
    Integration point before categorization.

    Contract between enumerate_callables.py (Stage 1) and CFG enumeration (Stage 3).
    Stage 3 populates the execution_paths field.
    """
    type: str  # IntegrationType value
    target: str
    line: int
    signature: str
    execution_paths: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate integration candidate."""
        if not self.target:
            raise ValueError("Integration target cannot be empty")
        if self.line <= 0:
            raise ValueError(f"Invalid line number: {self.line}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationCandidate:
        """Parse from inventory dict format."""
        return cls(
            type=data['type'],
            target=data['target'],
            line=data['line'],
            signature=data['signature'],
            execution_paths=data.get('execution_paths', [])
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {
            'type': self.type,
            'target': self.target,
            'line': self.line,
            'signature': self.signature
        }
        if self.execution_paths:
            result['execution_paths'] = self.execution_paths
        return result

    def to_ledger_integration_fact(self) -> dict[str, Any]:
        """
        Transform to ledger IntegrationFact format.

        Returns:
            dict in ledger IntegrationFact format
        """
        # Generate integration ID from the last EI in the first path
        integration_id: str | None = None
        if self.execution_paths and self.execution_paths[0]:
            last_ei = self.execution_paths[0][-1]
            integration_id = ei_id_to_integration_id(last_ei)

        fact: dict[str, Any] = {}

        # ID must be first (if present)
        if integration_id:
            fact['id'] = integration_id

        # Then required fields
        fact['target'] = self.target
        fact['kind'] = self.type
        fact['signature'] = self.signature
        fact['execution_paths'] = self.execution_paths

        return fact


# =============================================================================
# Callable Entries
# =============================================================================

@dataclass
class CallableEntry:
    """
    Entry for any code element (unit, class, enum, function, method).

    Central data structure that flows through all pipeline stages.
    Can represent both callable entries (functions/methods) and
    non-callable entries (classes/enums) via the 'kind' field.
    """
    id: str
    kind: str  # 'unit', 'class', 'enum', 'function', 'method'
    name: str
    line_start: int
    line_end: int
    signature: str | None = None
    visibility: str | None = None  # 'public', 'protected', 'private'
    decorators: list[dict[str, Any]] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)  # ['async', 'static', etc.]
    base_classes: list[str] = field(default_factory=list)  # For classes/enums
    children: list[CallableEntry] = field(default_factory=list)  # Nested classes/methods
    params: list[ParamSpec] = field(default_factory=list)
    return_type: TypeRef | None = None
    branches: list[Branch] = field(default_factory=list)
    integration_candidates: list[IntegrationCandidate] = field(default_factory=list)
    total_eis: int = 0
    needs_callable_analysis: bool = False  # True for functions/methods

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallableEntry:
        """
        Parse from inventory dict format.

        Handles both pre-EI and post-EI merge states.
        """
        # Extract params from ast_analysis if present
        params: list[ParamSpec] = []
        if 'params' in data:
            params = [ParamSpec.from_dict(p) for p in data['params']]
        elif 'ast_analysis' in data and 'params' in data['ast_analysis']:
            params = [ParamSpec.from_dict(p) for p in data['ast_analysis']['params']]

        # Extract return type
        return_type: TypeRef | None = None
        if 'return_type' in data:
            return_type = TypeRef.from_dict(data['return_type'])
        elif 'ast_analysis' in data and 'return_type' in data['ast_analysis']:
            return_type = TypeRef.from_dict(data['ast_analysis']['return_type'])

        # Extract branches (EIs)
        branches = [Branch.from_dict(b) for b in data.get('branches', [])]

        # Extract integration candidates
        integration_candidates: list[IntegrationCandidate] = []
        if 'ast_analysis' in data and 'integration_candidates' in data['ast_analysis']:
            integration_candidates = [
                IntegrationCandidate.from_dict(ic)
                for ic in data['ast_analysis']['integration_candidates']
            ]

        # Extract children (recursive)
        children = [cls.from_dict(c) for c in data.get('children', [])]

        return cls(
            id=data['id'],
            kind=data['kind'],
            name=data['name'],
            line_start=data.get('line_start', 0),
            line_end=data.get('line_end', 0),
            signature=data.get('signature'),
            visibility=data.get('visibility'),
            decorators=data.get('decorators', []),
            modifiers=data.get('modifiers', []),
            base_classes=data.get('base_classes', []),
            children=children,
            params=params,
            return_type=return_type,
            branches=branches,
            integration_candidates=integration_candidates,
            total_eis=data.get('total_eis', len(branches)),
            needs_callable_analysis=data.get('needs_callable_analysis', False)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to inventory dict format."""
        result: dict[str, Any] = {
            'id': self.id,
            'kind': self.kind,
            'name': self.name,
            'line_start': self.line_start,
            'line_end': self.line_end
        }

        if self.signature:
            result['signature'] = self.signature

        if self.visibility:
            result['visibility'] = self.visibility

        if self.decorators:
            result['decorators'] = self.decorators

        if self.modifiers:
            result['modifiers'] = self.modifiers

        if self.base_classes:
            result['base_classes'] = self.base_classes

        if self.children:
            result['children'] = [c.to_dict() for c in self.children]

        if self.needs_callable_analysis:
            result['needs_callable_analysis'] = self.needs_callable_analysis

        if self.params:
            result['params'] = [p.to_dict() for p in self.params]

        if self.return_type:
            result['return_type'] = self.return_type.to_dict()

        if self.branches:
            result['branches'] = [b.to_dict() for b in self.branches]
            result['total_eis'] = len(self.branches)

        if self.integration_candidates:
            result['ast_analysis'] = {
                'integration_candidates': [
                    ic.to_dict() for ic in self.integration_candidates
                ]
            }

        return result

    def categorize_integrations(self, project_types: set[str], known_types: dict[str, str] | None = None) -> dict[
        str, list[dict[str, Any]]]:
        """
        Categorize integration candidates into interunit/extlib/boundaries/unknown.

        Args:
            project_types: set of FQNs from project inventory
            known_types: optional dict mapping variable names to their types

        Returns:
            dict mapping category string to a list of IntegrationFact dicts
        """
        if known_types is None:
            known_types = {}

        categorized: dict[IntegrationCategory, list[dict[str, Any]]] = {
            IntegrationCategory.INTERUNIT: [],
            IntegrationCategory.STDLIB: [],
            IntegrationCategory.EXTLIB: [],
            IntegrationCategory.BOUNDARY: [],
            IntegrationCategory.UNKNOWN: []
        }

        for candidate in self.integration_candidates:
            # Check if this is actually an integration (not a non-integration)
            if self._is_non_integration(candidate.target):
                continue  # Skip non-integrations entirely

            category = self._determine_category(candidate.target, project_types, known_types)
            fact = candidate.to_ledger_integration_fact()

            # Add boundary details if it's a boundary integration
            if category == IntegrationCategory.BOUNDARY:
                boundary_info = self._get_boundary_info(candidate.target)
                if boundary_info:
                    fact['boundary'] = {
                        'kind': boundary_info['kind']
                    }
                    if 'operation' in boundary_info:
                        fact['boundary']['operation'] = boundary_info['operation']
                    if 'protocol' in boundary_info:
                        fact['boundary']['protocol'] = boundary_info['protocol']

            categorized[category].append(fact)

        # Remove empty categories and convert enum keys to strings
        return {
            cat.value: facts
            for cat, facts in categorized.items()
            if facts
        }

    def _determine_category(self, target: str, project_types: set[str],
                            known_types: dict[str, str]) -> IntegrationCategory:
        """
        Determine integration category.

        Priority order:
        1. Boundary operations (crosses system boundary)
        2. Project types (interunit)
        3. Standard library (stdlib)
        4. Known third-party libraries (extlib)
        5. Unknown (can't determine)
        """

        # 1. BOUNDARY OPERATIONS - highest priority
        # Check knowledge_base for known boundary operations
        boundary_info = self._get_boundary_info(target)
        if boundary_info:
            return IntegrationCategory.BOUNDARY

        # 2. PROJECT TYPES - check if it's from the project
        if self._is_project_type(target, project_types):
            return IntegrationCategory.INTERUNIT

        # Check if it's a method call on a typed variable from the project
        if '.' in target:
            first_part = target.split('.')[0]
            if first_part in known_types:
                receiver_type = known_types[first_part]
                # Check if the receiver's type is a project type
                # Handle both short names and FQNs
                if receiver_type in project_types:
                    return IntegrationCategory.INTERUNIT
                # Check if any project type contains .TypeName (as a class)
                for pt in project_types:
                    if f'.{receiver_type}.' in pt or pt.endswith(f'.{receiver_type}'):
                        return IntegrationCategory.INTERUNIT

        # 3. STDLIB - check if it's Python standard library
        if self._is_stdlib_call(target, known_types):
            return IntegrationCategory.STDLIB

        # 4. KNOWN THIRD-PARTY LIBRARIES
        if self._is_known_third_party(target):
            return IntegrationCategory.EXTLIB

        # 5. UNKNOWN - can't determine
        return IntegrationCategory.UNKNOWN

    def _is_non_integration(self, target: str) -> bool:
        """
        Check if target should be filtered out (not an actual integration).

        Returns True if this is NOT an integration point.
        """

        # Get base name (last part after final dot)
        parts = target.split('.')
        base_name = parts[-1].split('(')[0]  # Remove any parentheses

        # 1. Python builtins (language constructs)
        if base_name in PYTHON_BUILTINS:
            return True

        # 2. Common builtin methods that are never integrations
        # (these are dict/list/set methods that are just data structure operations)
        if base_name in BUILTIN_METHODS:
            return True

        # 3. Self methods - ALWAYS filter out self.* calls
        # These are calls within the same class, not integration points
        # MUST come BEFORE recursive check to avoid false positives
        if target.startswith('self.'):
            return True

        # 4. cls methods - ALWAYS filter out cls.* calls
        # These are class method calls within the same class
        # MUST come BEFORE recursive check to avoid false positives
        if target.startswith('cls.'):
            return True

        # 5. Recursive call - function calling itself
        # Check if target matches the callable's own name
        # This comes AFTER self/cls checks to avoid matching self.same_method_name
        if target == self.name or target.endswith(f'.{self.name}'):
            return True

        return False

    @staticmethod
    def _get_boundary_info(target: str) -> dict[str, Any] | None:
        """
        Get boundary information if this crosses a system boundary.

        Returns:
            dict with 'kind', 'operation', 'protocol' etc, or None if not a boundary
        """
        if target in BOUNDARY_OPERATIONS:
            return BOUNDARY_OPERATIONS[target]

        # Pattern matching for common boundary patterns

        # Filesystem operations
        if 'Path.read_text' in target or '.read_text' in target:
            return {'kind': 'filesystem', 'operation': 'read'}
        if 'Path.write_text' in target or '.write_text' in target:
            return {'kind': 'filesystem', 'operation': 'write'}
        if 'Path.mkdir' in target or '.mkdir' in target:
            return {'kind': 'filesystem', 'operation': 'write'}
        if 'Path.unlink' in target or '.unlink' in target:
            return {'kind': 'filesystem', 'operation': 'write'}
        if 'Path.open' in target or target.startswith('open('):
            return {'kind': 'filesystem', 'operation': 'read/write'}
        if 'Path.resolve' in target or '.resolve' in target:
            return {'kind': 'filesystem', 'operation': 'read'}
        if 'Path.relative_to' in target or '.relative_to' in target:
            return {'kind': 'filesystem', 'operation': 'read'}
        if '.as_posix' in target:
            return {'kind': 'filesystem', 'operation': 'access'}

        # Network operations
        if 'requests.get' in target:
            return {'kind': 'network', 'protocol': 'http', 'operation': 'read'}
        if 'requests.post' in target or 'requests.put' in target:
            return {'kind': 'network', 'protocol': 'http', 'operation': 'write'}
        if 'requests.delete' in target:
            return {'kind': 'network', 'protocol': 'http', 'operation': 'delete'}
        if 'requests.' in target:
            return {'kind': 'network', 'protocol': 'http', 'operation': 'request'}
        if 'urllib.request' in target or 'urllib.urlopen' in target:
            return {'kind': 'network', 'protocol': 'http', 'operation': 'request'}

        # Environment operations
        if 'os.getenv' in target or 'os.environ' in target:
            return {'kind': 'env', 'operation': 'read'}

        # Clock/time operations
        if 'datetime.now' in target or 'datetime.utcnow' in target or 'datetime.today' in target:
            return {'kind': 'clock', 'operation': 'read'}
        if 'time.time' in target or 'time.monotonic' in target or 'time.perf_counter' in target:
            return {'kind': 'clock', 'operation': 'read'}
        if 'time.sleep' in target:
            return {'kind': 'clock', 'operation': 'sleep'}

        # Randomness operations
        if target.startswith('random.') or '.random.' in target:
            return {'kind': 'randomness', 'operation': 'generate'}

        # Subprocess operations
        if 'subprocess.run' in target or 'subprocess.call' in target or 'subprocess.Popen' in target:
            return {'kind': 'subprocess', 'operation': 'execute'}

        # Database operations
        if 'cursor.execute' in target or '.execute(' in target:
            return {'kind': 'database', 'operation': 'query'}
        if 'sqlite3.connect' in target or '.connect(' in target:
            return {'kind': 'database', 'operation': 'connect'}

        return None

    @staticmethod
    def _is_project_type(target: str, project_types: set[str]) -> bool:
        """
        Check if target is from project inventory.
        """
        # Normalize: remove empty parens from constructor calls
        # e.g., "ResolutionPolicy().to_mapping" -> "ResolutionPolicy.to_mapping"
        normalized_target = target.replace('()', '')

        # Direct match
        if normalized_target in project_types:
            return True

        # Check if target is a method/attribute of a project type
        # e.g., target="MyClass.method" matches project_type="module.MyClass.method"
        for project_type in project_types:
            if normalized_target.startswith(project_type + '.'):
                return True
            # Check if project_type ends with the target
            # e.g., target="WheelKey.as_tuple" matches "project.keys.WheelKey.as_tuple"
            if project_type.endswith(normalized_target) or project_type.endswith('.' + normalized_target):
                return True

        return False

    @staticmethod
    def _is_stdlib_call(target: str, known_types: dict[str, str]) -> bool:
        """
        Check if target is from Python standard library.
        """
        parts = target.split('.')
        if not parts:
            return False

        first_part = parts[0]

        # Check if it's a method call on a typed variable
        if '.' in target and first_part in known_types:
            receiver_type = known_types[first_part]
            if receiver_type in STDLIB_CLASSES:
                return True

        # Check if the target starts with a `stdlib` class name (e.g., str.encode, Path.resolve)
        if first_part in STDLIB_CLASSES and len(parts) > 1:
            return True

        return is_stdlib_module(first_part)

    def _is_known_third_party(self, target: str) -> bool:
        """
        Check if target is from known third-party libraries.
        """
        return any(target.startswith(f"{prefix}.") for prefix in COMMON_EXTLIB_MODULES)

    def to_ledger_callable_spec(self, project_types: set[str], known_types: dict[str, str] | None = None) -> dict[
        str, Any]:
        """
        Transform to ledger CallableSpec format.

        Args:
            project_types: set of FQNs for categorizing integrations
            known_types: optional dict mapping variable names to their types

        Returns:
            dict in ledger CallableSpec format
        """
        if known_types is None:
            known_types = {}

        spec: dict[str, Any] = {
            'branches': [b.to_ledger_ei_spec() for b in self.branches]
        }

        if self.params:
            spec['params'] = [p.to_dict() for p in self.params]

        if self.return_type:
            spec['return_type'] = self.return_type.to_dict()

        # Add categorized integrations
        integration = self.categorize_integrations(project_types, known_types)
        if integration:
            spec['integration'] = integration

        return spec
