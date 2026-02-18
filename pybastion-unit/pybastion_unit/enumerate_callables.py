#!/usr/bin/env python3
"""Enhanced callable enumerator with complete AST structural analysis."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import yaml

from callable_id_generation import (
    generate_class_id,
    generate_function_id,
    generate_nested_function_id,
    generate_method_id
)
from knowledge_base import PYTHON_BUILTINS, BUILTIN_METHODS
from models import Branch, TypeRef, ParamSpec, IntegrationCandidate, CallableEntry
from smt_path_checker import filter_feasible_paths


def load_callable_inventory(filepath: Path | None) -> dict[str, str]:
    """
    Load callable inventory file (FQN:ID pairs).

    Returns:
        Dict mapping fully qualified names to callable IDs
    """
    inventory = {}
    if not filepath or not filepath.exists():
        return inventory

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            fqn, callable_id = line.split(':', 1)
            inventory[fqn] = callable_id

    return inventory


# ============================================================================
# CFG and Path Enumeration
# ============================================================================

def build_cfg(branches: list[Branch]) -> dict[str, list[str]]:
    """
    Build control flow graph from branches.

    Exception-raising branches are terminal and don't connect to subsequent lines.

    Returns: adjacency dict {ei_id: [next_ei_ids]}
    """
    if not branches:
        return {}

    # Sort by line number
    sorted_branches = sorted(branches, key=lambda b: b.line)

    # Build adjacency graph
    graph: dict[str, list[str]] = {}

    for i, branch in enumerate(sorted_branches):
        ei_id = branch.id
        current_line = branch.line
        outcome = branch.outcome.lower()

        # Check if this branch raises an exception (terminal)
        is_exception = any(indicator in outcome for indicator in [
            'raises',
            'raise ',
            'exception',
            'error',
            'returns',
            'return ',
            '→ returns',
        ])

        if is_exception:
            # Exception paths terminate - no successors
            graph[ei_id] = []
            continue

        # Find next line's EIs
        next_eis: list[str] = []
        for j in range(i + 1, len(sorted_branches)):
            next_branch = sorted_branches[j]
            if next_branch.line > current_line:
                # This is the next executable line
                next_line = next_branch.line
                next_eis = [
                    b.id for b in sorted_branches
                    if b.line == next_line
                ]
                break

        graph[ei_id] = next_eis

    return graph


def enumerate_paths(
        graph: dict[str, list[str]],
        start_ei: str,
        target_ei: str
) -> list[list[str]]:
    """
    Enumerate all paths from start to target in the CFG.

    Handles same-line branches as alternative paths.
    Uses DFS with cycle detection.
    """

    def dfs(current: str, target: str, path: list[str], visited: set[str]) -> list[list[str]]:
        # Reached target
        if current == target:
            return [path + [current]]

        # Cycle detection
        if current in visited:
            return []

        # Mark visited
        visited_copy = visited | {current}
        all_paths: list[list[str]] = []

        # Explore all successors
        successors = graph.get(current, [])
        if not successors:
            return []

        for next_ei in successors:
            sub_paths = dfs(next_ei, target, path + [current], visited_copy)
            all_paths.extend(sub_paths)

        return all_paths

    # Try from the specified start
    paths = dfs(start_ei, target_ei, [], set())

    # The standalone path should only happen if target IS the start
    if start_ei == target_ei:
        paths = [[target_ei]]

    return paths


def add_execution_paths(entries: list[dict[str, Any]]) -> None:
    """
    Add execution_paths to integration_candidates in all callables.

    Recursively processes all entries and their children.
    """
    for entry in entries:
        # Skip if this has MechanicalOperation decorator
        skip_entry = False
        if entry.get('decorators'):
            for decorator in entry['decorators']:
                if decorator.get('name') in ['MechanicalOperation', 'UtilityOperation']:
                    # Don't enumerate paths for mechanical operations
                    for integration in entry.get('ast_analysis', {}).get('integration_candidates', []):
                        integration['execution_paths'] = []
                        integration['suppressed_by'] = decorator.get('name')
                    skip_entry = True
                    break

        if skip_entry:
            continue

        # Process this entry if it's a callable with branches and integrations
        if (entry.get('needs_callable_analysis', False) and
                'branches' in entry and
                'ast_analysis' in entry and
                'integration_candidates' in entry['ast_analysis']):

            branches_data = entry['branches']
            integration_candidates = entry['ast_analysis']['integration_candidates']

            if not branches_data or not integration_candidates:
                continue

            # Convert branch dicts to Branch objects for CFG building
            branches = [Branch.from_dict(b) for b in branches_data]

            # Build CFG for this callable
            graph = build_cfg(branches)

            # Find entry points (EIs with no predecessors)
            all_ei_ids = [b.id for b in branches]
            predecessors: dict[str, list[str]] = {ei: [] for ei in all_ei_ids}
            for ei, successors in graph.items():
                for succ in successors:
                    predecessors[succ].append(ei)

            entry_eis = [ei for ei, preds in predecessors.items() if not preds]

            # If no entry points found, use first line EIs
            if not entry_eis:
                first_line = min(b.line for b in branches)
                entry_eis = [b.id for b in branches if b.line == first_line]

            # Build line -> EI mapping
            line_to_eis: dict[int, list[str]] = {}
            for branch in branches:
                line = branch.line
                if line not in line_to_eis:
                    line_to_eis[line] = []
                line_to_eis[line].append(branch.id)

            # Enumerate paths for each integration
            for integration in integration_candidates:
                line = integration.get('line')
                if not line or line not in line_to_eis:
                    # Integration has no corresponding EI, set empty paths
                    integration['execution_paths'] = []
                    continue

                integration_eis = line_to_eis[line]

                # Match integration signature to correct EI on this line
                target_ei = None
                raw_sig = integration.get('signature', '').strip()
                # Normalize quotes by round-tripping through AST
                try:
                    integration_sig = ast.unparse(ast.parse(raw_sig, mode='eval'))
                except:
                    integration_sig = raw_sig

                # Find which EI matches this integration's signature
                for ei_id in integration_eis:
                    matching_branch = next((b for b in branches if b.id == ei_id), None)
                    if matching_branch:
                        ei_text = f"{matching_branch.condition} {matching_branch.outcome}"
                        if integration_sig in ei_text:
                            target_ei = ei_id
                            break

                # Fallback to first EI if no match
                if target_ei is None:
                    target_ei = integration_eis[0]

                all_paths: list[list[str]] = []
                for start_ei in entry_eis:
                    paths = enumerate_paths(graph, start_ei, target_ei)
                    all_paths.extend(paths)

                # Deduplicate paths
                unique_paths: list[list[str]] = []
                for path in all_paths:
                    if path not in unique_paths:
                        unique_paths.append(path)

                # === SMT FILTERING ===
                feasible_paths, feasibility_results = filter_feasible_paths(
                    unique_paths,
                    branches,
                    timeout_ms=5000
                )

                # Attach feasible paths to integration
                integration['execution_paths'] = feasible_paths

                # Store analysis metadata
                integration['path_analysis'] = {
                    'total_syntactic_paths': len(unique_paths),
                    'feasible_paths': len(feasible_paths),
                    'infeasible_paths': len(unique_paths) - len(feasible_paths),
                    'filter_effectiveness': (
                        100 * (1 - len(feasible_paths) / len(unique_paths))
                        if unique_paths else 0
                    ),
                }

                # Store witness values for test generation
                if feasible_paths:
                    witnesses = []
                    for path in feasible_paths:
                        path_id = '->'.join(path)
                        if path_id in feasibility_results:
                            result = feasibility_results[path_id]
                            if result.witness_values:
                                witnesses.append({
                                    'path': path,
                                    'witness': result.witness_values
                                })

                    if witnesses:
                        integration['test_witnesses'] = witnesses

        # Recurse into children
        if 'children' in entry and entry['children']:
            add_execution_paths(entry['children'])


# ============================================================================
# AST Visitor for Callable Enumeration
# ============================================================================

class EnhancedCallableEnumerator(ast.NodeVisitor):
    """Enumerate callables with complete structural analysis."""

    def __init__(self, source: str, unit_id: str, module_fqn: str | None = None,
                 callable_inventory: dict[str, str] | None = None) -> None:
        self.source = source
        self.unit_id = unit_id
        self.module_fqn = module_fqn or ""
        self.callable_inventory = callable_inventory or {}
        self.source_lines = source.split('\n')
        self.function_counter = 0
        self.class_counter = 0
        self.method_counters: dict[str, int] = {}
        self.context_stack: list[str] = [unit_id]  # Track current nesting: [unit_id, class_id, method_id, ...]
        self.fqn_stack: list[str] = [module_fqn] if module_fqn else []  # Track FQN stack
        self.entries: list[CallableEntry] = []  # Store CallableEntry objects
        self.import_map = {}  # bare_name -> FQN
        self.interunit_imports = set()  # FQNs that are from the project
        self.local_symbols: set[str] = set()  # All callables defined in this unit

    def _get_callable_id(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, context: str) -> str:
        """
        Get callable ID from inventory or generate it.

        Args:
            node: AST node
            context: "class" | "method" | "function"

        Returns:
            Callable ID
        """
        # Build FQN
        if self.fqn_stack:
            fqn = f"{'.'.join(self.fqn_stack)}.{node.name}"
        else:
            fqn = f"{self.module_fqn}.{node.name}" if self.module_fqn else node.name

        # Try inventory first
        if fqn in self.callable_inventory:
            return self.callable_inventory[fqn]

        # Fallback: generate ID
        if context == "class":
            self.class_counter += 1
            return generate_class_id(self.unit_id, self.class_counter)
        elif context == "method":
            parent_id = self.context_stack[-1]
            self.method_counters[parent_id] += 1
            return generate_method_id(parent_id, self.method_counters[parent_id])
        elif context == "function":
            self.function_counter += 1
            return generate_function_id(self.unit_id, self.function_counter)
        else:
            raise ValueError(f"Unknown context: {context}")

    def build_import_map(self, tree: ast.Module, project_fqns: set[str]) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    fqn = alias.name
                    self.import_map[name] = fqn
                    if fqn in project_fqns:
                        self.interunit_imports.add(name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    fqn = f"{module}.{alias.name}" if module else alias.name
                    self.import_map[name] = fqn
                    if fqn in project_fqns:
                        self.interunit_imports.add(name)

    def build_symbol_table(self, tree: ast.Module) -> None:
        """
        Build a set of all callable names defined in this module.
        Must be called before visiting the tree.
        """
        for node in ast.walk(tree):
            # Module-level functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.local_symbols.add(node.name)
                # Also add nested functions
                for nested_item in ast.walk(node):
                    if isinstance(nested_item, (ast.FunctionDef, ast.AsyncFunctionDef)) and nested_item != node:
                        self.local_symbols.add(nested_item.name)

            # Classes
            elif isinstance(node, ast.ClassDef):
                self.local_symbols.add(node.name)
                # Also add all methods in the class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.local_symbols.add(item.name)
                        # Also add nested functions
                        for nested_item in ast.walk(node):
                            if isinstance(nested_item, (ast.FunctionDef, ast.AsyncFunctionDef)) and nested_item != node:
                                self.local_symbols.add(nested_item.name)

            elif isinstance(node, (ast.AnnAssign, ast.Assign, ast.AugAssign)):
                if isinstance(node, ast.Assign):
                    # For multiple targets like a = b = value, just use the first one
                    if not node.targets:
                        return
                    first_target = node.targets[0]
                    if not isinstance(first_target, ast.Name):
                        return  # Skip non-Name targets (tuples, attributes, etc.)
                    if first_target.id.startswith('_') and first_target.id.isupper():
                        self.local_symbols.add(first_target.id)
                else:  # AnnAssign or AugAssign
                    if not isinstance(node.target, ast.Name):
                        return
                    if node.target.id.startswith('_') and node.target.id.isupper():
                        self.local_symbols.add(node.target.id)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        parent_id = self.context_stack[-1]

        # Get class ID from inventory or generate
        class_id = self._get_callable_id(node, "class")

        # Check if this is an enum
        is_enum = any(
            isinstance(base, ast.Name) and base.id == 'Enum'
            for base in node.bases
        )

        entry = CallableEntry(
            id=class_id,
            kind='enum' if is_enum else 'class',
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            decorators=self._extract_decorators(node.decorator_list),
            base_classes=[ast.unparse(base) for base in node.bases],
            children=[]
        )

        self.entries.append(entry)

        # Push this class onto context stack
        self.context_stack.append(class_id)
        if self.fqn_stack:
            self.fqn_stack.append(f"{'.'.join(self.fqn_stack)}.{node.name}")
        else:
            self.fqn_stack.append(f"{self.module_fqn}.{node.name}" if self.module_fqn else node.name)

        # Visit methods in this class
        self.method_counters[class_id] = 0
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._visit_method(item, class_id)

        # Pop class from context stack
        self.context_stack.pop()
        if self.fqn_stack:
            self.fqn_stack.pop()

        # Don't call generic_visit - we've handled children explicitly

    def _visit_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent_id: str) -> None:
        """Visit a method inside a class."""
        # Get method ID from inventory or generate
        method_id = self._get_callable_id(node, "method")

        # Analyze this method as a callable
        entry = self._analyze_callable(node, method_id, is_method=True)

        # Find parent class in entries and add as child
        for e in self.entries:
            if e.id == parent_id:
                e.children.append(entry)
                break

        # Check for nested functions
        self.context_stack.append(method_id)
        if self.fqn_stack:
            self.fqn_stack.append(f"{'.'.join(self.fqn_stack)}.{node.name}")
        for item in ast.walk(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item != node:
                # This is a nested function
                self._visit_nested_function(item, method_id)
        self.context_stack.pop()
        if self.fqn_stack:
            self.fqn_stack.pop()

    def _visit_nested_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent_id: str) -> None:
        """Visit a nested function."""
        # Generate nested function ID
        self.function_counter += 1
        nested_id = generate_nested_function_id(parent_id, self.function_counter)

        # Analyze as callable
        entry = self._analyze_callable(node, nested_id, is_method=False)

        # Find parent callable and add as child
        def find_and_add(entries: list[CallableEntry]) -> bool:
            for e in entries:
                if e.id == parent_id:
                    e.children.append(entry)
                    return True
                if e.children and find_and_add(e.children):
                    return True
            return False

        find_and_add(self.entries)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a top-level function definition."""
        # Only process if we're at unit level (not inside a class)
        if len(self.context_stack) > 1:
            return  # Inside a class, handled by _visit_method

        # Get function ID from inventory or generate
        func_id = self._get_callable_id(node, "function")
        entry = self._analyze_callable(node, func_id, is_method=False)
        self.entries.append(entry)

        # Check for nested functions
        self.context_stack.append(func_id)
        if self.fqn_stack:
            self.fqn_stack.append(f"{'.'.join(self.fqn_stack)}.{node.name}")
        elif self.module_fqn:
            self.fqn_stack.append(f"{self.module_fqn}.{node.name}")
        for item in ast.walk(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item != node:
                self._visit_nested_function(item, func_id)
        self.context_stack.pop()
        if self.fqn_stack:
            self.fqn_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition."""
        # Only process if we're at unit level
        if len(self.context_stack) > 1:
            return

        # Get function ID from inventory or generate
        func_id = self._get_callable_id(node, "function")

        entry = self._analyze_callable(node, func_id, is_method=False)
        self.entries.append(entry)

        # Check for nested functions
        self.context_stack.append(func_id)
        if self.fqn_stack:
            self.fqn_stack.append(f"{'.'.join(self.fqn_stack)}.{node.name}")
        elif self.module_fqn:
            self.fqn_stack.append(f"{self.module_fqn}.{node.name}")
        for item in ast.walk(node):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item != node:
                self._visit_nested_function(item, func_id)
        self.context_stack.pop()
        if self.fqn_stack:
            self.fqn_stack.pop()

    def _analyze_callable(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef,
            callable_id: str,
            is_method: bool
    ) -> CallableEntry:
        """Analyze a callable (function or method) and return CallableEntry object."""

        # Extract signature
        signature = self._build_signature(node)

        # Extract decorators (both regular and operation metadata)
        decorators = self._extract_decorators(node.decorator_list)
        op_metadata_decorators = self._extract_operation_metadata_decorators(node)
        all_decorators = decorators + op_metadata_decorators

        # Extract modifiers
        modifiers = self._extract_modifiers(node)

        # Extract visibility
        visibility = self._extract_visibility(node.name)

        # Extract params with types
        params = self._extract_params(node)

        # Extract return type
        return_type = self._extract_type_ref(node.returns)

        # Build parameter type map for type resolution
        param_types = self._build_param_type_map(node)

        # Find types for local variables
        local_types = self._build_local_type_map(node)

        # Find integration candidates (with type resolution)
        integration_candidates = self._find_integration_candidates(node, param_types, local_types)

        return CallableEntry(
            id=callable_id,
            kind='method' if is_method else 'function',
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            needs_callable_analysis=True,
            visibility=visibility,
            signature=signature,
            decorators=all_decorators,
            modifiers=modifiers,
            params=params,
            return_type=return_type,
            integration_candidates=integration_candidates
        )

    def _build_local_type_map(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
        """Extract local variable types from the function body."""
        local_types: dict[str, str] = {}

        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                type_ref = self._extract_type_ref(stmt.annotation)
                if type_ref:
                    # For simple types, use the name
                    # For generics like list[WheelKey], you might want just the base or the full thing
                    local_types[stmt.target.id] = type_ref.name

        return local_types

    def _build_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Build function signature string."""
        try:
            # Get the line containing the function definition
            start_line = node.lineno - 1
            sig_line = self.source_lines[start_line].strip()

            # If signature spans multiple lines, get all of them
            if not sig_line.endswith(':'):
                end_line = start_line
                for i in range(start_line + 1, min(start_line + 20, len(self.source_lines))):
                    sig_line += ' ' + self.source_lines[i].strip()
                    if self.source_lines[i].strip().endswith(':'):
                        break

            # Remove 'def ' and trailing ':'
            sig_line = sig_line.replace('async def ', '').replace('def ', '')
            if sig_line.endswith(':'):
                sig_line = sig_line[:-1].strip()

            # Normalize whitespace (collapse multiple spaces to single space)
            import re
            sig_line = re.sub(r'\s+', ' ', sig_line)
            sig_line = (sig_line
                        .replace('  ', ' ')
                        .replace('( ', '(')
                        .replace(' )', ')')
                        .replace(' ,', ','))
            return sig_line
        except Exception:
            # Fallback to unparsing
            return f"{node.name}(...)"

    def _extract_decorators(self, decorator_list: list[ast.expr]) -> list[dict[str, Any]]:
        """Extract decorators from AST nodes."""
        decorators: list[dict[str, Any]] = []

        for dec in decorator_list:
            decorator_info: dict[str, Any] = {}

            if isinstance(dec, ast.Name):
                decorator_info['name'] = dec.id
            elif isinstance(dec, ast.Call):
                decorator_info['name'] = ast.unparse(dec.func)
                if dec.args:
                    decorator_info['args'] = [ast.unparse(arg) for arg in dec.args]
                if dec.keywords:
                    decorator_info['kwargs'] = {kw.arg: ast.unparse(kw.value) for kw in dec.keywords if kw.arg}
            else:
                decorator_info['name'] = ast.unparse(dec)

            decorators.append(decorator_info)

        return decorators

    def _extract_operation_metadata_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[
        dict[str, Any]]:
        """Extract operation metadata decorators from comments."""
        decorators: list[dict[str, Any]] = []

        # Scan upward from line before function/first decorator
        if node.lineno > 1:
            # Start from line before the function/first decorator
            for line_idx in range(node.lineno - 2, -1, -1):  # -2 because lineno is 1-indexed, -1 to go before
                line = self.source_lines[line_idx].strip()

                # If it's an operation metadata decorator comment, grab it
                if line.startswith('#') and '::' in line:
                    decorator = self._parse_decorator_comment(line)
                    if decorator:
                        decorators.append(decorator)
                # If it's a Python decorator, continue (we'll skip it)
                elif line.startswith('@'):
                    continue
                # Stop at anything else (blank line, code, other comment, top of file)
                else:
                    break

        # Check docstring for decorators
        if (node.body and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):

            docstring = node.body[0].value.value
            for line in docstring.split('\n'):
                line = line.strip()
                if '::' in line:
                    decorator = self._parse_decorator_comment(line)
                    if decorator:
                        decorators.append(decorator)

        return decorators

    def _parse_decorator_comment(self, line: str) -> dict[str, Any] | None:
        """Parse operation metadata decorator from comment line."""
        # Format: # :: DecoratorName | type=value | field=value ...
        if '::' not in line:
            return None

        # Remove comment markers and leading/trailing whitespace
        line = line.lstrip('#').strip()
        if not line.startswith('::'):
            return None

        # Remove :: marker
        line = line[2:].strip()

        # Split by pipe
        parts = [p.strip() for p in line.split('|')]
        if not parts:
            return None

        decorator_name = parts[0]
        kwargs: dict[str, str] = {}

        # Parse field=value pairs
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                kwargs[key.strip()] = value.strip().strip('"').strip("'")

        return {
            'name': decorator_name,
            'kwargs': kwargs
        }

    def _extract_modifiers(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract modifiers (async, static, class, property, etc.)."""
        modifiers: list[str] = []

        if isinstance(node, ast.AsyncFunctionDef):
            modifiers.append('async')

        return modifiers

    def _extract_visibility(self, name: str) -> str:
        """Extract visibility from name (public/protected/private)."""
        if name.startswith('__') and not name.endswith('__'):
            return 'private'
        elif name.startswith('_'):
            return 'protected'
        else:
            return 'public'

    def _extract_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ParamSpec]:
        """Extract parameter information as ParamSpec objects."""
        params: list[ParamSpec] = []

        for arg in node.args.args:
            param_type = self._extract_type_ref(arg.annotation)
            param = ParamSpec(
                name=arg.arg,
                type=param_type,
                default=None
            )
            params.append(param)

        # Add defaults
        defaults = node.args.defaults
        if defaults:
            num_params = len(params)
            num_defaults = len(defaults)
            start_idx = num_params - num_defaults

            for i, default in enumerate(defaults):
                params[start_idx + i].default = ast.unparse(default)

        return params

    def _build_param_type_map(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
        """
        Build a map of parameter names to their annotated type names.

        Returns:
            dict mapping param_name -> type_name (e.g., {'path': 'Path', 'fmt': 'str'})
        """
        type_map = {}

        for arg in node.args.args:
            if arg.annotation:
                # Extract type name from annotation
                type_str = ast.unparse(arg.annotation)
                type_map[arg.arg] = type_str

        return type_map

    def _extract_type_ref(self, annotation: ast.expr | None) -> TypeRef | None:
        """Extract TypeRef from type annotation."""
        if not annotation:
            return None

        if isinstance(annotation, ast.Name):
            return TypeRef(name=annotation.id)

        if isinstance(annotation, ast.Subscript):
            base_type = ast.unparse(annotation.value)
            args: list[TypeRef] = []

            if isinstance(annotation.slice, ast.Tuple):
                for elt in annotation.slice.elts:
                    arg_ref = self._extract_type_ref(elt)
                    if arg_ref:
                        args.append(arg_ref)
            else:
                arg_ref = self._extract_type_ref(annotation.slice)
                if arg_ref:
                    args.append(arg_ref)

            return TypeRef(name=base_type, args=args)

        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            left_ref = self._extract_type_ref(annotation.left)
            right_ref = self._extract_type_ref(annotation.right)
            args = []
            if left_ref:
                args.append(left_ref)
            if right_ref:
                args.append(right_ref)
            return TypeRef(name='Union', args=args)

        return TypeRef(name=ast.unparse(annotation))

    def _find_integration_candidates(
            self,
            node: ast.FunctionDef | ast.AsyncFunctionDef,
            param_types: dict[str, str],
            local_types: dict[str, str]
    ) -> list[IntegrationCandidate]:
        """Find all potential integration points (function/method calls)."""
        candidates: list[IntegrationCandidate] = []

        # Combine param and local types for full known types
        known_types = {**param_types, **local_types}

        # Build parent map: child -> parent
        parent_map: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        # Find all Call nodes
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                target = self._get_call_target(child)
                if target and self._is_external_call(target):
                    resolved_target: str = self._resolve_target(target, known_types)

                    # Walk up parent chain to find containing statement
                    current = child
                    containing_stmt = None
                    while current in parent_map:
                        parent = parent_map[current]
                        if isinstance(parent, ast.stmt):
                            containing_stmt = parent
                            break
                        current = parent

                    # Use statement line if found, otherwise call line
                    line = containing_stmt.lineno if containing_stmt else child.lineno

                    candidate = IntegrationCandidate(
                        type='call',
                        target=resolved_target,
                        line=line,
                        signature=ast.unparse(child)
                    )
                    candidates.append(candidate)

        return candidates

    def _resolve_target(self, target: str, param_types: dict[str, str]) -> str:
        """
        Resolve target to FQN using import map and type annotations.

        Resolution strategies:
        1. Import resolution: bare_name -> FQN from imports
        2. Type annotation resolution: param.method -> Type.method
        """
        first_part = target.split('.')[0]

        # Strategy 1: Import resolution
        if first_part in self.import_map:
            fqn_first = self.import_map[first_part]
            return target.replace(first_part, fqn_first, 1)

        # Strategy 2: Type annotation resolution
        if first_part in param_types:
            param_type = param_types[first_part]
            base_type = param_type.split('[')[0]

            # Don't resolve if type is Any - leave as variable name
            if base_type == 'Any':
                return target

            # Try to find FQN for the type
            if base_type in self.import_map:
                # Type was imported, use FQN from imports
                fqn_type = self.import_map[base_type]
                return target.replace(first_part, fqn_type, 1)

            # Try to find in callable inventory (search for FQN ending with .TypeName)
            for fqn in self.callable_inventory.keys():
                if fqn.endswith(f'.{base_type}'):
                    return target.replace(first_part, fqn, 1)

            # Fallback: just use the short type name
            return target.replace(first_part, base_type, 1)

        return target

    def _is_external_call(self, target: str) -> bool:
        """
        Check if target is a call outside the current unit.

        Returns False for:
        - super() calls
        - self.* calls (same class methods)
        - cls.* calls (same class class methods)
        - Python builtins
        - Builtin collection methods
        - Same-unit function/method calls
        """

        # 1. Skip super() calls
        if target == 'super':
            return False

        # 2. Skip self/cls calls - these are NOT integration points
        if target.startswith('self.') or target.startswith('cls.') or target.startswith('object.'):
            return False

        # 3. Skip Python builtins
        parts = target.split('.')
        base_name = parts[-1].split('(')[0]
        if base_name in PYTHON_BUILTINS or base_name in BUILTIN_METHODS:
            return False

        # 4. Skip same-unit calls - if the base name is defined in this file
        # For bare function calls like "_normalize", the base_name is the function itself
        # For qualified calls like "obj.method", check if base_name (obj) is a class in this file
        first_part = target.split('.')[0]
        base_name = first_part.split('[')[0].split('(')[0]
        if base_name in self.local_symbols:
            return False

        return True

    def _get_call_target(self, call_node: ast.Call) -> str | None:
        """Extract the target of a function/method call."""
        try:
            return ast.unparse(call_node.func)
        except Exception:
            return None


def derive_fqn(filepath: Path, source_root: Path) -> str:
    """Derive fully qualified name from filepath."""
    try:
        relative = filepath.relative_to(source_root)
    except ValueError:
        # If filepath is not relative to source_root, use absolute
        relative = filepath

    # Remove .py extension and convert path separators to dots
    fqn = str(relative.with_suffix('')).replace('/', '.').replace('\\', '.')

    # Remove __init__ if present
    if fqn.endswith('.__init__'):
        fqn = fqn[:-9]

    return fqn


# ============================================================================
# File Processing
# ============================================================================

def process_file(
        filepath: Path,
        fqn: str,
        inventory_path: Path,
        unit_id: str,
        output_root: Path,
        ei_root: Path | None = None,
) -> dict[str, Any]:
    """Process a Python file and generate inventory."""

    project_types: set[str] = set()
    with open(inventory_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle both old format (FQN only) and new format (FQN:ID)
                if ':' in line:
                    fqn_part = line.split(':', 1)[0]
                    project_types.add(fqn_part)
                else:
                    project_types.add(line)

    # Load callable inventory if provided
    callable_inventory = load_callable_inventory(inventory_path) if inventory_path else {}

    # Read source
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    # Parse AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return {}

    # Enumerate callables
    enumerator = EnhancedCallableEnumerator(source, unit_id, fqn, callable_inventory)
    enumerator.entries = []  # Initialize entries list
    enumerator.build_import_map(tree, project_types)
    enumerator.build_symbol_table(tree)  # Build symbol table for same-unit filtering
    enumerator.visit(tree)

    # Convert CallableEntry objects to dicts for downstream processing
    entries = [e.to_dict() for e in enumerator.entries]

    # Load and merge EI data if provided
    if ei_root:
        # Use FQN to construct EI file path (matches Stage 2 output structure)
        ei_file = ei_root / (fqn.replace('.', '/') + '_eis.yaml')

        if ei_file.exists():
            print(f"  → Loading EI data: {ei_file}")
            with open(ei_file, 'r', encoding='utf-8') as f:
                ei_data = yaml.safe_load(f)

                if ei_data and 'functions' in ei_data:
                    def merge_ei_recursive(entries_list: list[dict[str, Any]],
                                           parent_ei_func: dict | None = None) -> None:
                        """Recursively merge EI data into entries."""
                        for entry in entries_list:
                            current_ei_func = None
                            if entry.get('needs_callable_analysis', False):
                                # Match by line range (handles duplicate names like 'resolve')
                                line_start = entry.get('line_start')
                                line_end = entry.get('line_end')

                                # Find EI func that matches this line range
                                ei_func = None
                                for func in ei_data['functions']:
                                    if func.get('line_start') == line_start and func.get('line_end') == line_end:
                                        ei_func = func
                                        break

                                if ei_func:
                                    entry['branches'] = ei_func.get('branches', [])
                                    entry['total_eis'] = ei_func.get('total_eis', 0)
                                    current_ei_func = ei_func
                                elif parent_ei_func:
                                    # Nested function - extract branches from parent by line range
                                    parent_branches = parent_ei_func.get('branches', [])
                                    nested_branches = [
                                        b for b in parent_branches
                                        if line_start <= b['line'] <= line_end
                                    ]
                                    entry['branches'] = nested_branches
                                    entry['total_eis'] = len(nested_branches)
                                    current_ei_func = None
                                else:
                                    current_ei_func = None

                            # Recurse into children
                            if 'children' in entry and entry['children']:
                                merge_ei_recursive(entry['children'], current_ei_func or parent_ei_func)

                    merge_ei_recursive(entries)

    # Add execution paths to integration candidates
    add_execution_paths(entries)

    # Count entries
    def count_all_entries(entries_list: list[dict[str, Any]]) -> int:
        """Recursively count all entries including nested."""
        count = 0
        for entry in entries_list:
            count += 1
            if 'children' in entry and entry['children']:
                count += count_all_entries(entry['children'])
        return count

    total_entries = count_all_entries(entries)

    # Count callables that need analysis
    def count_needs_analysis(entries_list: list[dict[str, Any]]) -> int:
        """Recursively count entries that need callable analysis."""
        count = 0
        for entry in entries_list:
            if entry.get('needs_callable_analysis', False):
                count += 1
            if 'children' in entry and entry['children']:
                count += count_needs_analysis(entry['children'])
        return count

    needs_analysis = count_needs_analysis(entries)

    # Count by kind
    def count_by_kind(entries_list: list[dict[str, Any]]) -> dict[str, int]:
        """Recursively count entries by kind."""
        counts: dict[str, int] = {}
        for entry in entries_list:
            kind = entry['kind']
            counts[kind] = counts.get(kind, 0) + 1
            if 'children' in entry and entry['children']:
                child_counts = count_by_kind(entry['children'])
                for k, v in child_counts.items():
                    counts[k] = counts.get(k, 0) + v
        return counts

    kind_counts = count_by_kind(entries)

    # Build inventory
    inventory: dict[str, Any] = {
        'unit': filepath.stem,
        'fully_qualified_name': fqn,
        'unit_id': unit_id,
        'filepath': str(filepath),
        'language': 'python',
        'entries': entries,
        'summary': {
            'total_entries': total_entries,
            'needs_analysis': needs_analysis,
            'classes': kind_counts.get('class', 0),
            'enums': kind_counts.get('enum', 0),
            'methods': kind_counts.get('method', 0),
            'functions': kind_counts.get('function', 0),
        }
    }

    # Save inventory
    output_path = output_root / (fqn.replace('.', '/') + '.inventory.yaml')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(inventory, f, sort_keys=False, allow_unicode=True, width=float('inf'))

    print(f"  → Unit ID: {unit_id}")
    print(f"  → {total_entries} entries, {needs_analysis} need analysis")
    print(f"  → Saved: {output_path}")

    return inventory


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Enumerate callables from Python source with AST analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--file', type=Path, required=True, help='Python source file')
    parser.add_argument('--fqn', type=str, required=True, help='Fully qualified name')
    parser.add_argument('--callable-inventory', type=Path, help='Path to callable inventory file (FQN:ID pairs)')
    parser.add_argument('--unit-id', type=str, required=True, help='Unique ID for unit')
    parser.add_argument('--output-root', type=Path, default=Path('dist/inventory'),
                        help='Root directory for inventory output')
    parser.add_argument('--ei-root', type=Path, help='Root directory containing EI YAML files')

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    print(f"Processing: {args.file}")
    print(f"  → FQN: {args.fqn}")

    process_file(args.file, args.fqn, args.callable_inventory, args.unit_id, args.output_root, args.ei_root)

    return 0


if __name__ == '__main__':
    sys.exit(main())
