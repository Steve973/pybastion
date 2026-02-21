#!/usr/bin/env python3
"""
Find Unannotated Loop Variables

Scans ledger files to find 'unknown' integration points where the target
is a dotted name with a local variable receiver (e.g. c.get, ws.marker.evaluate).
These are candidates for pre-loop type annotations in source code.

Usage:
    python find_unannotated_loop_vars.py <ledgers_root>
    python find_unannotated_loop_vars.py /path/to/ledgers --source-root /path/to/src

Output:
    Checklist of files, line numbers, variable names, and annotation suggestions.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Ledger scanning
# =============================================================================

def load_ledger(path: Path) -> list[dict]:
    """Load all YAML documents from a ledger file."""
    with open(path, 'r', encoding='utf-8') as f:
        return list(yaml.safe_load_all(f))


def find_ledger_doc(documents: list[dict]) -> dict | None:
    """Find the main ledger document (docKind: ledger)."""
    for doc in documents:
        if doc and doc.get('docKind') == 'ledger':
            return doc
    return None


def find_integration_doc(documents: list[dict]) -> dict | None:
    """Find the integration points document (docKind: integration-points or similar)."""
    for doc in documents:
        if doc and doc.get('docKind') in ('integration-points', 'integrations'):
            return doc
    return None


def is_local_variable_receiver(target: str) -> tuple[bool, str, str]:
    """
    Check if a dotted target looks like a local variable receiver.

    e.g. 'c.get' -> (True, 'c', 'get')
         'ws.marker.evaluate' -> (True, 'ws', 'marker.evaluate')
         'ResolutionStrategyConfig.get' -> (False, '', '')
         'open_repository' -> (False, '', '')

    Heuristic: receiver is a local variable if it's a short lowercase name
    (1-3 chars) OR matches a known loop variable pattern, AND is not a
    known module/class name (starts with uppercase or contains underscore
    suggesting a module).
    """
    if '.' not in target:
        return False, '', ''

    parts = target.split('.', 1)
    receiver = parts[0]
    method = parts[1]

    # Skip if receiver looks like a module or class (uppercase start)
    if receiver[0].isupper():
        return False, '', ''

    # Skip if receiver looks like a module path (contains underscores suggesting module)
    # but allow short names like 'wk', 'ws', 'c', 'env'
    if '_' in receiver and len(receiver) > 4:
        return False, '', ''

    # Skip if receiver is an expression (contains parens) not a simple variable
    if '(' in receiver or ')' in receiver:
        return False, '', ''

    # It looks like a local variable receiver
    return True, receiver, method


def scan_ledger_for_unknowns(ledger_path: Path) -> list[dict[str, Any]]:
    """
    Scan a ledger file for unknown integration points with local variable receivers.

    Returns list of findings with context for annotation.
    """
    findings = []

    try:
        documents = load_ledger(ledger_path)
    except Exception as e:
        print(f"  WARNING: Could not load {ledger_path}: {e}", file=sys.stderr)
        return findings

    ledger_doc = find_ledger_doc(documents)
    if not ledger_doc:
        return findings

    unit = ledger_doc.get('unit', {})
    unit_name = unit.get('name', 'unknown')

    # Walk the callable tree to find integration points
    def walk_callable(entry: dict, callable_name: str, callable_id: str) -> None:
        # Skip callables decorated with MechanicalOperation or UtilityOperation —
        # these are fixtured out in integration testing, unknowns are irrelevant
        decorators = entry.get('decorators', [])
        for decorator in decorators:
            if decorator.get('name') in ('MechanicalOperation', 'UtilityOperation'):
                return

        callable_data = entry.get('callable', {})
        integration = callable_data.get('integration', {})

        # Integrations are grouped by category: interunit, unknown, boundary, stdlib, extlib
        unknown_integrations = integration.get('unknown', []) or []

        for integ in unknown_integrations:
            target = integ.get('target', '')
            if not target:
                continue

            is_local, receiver, method = is_local_variable_receiver(target)
            if not is_local:
                continue

            # Get line hint from the last EI in the first execution path
            line_hint = None
            exec_paths = integ.get('execution_paths', [])
            if exec_paths and exec_paths[0]:
                last_ei_id = exec_paths[0][-1]
                branches = callable_data.get('branches', [])
                for branch in branches:
                    if branch.get('id') == last_ei_id:
                        line_hint = branch.get('line')
                        break

            findings.append({
                'unit': unit_name,
                'callable': callable_name,
                'callable_id': callable_id,
                'integration_id': integ.get('id', ''),
                'target': target,
                'receiver': receiver,
                'method': method,
                'line_hint': line_hint,
                'ledger_path': str(ledger_path),
            })

    def walk_entries(entry: dict) -> None:
        kind = entry.get('kind', '')
        name = entry.get('name', '')
        entry_id = entry.get('id', '')

        if kind in ('function', 'method'):
            walk_callable(entry, name, entry_id)

        for child in entry.get('children', []):
            walk_entries(child)

    walk_entries(unit)

    return findings


def discover_ledgers(root: Path) -> list[Path]:
    """Find all ledger YAML files under root."""
    ledgers = []
    for pattern in ('**/*.ledger.yaml', '**/*_ledger.yaml', '**/*-ledger.yaml', '**/ledger.yaml'):
        ledgers.extend(root.glob(pattern))
    return sorted(set(ledgers))


# =============================================================================
# Source file analysis
# =============================================================================

def find_loop_variable_in_source(
        source_path: Path,
        receiver: str,
        callable_name: str
) -> list[dict[str, Any]]:
    """
    Find loop statements in source where `receiver` is the loop variable.

    Returns list of {line, loop_var, iterable_expr, already_annotated}
    """
    results = []

    if not source_path.exists():
        return results

    try:
        source = source_path.read_text(encoding='utf-8')
        source_lines = source.splitlines()
        tree = ast.parse(source)
    except Exception:
        return results

    # Find the target function first
    target_func = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == callable_name:
                target_func = node
                break

    if not target_func:
        return results

    # Walk the function body looking for for-loops with this receiver as target
    for node in ast.walk(target_func):
        if not isinstance(node, ast.For):
            continue

        # Check if loop target matches our receiver
        if isinstance(node.target, ast.Name) and node.target.id == receiver:
            iterable_expr = ast.unparse(node.iter)
            line = node.lineno

            # Check if there's already an annotation on the line before
            already_annotated = False
            if line > 1:
                prev_line = source_lines[line - 2].strip()  # -2 because 0-indexed and previous line
                # Check for 'varname: Type' annotation pattern
                if re.match(rf'^{re.escape(receiver)}\s*:', prev_line):
                    already_annotated = True

            results.append({
                'line': line,
                'loop_var': receiver,
                'iterable_expr': iterable_expr,
                'already_annotated': already_annotated,
            })

    return results


# =============================================================================
# Report generation
# =============================================================================

def generate_report(
        findings: list[dict[str, Any]],
        source_root: Path | None
) -> str:
    """Generate human-readable checklist report."""
    if not findings:
        return "✓ No unannotated loop variables found!\n"

    lines = []
    lines.append("=" * 70)
    lines.append("UNANNOTATED LOOP VARIABLE REPORT")
    lines.append("Loop variables making method calls that need type annotations")
    lines.append("=" * 70)
    lines.append("")

    # Group by unit
    by_unit: dict[str, list[dict]] = {}
    for f in findings:
        unit = f['unit']
        if unit not in by_unit:
            by_unit[unit] = []
        by_unit[unit].append(f)

    total_needed = 0
    total_done = 0

    for unit_name, unit_findings in sorted(by_unit.items()):
        lines.append(f"MODULE: {unit_name}")
        lines.append("-" * 70)

        for f in unit_findings:
            receiver = f['receiver']
            target = f['target']
            callable_name = f['callable']

            # Try to find source info
            source_info = []
            if source_root:
                # Try to find the source file
                candidates = list(source_root.glob(f"**/{unit_name}.py"))
                for src_path in candidates:
                    loops = find_loop_variable_in_source(src_path, receiver, callable_name)
                    for loop in loops:
                        source_info.append({
                            'source_path': src_path,
                            **loop
                        })

            if source_info:
                for si in source_info:
                    already = si['already_annotated']
                    status = "✓ DONE" if already else "✗ NEEDED"
                    if already:
                        total_done += 1
                    else:
                        total_needed += 1

                    lines.append(f"  {status}")
                    lines.append(f"  Callable:  {callable_name}")
                    lines.append(f"  File:      {si['source_path']}")
                    lines.append(f"  Line:      {si['line']} (for {receiver} in {si['iterable_expr']}:)")
                    lines.append(f"  Target:    {target}")
                    if not already:
                        lines.append(f"  Fix:       Add '{receiver}: <Type>' on line {si['line'] - 1}")
                    lines.append("")
            else:
                total_needed += 1
                lines.append(f"  ✗ NEEDED")
                lines.append(f"  Callable:  {callable_name}")
                lines.append(f"  Target:    {target}")
                lines.append(f"  Receiver:  '{receiver}' (loop variable, type unknown)")
                if f['line_hint']:
                    lines.append(f"  Near line: {f['line_hint']}")
                lines.append("")

        lines.append("")

    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Total unknown dotted targets: {len(findings)}")
    lines.append(f"  Already annotated:            {total_done}")
    lines.append(f"  Still needed:                 {total_needed}")
    if total_needed == 0:
        lines.append("")
        lines.append("  ✓ All boobies have nipples!")
    else:
        lines.append("")
        lines.append(f"  {total_needed} annotation(s) needed across {len(by_unit)} module(s)")
    lines.append("=" * 70)

    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        'ledgers_root',
        type=Path,
        help='Root directory to search for ledger files'
    )
    ap.add_argument(
        '--source-root',
        type=Path,
        help='Source root for locating Python files (enables line numbers and done/needed status)'
    )
    ap.add_argument(
        '--output',
        type=Path,
        help='Save report to file instead of printing'
    )
    ap.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show ledgers being scanned'
    )

    args = ap.parse_args()

    if not args.ledgers_root.exists():
        print(f"ERROR: Ledgers root not found: {args.ledgers_root}", file=sys.stderr)
        return 1

    # Discover ledgers
    ledger_paths = discover_ledgers(args.ledgers_root)

    if not ledger_paths:
        print(f"No ledger files found under: {args.ledgers_root}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(ledger_paths)} ledger(s)")

    # Scan all ledgers
    all_findings = []
    for path in ledger_paths:
        if args.verbose:
            print(f"  Scanning: {path.name}")
        findings = scan_ledger_for_unknowns(path)
        all_findings.extend(findings)

    if args.verbose:
        print(f"Found {len(all_findings)} unknown dotted targets total")
        print()

    # Generate report
    report = generate_report(all_findings, args.source_root)

    if args.output:
        args.output.write_text(report, encoding='utf-8')
        print(f"Report saved to: {args.output}")
    else:
        print(report)

    # Return 1 if there are needed annotations, 0 if all done
    return 1 if all_findings else 0


if __name__ == '__main__':
    sys.exit(main())