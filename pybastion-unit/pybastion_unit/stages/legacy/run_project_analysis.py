#!/usr/bin/env python3
"""
Project Analysis Coordinator

Orchestrates the multi-stage pipeline for analyzing Python projects:
1. stage1_inspect_units.py - Basic unit structure
2. stage2_enumerate_exec_items.py - EI enumeration
3. stage3_enumerate_callables.py - Integration classification + EI merging
4. stage4_analyze_code_quality.py - Code quality metrics
5. stage5_inventory_to_ledger.py - Generate three-document ledgers with quality metrics
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pybastion_unit.shared.callable_id_generation import generate_unit_id


def run_command(cmd: list[str], description: str, log_dir: Path = Path("/tmp/ledger"), append: bool = False) -> bool:
    """Run a command and return success status."""
    Path.mkdir(log_dir, exist_ok=True, parents=True)

    # Create a log file name from description
    log_name = description.lower().replace(' ', '_').replace(':', '').replace('-', '_').split('__')[0] + '.log'
    log_file = log_dir / log_name

    if not log_file.exists():
        print(f"\n{'=' * 70}")
        print(f"{description}")
        print(f"{'=' * 70}")
        # print(f"Command: {' '.join(str(c) for c in cmd)}\n")

    mode = 'a' if append else 'w'
    with open(log_file, mode) as f:
        if append:
            f.write(f"\n\n{'=' * 70}\n{description}\n{'='*70}\n")
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return result.returncode == 0


def derive_fqn(filepath: Path, source_root: Path) -> str:
    """Derive fully qualified name from filepath."""
    try:
        relative = filepath.relative_to(source_root)
    except ValueError:
        # If the filepath is not relative to source_root, use absolute
        relative = filepath

    # Remove .py extension and convert path separators to dots
    fqn = str(relative.with_suffix('')).replace('/', '.').replace('\\', '.')

    # Remove __init__ if present
    if fqn.endswith('.__init__'):
        fqn = fqn[:-9]

    return fqn


def analyze_project(
        project_root: Path,
        source_root: str = "src",
        output_root: str = "dist",
) -> bool:
    """
    Run full analysis pipeline on a project.

    Pipeline stages:
    1. inspect_units - Generate basic unit structure
    2. enumerate_exec_items - Enumerate execution items (EIs)
    3. enumerate_callables - Classify integrations and merge EI data
    """

    project_root = project_root.absolute()
    source_path = project_root / source_root
    output_path = project_root / output_root

    if not source_path.exists():
        print(f"Error: Source root not found: {source_path}")
        return False

    # Output directories
    script_stages_dir = Path(__file__).parent / "stages"
    inspect_output = output_path / "inspect" / "callable-inventory.txt"
    eis_output = output_path / "eis"
    inventory_output = output_path / "inventory"
    quality_output = output_path / "quality"
    ledgers_output = output_path / "ledgers"
    log_dir = Path("/tmp/ledger")

    # Clean up old logs
    if log_dir.exists():
        import shutil
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Project Analysis Pipeline")
    print(f"{'=' * 70}")
    print(f"Project root:      {project_root}")
    print(f"Source root:       {source_path}")
    print(f"Inspect output:    {inspect_output}")
    print(f"EIs output:        {eis_output}")
    print(f"Inventory output:  {inventory_output}")
    print(f"Quality output:    {quality_output}")
    print(f"Ledgers output:    {ledgers_output}")
    print(f"{'=' * 70}")

    # Stage 1: inspect_units (if it exists)
    inspect_script = script_stages_dir / "stage1_inspect_units.py"
    if inspect_script.exists():
        cmd = [
            sys.executable,
            str(inspect_script),
            str(source_path),
            "--output", str(inspect_output)
        ]
        if not run_command(cmd, "Stage 1: Inspect Units"):
            return False
    else:
        print(f"\nℹ  Skipping Stage 1: stage1_inspect_units.py not found")

    # Stage 2: enumerate_exec_items
    enumerate_eis_script = script_stages_dir / "stage2_enumerate_exec_items.py"
    if not enumerate_eis_script.exists():
        print(f"Error: stage2_enumerate_exec_items.py not found at {enumerate_eis_script}")
        return False

    # Create output directory
    eis_output.mkdir(parents=True, exist_ok=True)

    # Find all Python files
    py_files = list(source_path.rglob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]

    print(f"Stage 2: Enumerate Execution Items")
    print(f"Found {len(py_files)} Python files\n")

    for py_file in sorted(py_files):
        rel_path = py_file.relative_to(source_path)
        output_file = eis_output / rel_path.parent / f"{py_file.stem}_eis.yaml"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(enumerate_eis_script),
            str(py_file),
            "--callable-inventory", str(inspect_output),
            "--source-root", str(source_path),
            "--unit-id", generate_unit_id(derive_fqn(py_file, source_path)),
            "--output", str(output_file)
        ]

        print(f"Processing: {rel_path}")
        if not run_command(cmd, f"Stage 2: Enumerate Execution Items", append=True):
            print(f"  ✗ Failed")
        else:
            print(f"  ✓ {output_file.relative_to(project_root)}")

    print(f"\n✓ Completed: Stage 2")

    # Stage 3: enumerate_callables (per-file processing)
    enumerate_callables_script = script_stages_dir / "stage3_enumerate_callables.py"
    if not enumerate_callables_script.exists():
        print(f"Error: stage3_enumerate_callables.py not found at {enumerate_callables_script}")
        return False

    # Create output directory
    inventory_output.mkdir(parents=True, exist_ok=True)

    print(f"Stage 3: Enumerate Callables + Merge EI Data")
    print(f"Processing {len(py_files)} Python files\n")

    for py_file in sorted(py_files):
        rel_path = py_file.relative_to(source_path)
        fqn = derive_fqn(py_file, source_path)
        if fqn.endswith('.__init__'):
            fqn = fqn[:-9]

        cmd = [
            sys.executable,
            str(enumerate_callables_script),
            "--file", str(py_file),
            "--fqn", fqn,
            "--callable-inventory", str(inspect_output),
            "--unit-id", generate_unit_id(fqn),
            "--output-root", str(inventory_output),
            "--ei-root", str(eis_output)
        ]

        print(f"Processing: {rel_path}")
        if not run_command(cmd, f"Stage 3: Enumerate Callables", append=True):
            print(f"  ✗ Failed")
        else:
            print(f"  ✓ Inventory generated")

    print(f"\n✓ Completed: Stage 3")

    # Stage 4: Quality Analysis
    analyze_quality_script = script_stages_dir / "stage4_analyze_code_quality.py"
    if not analyze_quality_script.exists():
        print(f"\nℹ  Skipping Stage 4: stage4_analyze_code_quality.py not found")
    else:
        # Create quality output directory
        quality_output.mkdir(parents=True, exist_ok=True)

        print(f"Stage 4: Quality Analysis")
        print(f"Analyzing {len(py_files)} Python files\n")

        for py_file in sorted(py_files):
            rel_path = py_file.relative_to(source_path)
            quality_file = quality_output / rel_path.parent / f"{py_file.stem}.quality.yaml"
            quality_file.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(analyze_quality_script),
                str(py_file),
                "--output", str(quality_file)
            ]

            print(f"Processing: {rel_path}")
            if not run_command(cmd, f"Stage 4: Quality Analysis", append=True):
                print(f"  ✗ Failed")
            else:
                # Parse quality grade for quick feedback
                try:
                    import yaml
                    quality_data = yaml.safe_load(quality_file.read_text())
                    grade = quality_data.get('overallGrade', 'unknown')
                    print(f"  ✓ Grade: {grade}")
                except Exception:
                    print(f"  ✓ {quality_file.relative_to(project_root)}")

        print(f"\n✓ Completed: Stage 4")

    # Stage 5: Generate ledgers
    inventory_to_ledger_script = script_stages_dir / "stage5_inventory_to_ledger.py"
    if not inventory_to_ledger_script.exists():
        print(f"\nℹ  Skipping Stage 5: stage5_inventory_to_ledger.py not found")
    else:
        # Create ledgers output directory
        ledgers_output.mkdir(parents=True, exist_ok=True)

        # Find all inventory files
        inventory_files = list(inventory_output.rglob("*.inventory.yaml"))

        print(f"Stage 5: Generate Ledgers")
        print(f"Found {len(inventory_files)} inventory files\n")

        for inventory_file in sorted(inventory_files):
            rel_path = inventory_file.relative_to(inventory_output)
            ledger_file = ledgers_output / rel_path.parent / f"{inventory_file.stem.replace('.inventory', '')}.ledger.yaml"
            ledger_file.parent.mkdir(parents=True, exist_ok=True)

            # Find corresponding quality file
            quality_file = quality_output / rel_path.parent / f"{inventory_file.stem.replace('.inventory', '')}.quality.yaml"

            cmd = [
                sys.executable,
                str(inventory_to_ledger_script),
                "--inventory", str(inventory_file),
                "--project-inventory", str(inspect_output),
                "--output", str(ledger_file)
            ]

            # Add quality file if it exists
            if quality_file.exists():
                cmd.extend(["--quality-file", str(quality_file)])

            print(f"Processing: {rel_path}")
            if not run_command(cmd, f"Stage 5: Generate Ledgers", append=True):
                print(f"  ✗ Failed")
            else:
                print(f"  ✓ {ledger_file.relative_to(project_root)}")

        print(f"\n✓ Completed: Stage 5")

    print(f"\n{'=' * 70}")
    print(f"✓ Analysis Complete!")
    print(f"{'=' * 70}")
    print(f"\nOutputs:")
    if inspect_output.exists():
        print(f"  - Unit inspection: {inspect_output} (YAML)")
    print(f"  - EI enumeration:  {eis_output} (YAML)")
    print(f"  - Final inventory: {inventory_output} (YAML)")
    if quality_output.exists():
        print(f"  - Quality metrics: {quality_output} (YAML)")
    if ledgers_output.exists():
        print(f"  - Unit ledgers:    {ledgers_output} (YAML)")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Coordinate multi-stage analysis pipeline for Python projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. stage1_inspect_units.py        - Generate basic unit structure (optional)
  2. stage2_enumerate_exec_items.py - Enumerate execution items (EIs)
  3. stage3_enumerate_callables.py  - Classify integrations + merge EI data
  4. stage4_analyze_code_quality.py - Code quality metrics (optional)
  5. stage5_inventory_to_ledger.py  - Generate three-document unit ledgers

Example:
  %(prog)s /path/to/project --source-root src --output-root dist
        """
    )

    parser.add_argument(
        'project_root',
        type=Path,
        default=Path.cwd(),
        help='Project root directory'
    )
    parser.add_argument(
        '--source-root',
        type=str,
        default='src',
        help='Source root relative to project root (default: "src")'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='dist/pybastion',
        help='Output root relative to project root (default: "dist")'
    )

    args = parser.parse_args()

    if not args.project_root.exists():
        print(f"Error: Project root not found: {args.project_root}")
        sys.exit(1)

    success = analyze_project(
        args.project_root,
        args.source_root,
        args.output_root
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()