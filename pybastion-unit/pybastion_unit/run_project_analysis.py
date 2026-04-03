#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from config import AnalysisConfig, load_analysis_config, validate_analysis_config


def enumerate_python_files(source_root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in source_root.rglob("*.py")
        if path.is_file() and path.name != "__init__.py"
    )


def stage2_output_path(py_file: Path, source_root: Path, eis_root: Path) -> Path:
    rel_path = py_file.resolve().relative_to(source_root.resolve())
    return eis_root / rel_path.parent / f"{py_file.stem}_eis.yaml"


def make_stage_log_name(stage_name: str, py_file: Path, source_root: Path) -> str:
    rel_path = py_file.resolve().relative_to(source_root.resolve())
    return f"{stage_name}__" + "__".join(rel_path.with_suffix("").parts) + ".log"


def run_command(cmd: list[str], log_file: Path) -> tuple[int, str]:
    rendered = " ".join(str(part) for part in cmd)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {rendered}\n")
        handle.flush()
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"\n[exit {result.returncode}]\n\n")
    return result.returncode, rendered


def print_summary(config: AnalysisConfig) -> None:
    print("=" * 72)
    print("Unit Analysis Pipeline")
    print("=" * 72)
    print(f"Target project root: {config.project_root}")
    print(f"Source root:         {config.source_root}")
    print(f"Output root:         {config.output_root}")
    print(f"Unit index:          {config.unit_index_path}")
    if config.emit_legacy_inventory:
        print(f"Legacy inventory:    {config.legacy_inventory_path}")
    print(f"EIs root:            {config.eis_root}")
    print(f"Inventory root:      {config.inventory_root}")
    print(f"Logs root:           {config.logs_root}")
    print(f"Stage 1 script:      {config.stage1_script}")
    print(f"Stage 2 script:      {config.stage2_script}")
    print(f"Stage 3 script:      {config.stage3_script}")
    print(f"Continue on error:   {config.continue_on_error}")
    print("=" * 72)


def run_stage1(config: AnalysisConfig) -> int:
    cmd = [
        sys.executable,
        str(config.stage1_script),
        str(config.source_root),
        "--output",
        str(config.unit_index_path),
    ]
    if config.emit_legacy_inventory:
        cmd.extend(["--legacy-output", str(config.legacy_inventory_path)])

    print("\nStage 1: Inspect Units")
    rc, rendered = run_command(cmd, config.logs_root / "stage1.log")
    if rc == 0:
        print("  completed")
    else:
        print(f"  failed: {rendered}")
    return rc


def run_stage2(config: AnalysisConfig, python_files: list[Path]) -> int:
    print(f"\nStage 2: Enumerate Execution Items ({len(python_files)} files)")
    failures = 0
    for py_file in python_files:
        output_file = stage2_output_path(py_file, config.source_root, config.eis_root)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(config.stage2_script),
            str(py_file),
            "--unit-index",
            str(config.unit_index_path),
            "--output",
            str(output_file),
        ]
        rc, rendered = run_command(
            cmd,
            config.logs_root / make_stage_log_name("stage2", py_file, config.source_root),
        )
        if rc != 0:
            failures += 1
            print(f"  failed: {py_file.relative_to(config.source_root)}")
            if not config.continue_on_error:
                print(f"  aborting after failure: {rendered}")
                return rc

    if failures:
        print(f"  completed with {failures} failure(s)")
        return 1
    print("  completed")
    return 0


def run_stage3(config: AnalysisConfig, python_files: list[Path]) -> int:
    print(f"\nStage 3: Enumerate Callables ({len(python_files)} files)")
    failures = 0
    for py_file in python_files:
        cmd = [
            sys.executable,
            str(config.stage3_script),
            "--unit-index",
            str(config.unit_index_path),
            "--file",
            str(py_file),
            "--ei-root",
            str(config.eis_root),
            "--output-root",
            str(config.inventory_root),
        ]
        rc, rendered = run_command(
            cmd,
            config.logs_root / make_stage_log_name("stage3", py_file, config.source_root),
        )
        if rc != 0:
            failures += 1
            print(f"  failed: {py_file.relative_to(config.source_root)}")
            if not config.continue_on_error:
                print(f"  aborting after failure: {rendered}")
                return rc

    if failures:
        print(f"  completed with {failures} failure(s)")
        return 1
    print("  completed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the unit analysis pipeline")
    parser.add_argument("project_root", type=Path, help="Root of the target project to analyze")
    parser.add_argument("--config", type=Path, help="Optional TOML config override")
    args = parser.parse_args(argv)

    config = load_analysis_config(args.project_root, args.config)
    validate_analysis_config(config)

    if config.clean_logs and config.logs_root.exists():
        shutil.rmtree(config.logs_root)

    if config.clean_outputs and config.output_root.exists():
        for output_dir in [
            config.inspect_root,
            config.eis_root,
            config.inventory_root,
        ]:
            if output_dir.exists():
                shutil.rmtree(output_dir)

    config.inspect_root.mkdir(parents=True, exist_ok=True)
    config.eis_root.mkdir(parents=True, exist_ok=True)
    config.inventory_root.mkdir(parents=True, exist_ok=True)
    config.logs_root.mkdir(parents=True, exist_ok=True)

    print_summary(config)

    if config.run_stage1:
        rc = run_stage1(config)
        if rc != 0:
            return rc
    elif not config.unit_index_path.exists():
        print(f"Stage 1 is disabled, but unit index does not exist: {config.unit_index_path}")
        return 1

    python_files = enumerate_python_files(config.source_root)
    print(f"\nDiscovered {len(python_files)} Python files under {config.source_root}")

    if config.run_stage2:
        rc = run_stage2(config, python_files)
        if rc != 0 and not config.continue_on_error:
            return rc

    if config.run_stage3:
        rc = run_stage3(config, python_files)
        if rc != 0 and not config.continue_on_error:
            return rc

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
