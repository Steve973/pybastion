#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from pybastion_unit.config import (
    AnalysisConfig,
    load_analysis_config,
    validate_analysis_config,
)

ALL_STAGE_NUMS = [1, 2, 3]
STAGE_NAMES = {
    1: "Inspect Units",
    2: "Enumerate Execution Items",
    3: "Enumerate Callables",
}


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


def shell_quote(parts: list[str]) -> str:
    return " ".join(str(part) for part in parts)


def run_command(cmd: list[str], log_file: Path, *, dry_run: bool = False, verbose: bool = False) -> tuple[int, str]:
    rendered = shell_quote(cmd)
    if dry_run:
        print(f"Would run: {rendered}")
        return 0, rendered

    if verbose:
        print(f"  $ {rendered}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {rendered}\n")
        handle.flush()
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"\n[exit {result.returncode}]\n\n")
    return result.returncode, rendered


def print_summary(config: AnalysisConfig, stages_to_run: list[int], *, run_readiness: bool, dry_run: bool) -> None:
    print("=" * 72)
    print("Unit Analysis Pipeline")
    print("=" * 72)
    print(f"Target project root:  {config.project_root}")
    print(f"Source root:          {config.source_root}")
    print(f"Analysis output root: {config.analysis_output_root}")
    print(f"Inspect root:         {config.inspect_root}")
    print(f"Unit index:           {config.unit_index_path}")
    print(f"EIs root:             {config.eis_root}")
    print(f"Inventory root:       {config.inventory_root}")
    print(f"Logs root:            {config.logs_root}")
    print(f"Stage 1 script:       {config.stage1_script}")
    print(f"Stage 2 script:       {config.stage2_script}")
    print(f"Stage 3 script:       {config.stage3_script}")
    print(f"Readiness script:     {config.readiness_script}")
    print(f"Readiness output:     {config.readiness_output_path}")
    print(f"Stages to run:        {stages_to_run}")
    print(f"Run readiness:        {run_readiness}")
    print(f"Continue on error:    {config.continue_on_error}")
    print(f"Dry run:              {dry_run}")
    print("=" * 72)


def readiness_command(config: AnalysisConfig) -> list[str]:
    cmd = [
        sys.executable,
        str(config.readiness_script),
        "--project-path",
        str(config.project_root),
        "--source-root",
        str(config.source_root),
        "--output",
        str(config.readiness_output_path),
        "--format",
        config.readiness_format,
        "--grouping",
        config.readiness_grouping,
        "--config",
        str(config.config_path),
    ]
    if config.readiness_max_findings_per_type is not None:
        cmd.extend(["--max-findings-per-type", str(config.readiness_max_findings_per_type)])
    return cmd


def run_readiness(config: AnalysisConfig, *, dry_run: bool, verbose: bool) -> int:
    print("\nReadiness: Source Preflight")
    rc, rendered = run_command(
        readiness_command(config),
        config.logs_root / "readiness.log",
        dry_run=dry_run,
        verbose=verbose,
    )
    if rc == 0:
        print("  completed")
    else:
        print(f"  failed: {rendered}")
    return rc


def run_stage1(config: AnalysisConfig, *, dry_run: bool, verbose: bool) -> int:
    cmd = [
        sys.executable,
        str(config.stage1_script),
        str(config.source_root),
        "--output",
        str(config.unit_index_path),
    ]

    print("\nStage 1: Inspect Units")
    rc, rendered = run_command(
        cmd,
        config.logs_root / "stage1.log",
        dry_run=dry_run,
        verbose=verbose,
    )
    if rc == 0:
        print("  completed")
    else:
        print(f"  failed: {rendered}")
    return rc


def run_stage2(config: AnalysisConfig, python_files: list[Path], *, dry_run: bool, verbose: bool) -> int:
    print(f"\nStage 2: Enumerate Execution Items ({len(python_files)} files)")
    failures = 0
    first_failure_rc = 0

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
            dry_run=dry_run,
            verbose=verbose,
        )
        if rc != 0:
            failures += 1
            if first_failure_rc == 0:
                first_failure_rc = rc
            print(f"  failed: {py_file.relative_to(config.source_root)}")
            if not config.continue_on_error:
                print(f"  aborting after failure: {rendered}")
                return rc

    if failures:
        print(f"  completed with {failures} failure(s)")
        return first_failure_rc or 1

    print("  completed")
    return 0


def run_stage3(config: AnalysisConfig, python_files: list[Path], *, dry_run: bool, verbose: bool) -> int:
    print(f"\nStage 3: Enumerate Callables ({len(python_files)} files)")
    failures = 0
    first_failure_rc = 0

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
            dry_run=dry_run,
            verbose=verbose,
        )
        if rc != 0:
            failures += 1
            if first_failure_rc == 0:
                first_failure_rc = rc
            print(f"  failed: {py_file.relative_to(config.source_root)}")
            if not config.continue_on_error:
                print(f"  aborting after failure: {rendered}")
                return rc

    if failures:
        print(f"  completed with {failures} failure(s)")
        return first_failure_rc or 1

    print("  completed")
    return 0


def determine_stages(args: argparse.Namespace) -> list[int]:
    if args.only is not None:
        return [args.only]

    start = args.start_from or ALL_STAGE_NUMS[0]
    stop = args.stop_at or ALL_STAGE_NUMS[-1]

    if start > stop:
        raise ValueError(f"--start-from ({start}) cannot be greater than --stop-at ({stop})")

    return list(range(start, stop + 1))


def clean_outputs(config: AnalysisConfig, stages_to_run: list[int], *, run_readiness: bool, verbose: bool) -> None:
    if config.clean_logs and config.logs_root.exists():
        if verbose:
            print(f"Removing logs root: {config.logs_root}")
        shutil.rmtree(config.logs_root)

    if not config.clean_outputs:
        return

    targets: list[Path] = []
    if run_readiness:
        targets.append(config.readiness_output_path)
    if 1 in stages_to_run:
        targets.append(config.inspect_root)
    if 2 in stages_to_run:
        targets.append(config.eis_root)
    if 3 in stages_to_run:
        targets.append(config.inventory_root)

    for path in targets:
        if not path.exists():
            continue
        if verbose:
            print(f"Removing output: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def ensure_output_dirs(config: AnalysisConfig) -> None:
    config.analysis_output_root.mkdir(parents=True, exist_ok=True)
    config.inspect_root.mkdir(parents=True, exist_ok=True)
    config.eis_root.mkdir(parents=True, exist_ok=True)
    config.inventory_root.mkdir(parents=True, exist_ok=True)
    config.logs_root.mkdir(parents=True, exist_ok=True)
    config.readiness_output_path.parent.mkdir(parents=True, exist_ok=True)


def validate_stage_inputs(config: AnalysisConfig, stages_to_run: list[int]) -> None:
    if 2 in stages_to_run and 1 not in stages_to_run and not config.unit_index_path.exists():
        raise FileNotFoundError(f"Stage 2 requires unit index: {config.unit_index_path}")

    if 3 in stages_to_run:
        if 1 not in stages_to_run and not config.unit_index_path.exists():
            raise FileNotFoundError(f"Stage 3 requires unit index: {config.unit_index_path}")
        if 2 not in stages_to_run and not config.eis_root.exists():
            raise FileNotFoundError(f"Stage 3 requires EI root: {config.eis_root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyBastion unit analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("project_root", type=Path, help="Root of the target project to analyze")
    parser.add_argument("--config", type=Path, help="TOML config file")
    parser.add_argument("--start-from", type=int, choices=ALL_STAGE_NUMS, help="Run from this stage through --stop-at")
    parser.add_argument("--stop-at", type=int, choices=ALL_STAGE_NUMS, help="Stop at this stage")
    parser.add_argument("--only", type=int, choices=ALL_STAGE_NUMS, help="Run only one stage")
    parser.add_argument("--readiness", action="store_true",
                        help="Run the source readiness scanner before selected stages")
    parser.add_argument("--only-readiness", action="store_true", help="Run only the source readiness scanner")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean selected outputs before running")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print commands as they run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = load_analysis_config(args.project_root, args.config)
        run_readiness_requested = args.readiness or args.only_readiness or config.readiness_enabled
        validate_analysis_config(config, require_readiness=run_readiness_requested)
        stages_to_run = [] if args.only_readiness else determine_stages(args)
        validate_stage_inputs(config, stages_to_run)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.no_clean and not args.dry_run:
        clean_outputs(
            config,
            stages_to_run,
            run_readiness=run_readiness_requested,
            verbose=args.verbose,
        )

    if not args.dry_run:
        ensure_output_dirs(config)

    print_summary(
        config,
        stages_to_run,
        run_readiness=run_readiness_requested,
        dry_run=args.dry_run,
    )

    if run_readiness_requested:
        rc = run_readiness(config, dry_run=args.dry_run, verbose=args.verbose)
        if rc != 0:
            return rc

    if not stages_to_run:
        print("\nDone.")
        return 0

    python_files: list[Path] | None = None

    for stage_num in stages_to_run:
        print(f"\n{'=' * 72}")
        print(f"Stage {stage_num}: {STAGE_NAMES[stage_num]}")
        print(f"{'=' * 72}")

        if stage_num == 1:
            rc = run_stage1(config, dry_run=args.dry_run, verbose=args.verbose)
        elif stage_num == 2:
            if python_files is None:
                python_files = enumerate_python_files(config.source_root)
                print(f"\nDiscovered {len(python_files)} Python files under {config.source_root}")
            rc = run_stage2(config, python_files, dry_run=args.dry_run, verbose=args.verbose)
        elif stage_num == 3:
            if python_files is None:
                python_files = enumerate_python_files(config.source_root)
                print(f"\nDiscovered {len(python_files)} Python files under {config.source_root}")
            rc = run_stage3(config, python_files, dry_run=args.dry_run, verbose=args.verbose)
        else:  # pragma: no cover
            raise AssertionError(f"Unsupported stage: {stage_num}")

        if rc != 0 and not config.continue_on_error:
            return rc
        if rc != 0 and stage_num == 1:
            return rc

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
