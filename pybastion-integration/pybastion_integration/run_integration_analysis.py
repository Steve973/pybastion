#!/usr/bin/env python3
"""
Integration analysis pipeline driver.

Runs the current four-stage pybastion integration analysis pipeline:

  Stage 1: Build the EI-level call graph from inventory files
  Optional: Check the inventory graph produced by Stage 1
  Stage 2: Trace feature-flow cases from feature markers and the EI CFG
  Stage 3: Generate integration test specifications
  Stage 4: Split integration test specs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from pybastion_integration import config

TOOL_ROOT = Path(__file__).resolve().parent
STAGES_DIR = TOOL_ROOT / "stages"
ALL_STAGE_NUMS = [1, 2, 3, 4]

STAGE_NAMES = {
    1: "Build EI call graph",
    2: "Trace feature-flow cases",
    3: "Generate integration test specs",
    4: "Split integration test specs",
}


def apply_cli_overrides(args: argparse.Namespace) -> None:
    config.set_path_override("analysis_output_root", args.analysis_output_root)
    config.set_path_override("inventories_root", args.inventories_root)
    config.set_path_override("integration_output_dir", args.integration_output_dir)
    config.set_path_override("spec_split_output_dir", args.spec_split_output_dir)

    config.set_stage_override("stage1_output", args.stage1_output)
    config.set_stage_override("stage2_output", args.stage2_output)
    config.set_stage_override("stage3_output", args.stage3_output)
    config.set_stage_override("stage1_format", args.graph_format)

    config.set_graph_checker_override("script", args.graph_checker_script)
    config.set_graph_checker_override("report", args.graph_checker_report)
    config.set_graph_checker_override("summary", args.checker_summary)

    config.set_stage_override(
        "stage2_marker_inventory_output",
        args.stage2_marker_inventory_output,
    )
    config.set_stage_override(
        "stage2_branch_points_output",
        args.stage2_branch_points_output,
    )
    config.set_stage_override(
        "stage2_converge_points_output",
        args.stage2_converge_points_output,
    )


def build_paths() -> dict[str, Path]:
    return {
        "analysis_output_root": config.get_analysis_output_root(),
        "inventories_root": config.get_inventories_root(),
        "output_dir": config.get_integration_output_dir(),
        "stage1_output": config.get_stage_output(1),
        "stage2_output": config.get_stage_output(2),
        "stage3_output": config.get_stage_output(3),
        "stage2_marker_inventory_output": config.get_stage2_marker_inventory_output(),
        "stage2_branch_points_output": config.get_stage2_branch_points_output(),
        "stage2_converge_points_output": config.get_stage2_converge_points_output(),
        "spec_split_output_dir": config.get_spec_split_output_dir(),
        "graph_check_report": config.get_graph_checker_report(),
        "graph_checker_script": config.get_graph_checker_script(),
    }


def build_stage_cmd(
    stage_num: int,
    *,
    target_root: Path,
    paths: dict[str, Path],
    graph_format: str,
    verbose: bool,
    emit_all_output: bool = False,
) -> list[str]:
    stage_scripts = {
        1: STAGES_DIR / "stage1_build_call_graph.py",
        2: STAGES_DIR / "stage2_trace_feature_flows.py",
        3: STAGES_DIR / "stage3_generate_test_specs.py",
        4: STAGES_DIR / "stage4_split_specs.py",
    }

    cmd = [sys.executable, str(stage_scripts[stage_num])]

    match stage_num:
        case 1:
            cmd += [
                "--inventories-root",
                str(paths["inventories_root"]),
                "--output",
                str(paths["stage1_output"]),
                "--format",
                graph_format,
            ]
        case 2:
            cmd += [
                "--inventory-root",
                str(paths["inventories_root"]),
                "--marker-inventory-output",
                str(paths["stage2_marker_inventory_output"]),
                "--cfg",
                str(paths["stage1_output"]),
                "--branch-points-output",
                str(paths["stage2_branch_points_output"]),
                "--converge-points-output",
                str(paths["stage2_converge_points_output"]),
                "--output",
                str(paths["stage2_output"]),
            ]
        case 3:
            cmd += [
                "--target-root",
                str(target_root),
                "--inventories-root",
                str(paths["inventories_root"]),
                "--input",
                str(paths["stage1_output"]),
                "--graph-format",
                graph_format,
                "--output",
                str(paths["stage3_output"]),
            ]
            if emit_all_output:
                cmd.append("--emit-all-output")
        case 4:
            cmd += [
                "--target-root",
                str(target_root),
                "--input",
                str(paths["stage3_output"]),
                "--output-dir",
                str(paths["spec_split_output_dir"]),
            ]
        case _:
            raise ValueError(f"Unsupported stage: {stage_num}")

    if verbose:
        cmd.append("-v")

    return cmd


def build_graph_check_cmd(paths: dict[str, Path], summary: str) -> list[str]:
    return [
        sys.executable,
        str(paths["graph_checker_script"]),
        str(paths["stage1_output"]),
        "--inventories-root",
        str(paths["inventories_root"]),
        "--write-report",
        str(paths["graph_check_report"]),
        "--summary",
        summary,
    ]


def print_command(cmd: Iterable[str], *, prefix: str = "Running") -> None:
    print(f"{prefix}: {' '.join(cmd)}")


def run_command(cmd: list[str], *, dry_run: bool, verbose: bool) -> int:
    if dry_run:
        print_command(cmd, prefix="Would run")
        return 0

    if verbose:
        print_command(cmd)

    return subprocess.run(cmd).returncode


def run_stage(
    stage_num: int,
    *,
    target_root: Path,
    paths: dict[str, Path],
    graph_format: str,
    verbose: bool,
    dry_run: bool,
    emit_all_output: bool = False,
) -> int:
    print(f"\n{'=' * 70}")
    print(f"Stage {stage_num}: {STAGE_NAMES[stage_num]}")
    print(f"{'=' * 70}")

    cmd = build_stage_cmd(
        stage_num,
        target_root=target_root,
        paths=paths,
        graph_format=graph_format,
        verbose=verbose,
        emit_all_output=emit_all_output,
    )
    exit_code = run_command(cmd, dry_run=dry_run, verbose=verbose)

    if exit_code != 0:
        print(
            f"\nERROR: Stage {stage_num} failed with exit code {exit_code}",
            file=sys.stderr,
        )

    return exit_code


def run_graph_checker(
    *,
    paths: dict[str, Path],
    summary: str,
    verbose: bool,
    dry_run: bool,
) -> int:
    print(f"\n{'=' * 70}")
    print("Graph check: inventory graph consistency")
    print(f"{'=' * 70}")

    cmd = build_graph_check_cmd(paths, summary)
    exit_code = run_command(cmd, dry_run=dry_run, verbose=verbose)

    if exit_code != 0:
        print(
            f"\nERROR: Graph checker failed with exit code {exit_code}", file=sys.stderr
        )

    return exit_code


def clean_outputs(
    paths: dict[str, Path],
    *,
    stages_to_run: list[int],
    check_graph: bool,
    verbose: bool,
) -> None:
    print(f"\nCleaning selected outputs in: {paths['output_dir']}")

    files_to_remove: list[Path] = []

    if 1 in stages_to_run:
        files_to_remove.append(paths["stage1_output"])

    if check_graph and 1 in stages_to_run:
        files_to_remove.append(paths["graph_check_report"])

    if 2 in stages_to_run:
        files_to_remove.extend(
            [
                paths["stage2_output"],
                paths["stage2_marker_inventory_output"],
                paths["stage2_branch_points_output"],
                paths["stage2_converge_points_output"],
            ]
        )

    if 3 in stages_to_run:
        files_to_remove.append(paths["stage3_output"])

    if 4 in stages_to_run:
        split_dir = paths["spec_split_output_dir"]
        if split_dir.exists():
            files_to_remove.extend(split_dir.glob("*.yaml"))

    for path in files_to_remove:
        if path.exists():
            if verbose:
                print(f"  Removing: {path}")
            path.unlink()

    print("✓ Selected outputs cleaned")


def validate_prerequisites(
    *,
    paths: dict[str, Path],
    stages_to_run: list[int],
    check_graph: bool,
    verbose: bool,
) -> bool:
    errors: list[str] = []

    config_errors = config.validate_config()
    errors.extend(config_errors)

    if verbose and paths["inventories_root"].exists():
        print(f"✓ Inventories root exists: {paths['inventories_root']}")

    if 2 in stages_to_run:
        if config.get_stage1_format() != "pickle":
            errors.append(
                "Stage 2 currently requires --graph-format pickle / "
                "stages.stage1_format = 'pickle'."
            )

        if 1 not in stages_to_run and not paths["stage1_output"].exists():
            errors.append(f"Stage 2 CFG input not found: {paths['stage1_output']}")

    if 3 in stages_to_run:
        if 1 not in stages_to_run and not paths["stage1_output"].exists():
            errors.append(f"Stage 3 graph input not found: {paths['stage1_output']}")

        if 2 not in stages_to_run and not paths["stage2_output"].exists():
            errors.append(
                f"Stage 3 feature-flow input not found: {paths['stage2_output']}"
            )

    if (
        4 in stages_to_run
        and 3 not in stages_to_run
        and not paths["stage3_output"].exists()
    ):
        errors.append(f"Stage 4 input not found: {paths['stage3_output']}")

    if check_graph and not paths["graph_checker_script"].exists():
        errors.append(
            f"Graph checker script not found: {paths['graph_checker_script']}"
        )

    if check_graph and 1 not in stages_to_run and not paths["stage1_output"].exists():
        errors.append(f"Graph checker input not found: {paths['stage1_output']}")

    if errors:
        print(
            "\nERROR: Missing prerequisites or invalid configuration:", file=sys.stderr
        )
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return False

    return True


def determine_stages(args: argparse.Namespace) -> list[int]:
    if args.only is not None:
        return [args.only]

    start = args.start_from or ALL_STAGE_NUMS[0]
    stop = args.stop_at or ALL_STAGE_NUMS[-1]

    if start > stop:
        raise ValueError(
            f"--start-from ({start}) cannot be greater than --stop-at ({stop})"
        )

    return list(range(start, stop + 1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Root directory of the project to analyze",
    )
    parser.add_argument(
        "--analysis-output-root",
        type=Path,
        default=None,
        help="Override paths.analysis_output_root. Relative values resolve against --target-root.",
    )
    parser.add_argument(
        "--inventories-root",
        type=Path,
        default=None,
        help="Override paths.inventories_root. Relative values resolve against analysis_output_root.",
    )
    parser.add_argument(
        "--integration-output-dir",
        type=Path,
        default=None,
        help="Override paths.integration_output_dir. Relative values resolve against analysis_output_root.",
    )
    parser.add_argument(
        "--spec-split-output-dir",
        "--spec-output-dir",
        type=Path,
        default=None,
        help="Override paths.spec_split_output_dir. Relative values resolve against integration_output_dir.",
    )
    parser.add_argument(
        "--stage1-output",
        default=None,
        help="Override stages.stage1_output. Relative values resolve against integration_output_dir.",
    )
    parser.add_argument(
        "--stage2-output",
        default=None,
        help=(
            "Override stages.stage2_output feature-flow cases path. Relative "
            "values resolve against integration_output_dir."
        ),
    )
    parser.add_argument(
        "--stage2-marker-inventory-output",
        default=None,
        help=(
            "Override stages.stage2_marker_inventory_output. Relative values "
            "resolve against integration_output_dir."
        ),
    )
    parser.add_argument(
        "--stage2-branch-points-output",
        default=None,
        help=(
            "Override stages.stage2_branch_points_output. Relative values "
            "resolve against integration_output_dir."
        ),
    )
    parser.add_argument(
        "--stage2-converge-points-output",
        default=None,
        help=(
            "Override stages.stage2_converge_points_output. Relative values "
            "resolve against integration_output_dir."
        ),
    )
    parser.add_argument(
        "--stage3-output",
        default=None,
        help=(
            "Override stages.stage3_output integration specs path. Relative "
            "values resolve against integration_output_dir."
        ),
    )
    parser.add_argument(
        "--emit-all-output",
        action="store_true",
        help=("Emit all optional stage output YAML files to integration_output_dir."),
    )
    parser.add_argument(
        "--start-from",
        type=int,
        choices=ALL_STAGE_NUMS,
        help="Start from a specific stage and run all following selected stages",
    )
    parser.add_argument(
        "--stop-at",
        type=int,
        choices=ALL_STAGE_NUMS,
        help="Stop at a specific stage, inclusive",
    )
    parser.add_argument(
        "--only",
        type=int,
        choices=ALL_STAGE_NUMS,
        help="Run only a specific stage",
    )
    parser.add_argument(
        "--graph-format",
        choices=("pickle", "yaml"),
        default=None,
        help="Override stages.stage1_format from integration_config.toml",
    )
    parser.add_argument(
        "--check-graph",
        action="store_true",
        default=None,
        help="Run utils/check_inventory_graph.py after Stage 1",
    )
    parser.add_argument(
        "--no-check-graph",
        action="store_false",
        dest="check_graph",
        help="Disable graph checking even if enabled in config",
    )
    parser.add_argument(
        "--graph-checker-script",
        type=Path,
        default=None,
        help="Override graph_checker.script. Relative values resolve against the pybastion_integration package dir.",
    )
    parser.add_argument(
        "--graph-checker-report",
        type=Path,
        default=None,
        help="Override graph_checker.report. Relative values resolve against integration_output_dir.",
    )
    parser.add_argument(
        "--checker-summary",
        default=None,
        help="Override graph_checker.summary from integration_config.toml",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean selected outputs before running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip prerequisite validation",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved configuration and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config.set_target_root(args.target_root)
    apply_cli_overrides(args)
    target_root = config.get_target_root()

    if args.print_config:
        config.print_config_summary()
        return 0

    try:
        stages_to_run = determine_stages(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    graph_format = config.get_stage1_format()
    check_graph = (
        config.graph_checker_enabled() if args.check_graph is None else args.check_graph
    )
    checker_summary = config.get_graph_checker_summary()
    paths = build_paths()

    if args.verbose:
        print(f"Target root: {target_root}")
        print(f"Analysis output root: {paths['analysis_output_root']}")
        print(f"Inventories root: {paths['inventories_root']}")
        print(f"Integration output dir: {paths['output_dir']}")
        print(f"Split specs dir: {paths['spec_split_output_dir']}")
        print(f"Feature flow cases: {paths['stage2_output']}")
        if args.emit_all_output:
            print(
                f"Feature marker inventory: {paths['stage2_marker_inventory_output']}"
            )
            print(f"Feature branch points: {paths['stage2_branch_points_output']}")
            print(f"Feature converge points: {paths['stage2_converge_points_output']}")
        print(f"Integration specs: {paths['stage3_output']}")
        print(f"Running stages: {stages_to_run}")
        print(f"Graph format: {graph_format}")
        print(f"Graph checker enabled: {check_graph}")

    if not args.skip_validation:
        is_valid = validate_prerequisites(
            paths=paths,
            stages_to_run=stages_to_run,
            check_graph=check_graph,
            verbose=args.verbose,
        )
        if not is_valid:
            return 1

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["spec_split_output_dir"].mkdir(parents=True, exist_ok=True)

    if not args.no_clean:
        clean_outputs(
            paths,
            stages_to_run=stages_to_run,
            check_graph=check_graph,
            verbose=args.verbose,
        )

    for stage_num in stages_to_run:
        exit_code = run_stage(
            stage_num,
            target_root=target_root,
            paths=paths,
            graph_format=graph_format,
            verbose=args.verbose,
            dry_run=args.dry_run,
            emit_all_output=args.emit_all_output,
        )
        if exit_code != 0:
            return exit_code

        if stage_num == 1 and check_graph:
            exit_code = run_graph_checker(
                paths=paths,
                summary=checker_summary,
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
            if exit_code != 0:
                return exit_code

    print(f"\n{'=' * 70}")
    print("✓ Pipeline completed successfully")
    print(f"{'=' * 70}")

    if not args.dry_run:
        final_stage = max(stages_to_run)
        final_output: Path | str = {
            1: paths["stage1_output"],
            2: paths["stage2_output"],
            3: paths["stage3_output"],
            4: str(paths["spec_split_output_dir"] / "*.yaml"),
        }[final_stage]
        print(f"\nFinal output: {final_output}")

        if check_graph:
            print(f"Graph check report: {paths['graph_check_report']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
