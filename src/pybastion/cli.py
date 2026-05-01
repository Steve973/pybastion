from __future__ import annotations

import argparse
from collections.abc import Sequence

from pybastion_integration.run_integration_analysis import main as integration_main
from pybastion_unit.run_project_analysis import main as unit_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pybastion",
        description="Run PyBastion unit and integration analysis pipelines.",
    )

    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")

    unit = subcommands.add_parser(
        "unit",
        help="Run the unit analysis pipeline.",
    )
    unit.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to pybastion-unit.",
    )

    integration = subcommands.add_parser(
        "integration",
        help="Run the integration analysis pipeline.",
    )
    integration.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to pybastion-integration.",
    )

    all_cmd = subcommands.add_parser(
        "all",
        help="Run unit analysis, then integration analysis.",
    )
    all_cmd.add_argument(
        "project_root",
        help="Root of the target project to analyze.",
    )
    all_cmd.add_argument(
        "--readiness",
        action="store_true",
        help="Run unit readiness preflight before unit analysis.",
    )
    all_cmd.add_argument(
        "--check-graph",
        action="store_true",
        help="Run integration graph checker after Stage 1.",
    )
    all_cmd.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output for both pipelines.",
    )
    all_cmd.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean generated outputs before running.",
    )
    all_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing pipeline subprocesses.",
    )

    return parser


def run_unit(args: Sequence[str]) -> int:
    return unit_main(list(args))


def run_integration(args: Sequence[str]) -> int:
    return integration_main(list(args))


def run_all(args: argparse.Namespace) -> int:
    unit_args = [args.project_root]

    if args.readiness:
        unit_args.append("--readiness")
    if args.verbose:
        unit_args.append("-v")
    if args.no_clean:
        unit_args.append("--no-clean")
    if args.dry_run:
        unit_args.append("--dry-run")

    unit_rc = run_unit(unit_args)
    if unit_rc != 0:
        return unit_rc

    integration_args = [
        "--target-root",
        args.project_root,
    ]

    if args.check_graph:
        integration_args.append("--check-graph")
    if args.verbose:
        integration_args.append("-v")
    if args.no_clean:
        integration_args.append("--no-clean")
    if args.dry_run:
        integration_args.append("--dry-run")

    return run_integration(integration_args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(list(argv) if argv is not None else None)

    match parsed.command:
        case "unit":
            return run_unit(parsed.args)
        case "integration":
            return run_integration(parsed.args)
        case "all":
            return run_all(parsed)
        case _:
            parser.print_help()
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
