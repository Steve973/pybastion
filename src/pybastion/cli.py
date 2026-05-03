from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import sys

from pybastion.config_init import config_init


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
        "--config",
        type=Path,
        default=None,
        help="Combined namespaced PyBastion config file.",
    )
    all_cmd.add_argument(
        "--readiness",
        action="store_true",
        help="Run unit readiness preflight before unit analysis.",
    )
    all_cmd.add_argument(
        "--check-graph",
        action="store_true",
        help="Run integration graph checker after integration Stage 1.",
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

    config = subcommands.add_parser(
        "config",
        help="Manage PyBastion configuration files.",
    )
    config_subcommands = config.add_subparsers(
        dest="config_command",
        metavar="COMMAND",
    )

    config_init_parser = config_subcommands.add_parser(
        "init",
        help="Write default PyBastion configuration files.",
    )
    config_init_parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Directory where config files should be written. Defaults to the current directory.",
    )
    config_init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config files.",
    )

    help_cmd = subcommands.add_parser(
        "help",
        help="Show help for pybastion or a subcommand.",
    )
    help_cmd.add_argument(
        "topic",
        nargs="?",
        choices=("unit", "integration", "all", "config"),
        help="Optional command to show help for.",
    )

    return parser


def run_unit(args: Sequence[str]) -> int:
    from pybastion_unit.run_project_analysis import main as unit_main

    return unit_main(list(args))


def run_integration(args: Sequence[str]) -> int:
    from pybastion_integration.run_integration_analysis import main as integration_main

    return integration_main(list(args))


def run_all(args: argparse.Namespace) -> int:
    unit_config = args.unit_config or args.config
    integration_config = args.integration_config or args.config

    unit_args = [args.project_root]

    if unit_config:
        unit_args.extend(["--config", str(unit_config)])
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

    if integration_config:
        integration_args.extend(["--config", str(integration_config)])
    if args.check_graph:
        integration_args.append("--check-graph")
    if args.verbose:
        integration_args.append("-v")
    if args.no_clean:
        integration_args.append("--no-clean")
    if args.dry_run:
        integration_args.append("--dry-run")

    return run_integration(integration_args)


def run_config(args: argparse.Namespace) -> int:
    match args.config_command:
        case "init":
            try:
                written = config_init(
                    dest_dir=args.dest_dir,
                    force=args.force,
                )
            except FileExistsError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2

            print(f"Wrote {written}")

            return 0

        case _:
            print("Usage: pybastion config init [unit|integration|all] [--dest-dir DIR] [--force]")
            return 2


def print_topic_help(topic: str | None) -> int:
    parser = build_parser()

    if topic is None:
        parser.print_help()
        return 0

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparser = action.choices.get(topic)
            if subparser is not None:
                subparser.print_help()
                return 0

    parser.print_help()
    return 2


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
        case "config":
            return run_config(parsed)
        case "help":
            return print_topic_help(parsed.topic)
        case _:
            parser.print_help()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
