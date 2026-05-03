# pybastion-unit

`pybastion-unit` runs PyBastion's unit analysis pipeline.

It analyzes Python source code and produces the unit-level artifacts used by
the rest of PyBastion. These artifacts include inspected units, execution item
files, callable inventories, and readiness findings.

## Command

Run unit analysis:

    pybastion-unit /path/to/project -v

Run unit analysis with readiness scanning:

    pybastion-unit /path/to/project --readiness -v

Using the unified PyBastion config:

    pybastion-unit /path/to/project \
      --config pybastion_config.toml \
      --readiness \
      -v

The same pipeline can also be run through the root CLI:

    pybastion unit /path/to/project \
      --config pybastion_config.toml \
      --readiness \
      -v

## Pipeline

The unit pipeline currently runs three stages:

1. Inspect units
2. Enumerate execution items
3. Enumerate callables and inventory artifacts

The readiness scanner is an optional preflight step. It reports source
patterns that may reduce PyBastion's ability to infer receiver types, model
execution paths, or identify useful integration seams.

## Outputs

By default, unit analysis writes under:

    dist/pybastion/

Typical outputs include:

    dist/pybastion/
      inspect/
      eis/
      inventory/
      logs/

These artifacts are consumed by `pybastion-integration` and can also be
inspected directly by developers or coding agents.

## Configuration

`pybastion-unit` can run with internal defaults or with the unified PyBastion
config file.

Generate the config from the root CLI:

    pybastion config init

Then pass it to the unit pipeline:

    pybastion-unit /path/to/project \
      --config pybastion_config.toml \
      --readiness \
      -v

The unit pipeline reads the `[unit]` section of the unified config.

## Relationship to pybastion-integration

`pybastion-unit` produces the inventory artifacts consumed by
`pybastion-integration`.

A typical workflow is:

1. Run `pybastion-unit` to generate unit inventory artifacts.
2. Run `pybastion-integration` to build integration seam specifications.
3. Use the generated artifacts for review or agent-assisted test generation.

## Documentation

For the detailed pipeline model, see
[Unit Analysis Pipeline](../docs/unit/unit-analysis-pipeline.md).

For deterministic ID behavior, see
[Deterministic IDs](../docs/unit/branch_ids.md).

For guidance on generating unit tests from PyBastion artifacts, see
[Unit Testing Agent Contract](../docs/unit/unit-testing-agent-contract.md).

For full CLI documentation, see
[PyBastion CLI](../docs/common/pybastion-cli.md).
