# PyBastion

PyBastion is an experimental Python analysis toolkit for producing structured
unit and integration testing artifacts that can be consumed by developers or AI
agents.

It analyzes Python source code into stable intermediate artifacts, including
unit inventories, execution items, callable structure, integration seams,
execution paths, and split integration test specifications.

PyBastion is currently alpha software. It is intended for experimentation,
inspection, and agent-assisted test generation workflows, not as a polished
one-command test generator.

## Packages

This repository is organized as a small monorepo:

- `pybastion-common`: shared helpers and common analysis support
- `pybastion-unit`: source analysis and unit inventory generation
- `pybastion-integration`: integration seam analysis and integration test
  specification generation

## What PyBastion Produces

The unit pipeline analyzes your Python project source code and writes unit-level
artifacts under `dist/pybastion/` by default.

The integration pipeline consumes the generated unit inventory and writes
integration analysis artifacts under `dist/pybastion/integration-output/` by
default.

The expected flow is:

1. Run the unit analysis pipeline.
2. Review or optionally run readiness findings.
3. Run the integration analysis pipeline.
4. Use the generated specs as input for human or AI-assisted test generation.

## Unit Analysis

The unit analysis pipeline inspects Python source files and generates structured
inventory artifacts. These artifacts model source units, callables, execution
items, branches, and integration point metadata.

Run the full unit pipeline:

    poetry run pybastion-unit /path/to/project -v

Run the full unit pipeline with readiness scanning first:

    poetry run pybastion-unit /path/to/project --readiness -v

The readiness scan is a preflight diagnostic. It looks for source patterns that
may reduce analysis quality, such as unresolved receiver types, dynamic
dispatch, broad Any annotations, and opaque branch conditions.

## Integration Analysis

The integration pipeline consumes the unit inventory and builds integration seam
test specifications.

Run the full integration pipeline:

    poetry run pybastion-integration --target-root /path/to/project --check-graph -v

The pipeline currently performs:

1. Build an execution instance call graph from unit inventories.
2. Optionally check the generated graph against the inventory structure.
3. Generate integration seam test specifications.
4. Split specs by pairing the source unit to its target unit.

## Typical Output Layout

After running both pipelines, a target project will usually contain:

    dist/pybastion/
      inspect/
      eis/
      inventory/
      logs/
      integration-output/
        stage1-ei-cfg.pkl
        inventory-graph-report.yaml
        stage2-integration-test-specs.yaml
        specs/

## Current Status

PyBastion is under active development.

The current focus is producing accurate, inspectable analysis artifacts that can
support test generation. The tool is especially focused on making execution
behavior, integration seams, and test relevant paths explicit enough for a
developer or AI agent to reason about them.

Expect breaking changes. The internal artifact schemas, config files, and CLI
behavior may change while the project stabilizes.

## Development Setup

Clone the repository:

    git clone https://github.com/Steve973/pybastion.git
    cd pybastion

Install the package you want to work with using Poetry from the package
directory.

For unit analysis:

    cd pybastion-unit
    poetry install

For integration analysis:

    cd pybastion-integration
    poetry install

## License

MIT