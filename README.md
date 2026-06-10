# PyBastion

PyBastion is an experimental Python analysis toolkit for producing structured,
inspectable artifacts for unit and integration test generation.

It is built around a modeling-first approach: instead of asking an AI agent to
generate tests directly from raw source code, PyBastion decomposes a project
into unit inventories, execution items, callable artifacts, integration seams,
call graphs, and test specifications that can be reviewed by developers or
used by coding agents to generate tests.

## Current Status

PyBastion is alpha software. It is currently under active development, and it
will change while the concepts are validated and solidified.

The current focus is producing accurate, inspectable analysis artifacts that
can support developer review and agent-assisted test generation. Artifact
formats, config options, and command behavior may change while the project
stabilizes.

## Packages

This repository is organized as a small monorepo:

- `pybastion`: root CLI package for running the unit and integration pipelines
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
2. Optionally review readiness findings.
3. Run the integration analysis pipeline.
4. Use the generated specs as input for human or AI-assisted test generation.

## Unit Analysis

The unit analysis pipeline inspects Python source files and generates structured
inventory artifacts. These artifacts model source units, callables, execution
items, control flow, and integration point metadata.

## Integration Analysis

The integration pipeline consumes the unit inventory and builds integration seam
test specifications. The pipeline builds an execution instance call graph from
unit inventories. Optionally, it can check the generated graph against the
inventory structure. It will generate integration seam test specifications, then
split specs by pairing the source unit to its target unit.

## Getting Started

Clone the repository:

    git clone https://github.com/Steve973/pybastion.git
    cd pybastion
    poetry install

## Running PyBastion

Run the full unit and integration analysis pipeline:

    pybastion all /path/to/project \
      --readiness \
      --check-graph \
      -v

Run only the unit analysis:

    pybastion unit /path/to/project \
      --readiness \
      -v

Run only the integration analysis:

    pybastion integration \
      --target-root /path/to/project \
      --check-graph \
      -v

The readiness scan is a preflight diagnostic. It looks for source patterns that
may reduce analysis quality, such as unresolved receiver types, dynamic
dispatch, broad Any annotations, and opaque branch conditions.

## Typical Output Layout

After running both pipelines, a target project will usually contain:

    dist/pybastion/
      inspect/
      eis/
      inventory/
      integration-output/
        stage1-ei-cfg.pkl
        inventory-graph-report.yaml
        stage2-integration-test-specs.yaml
        specs/
      logs/

## Documentation

### Package READMEs

- [`pybastion-unit`](pybastion-unit/README.md)
- [`pybastion-integration`](pybastion-integration/README.md)

### Deep documentation

- [Unit Analysis Pipeline](docs/unit/unit-analysis-pipeline.md)
- [Deterministic IDs](docs/unit/branch-ids.md)
- [Unit Testing Agent Contract](docs/unit/unit-testing-agent-contract.md)
- [Integration Analysis Pipeline](docs/integration/integration-analysis-pipeline.md)
- [Project Analysis Markers](docs/integration/project-analysis-markers.md)
- [PyBastion CLI](docs/common/pybastion-cli.md)

## License

MIT