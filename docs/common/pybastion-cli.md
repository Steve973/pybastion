# PyBastion CLI

The `pybastion` command is the primary entry point for running PyBastion
analysis pipelines.

It provides commands for running the full analysis workflow, running only the
unit pipeline, running only the integration pipeline, generating configuration,
and viewing help.

## Table of Contents

- [Overview](#overview)
- [Commands](#commands)
  - [`pybastion all`](#pybastion-all)
  - [`pybastion unit`](#pybastion-unit)
  - [`pybastion integration`](#pybastion-integration)
  - [`pybastion config init`](#pybastion-config-init)
  - [`pybastion help`](#pybastion-help)
- [Common Options](#common-options)
  - [`--config`](#config)
  - [`--readiness`](#readiness)
  - [`--check-graph`](#check-graph)
  - [`-v`, `--verbose`](#-v---verbose)
  - [`--no-clean`](#no-clean)
  - [`--dry-run`](#dry-run)
- [Package-Specific Commands](#package-specific-commands)
- [Typical Workflows](#typical-workflows)

## Overview

The root `pybastion` command delegates to the unit and integration analysis
pipelines.

The main commands are:

- `pybastion all`
- `pybastion unit`
- `pybastion integration`
- `pybastion config init`
- `pybastion help`

For normal use, prefer the root `pybastion` command. The package-specific
commands remain available for working directly with one pipeline.

## Commands

### `pybastion all`

Runs the unit analysis pipeline, then the integration analysis pipeline.

```bash
pybastion all /path/to/project \
  --readiness \
  --check-graph \
  -v
```

With a unified config file:

```bash
pybastion all /path/to/project \
  --config pybastion_config.toml \
  --readiness \
  --check-graph \
  -v
```

The `all` command passes the project root to the unit pipeline first. If the
unit pipeline succeeds, it then runs the integration pipeline against the same
project root.

### `pybastion unit`

Runs only the unit analysis pipeline.

```bash
pybastion unit /path/to/project \
  --readiness \
  -v
```

With a unified config file:

```bash
pybastion unit /path/to/project \
  --config pybastion_config.toml \
  --readiness \
  -v
```

Arguments after `pybastion unit` are passed through to the unit pipeline.

### `pybastion integration`

Runs only the integration analysis pipeline.

```bash
pybastion integration \
  --target-root /path/to/project \
  --check-graph \
  -v
```

With a unified config file:

```bash
pybastion integration \
  --target-root /path/to/project \
  --config pybastion_config.toml \
  --check-graph \
  -v
```

Arguments after `pybastion integration` are passed through to the integration
pipeline.

### `pybastion config init`

Writes the default unified PyBastion configuration file.

```bash
pybastion config init
```

Write the config file to a specific directory:

```bash
pybastion config init --dest-dir ./config
```

Overwrite an existing generated config file:

```bash
pybastion config init --force
```

The generated config file is namespaced by pipeline. The `[unit.*]` sections
control unit analysis, and the `[integration.*]` sections control integration
analysis.

### `pybastion help`

Shows help for the root command or for a specific command.

```bash
pybastion help
```

Show help for a specific command:

```bash
pybastion help all
pybastion help unit
pybastion help integration
pybastion help config
```

The standard argparse help form also works:

```bash
pybastion --help
pybastion all --help
pybastion config init --help
```

## Common Options

### `--config`

Passes a unified PyBastion config file.

```bash
pybastion all /path/to/project \
  --config pybastion_config.toml
```

For `pybastion all`, the same config file is passed to both the unit and
integration pipelines.

### `--readiness`

Runs the unit readiness preflight before unit analysis.

```bash
pybastion all /path/to/project --readiness
```

The readiness scanner reports source patterns that may reduce analysis quality,
such as unresolved receiver types, broad `Any` annotations, dynamic dispatch,
and opaque branch conditions.

### `--check-graph`

Runs the integration graph checker after integration Stage 1.

```bash
pybastion all /path/to/project --check-graph
```

This validates the generated execution instance call graph against the unit
inventory structure.

### `-v`, `--verbose`

Prints commands as they run.

```bash
pybastion all /path/to/project -v
```

### `--no-clean`

Prevents selected generated outputs from being cleaned before running.

```bash
pybastion all /path/to/project --no-clean
```

### `--dry-run`

Prints the commands that would run without executing pipeline subprocesses.

```bash
pybastion all /path/to/project --dry-run
```

## Package-Specific Commands

The root CLI delegates to package-specific commands.

Run the unit pipeline directly:

```bash
pybastion-unit /path/to/project \
  --readiness \
  -v
```

Run the integration pipeline directly:

```bash
pybastion-integration \
  --target-root /path/to/project \
  --check-graph \
  -v
```

These commands are useful when working directly inside one subproject or when
debugging one pipeline in isolation.

## Typical Workflows

Run everything with defaults:

```bash
pybastion all /path/to/project \
  --readiness \
  --check-graph \
  -v
```

Generate a config, then run everything with that config:

```bash
pybastion config init

pybastion all /path/to/project \
  --config pybastion_config.toml \
  --readiness \
  --check-graph \
  -v
```

Run only unit analysis:

```bash
pybastion unit /path/to/project \
  --readiness \
  -v
```

Run only integration analysis after unit inventory already exists:

```bash
pybastion integration \
  --target-root /path/to/project \
  --check-graph \
  -v
```