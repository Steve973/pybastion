"""
Configuration loader for integration flow testing.

Loads settings from integration_config.toml (shipped with tool) and provides
convenient access to all configuration values with proper path resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Handle Python 3.11+ (tomllib in stdlib) vs earlier (tomli package)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        print(
            "ERROR: tomli package required for Python <3.11",
            file=sys.stderr
        )
        print("Install with: pip install tomli", file=sys.stderr)
        print("Run ./integration-setup.sh to install dependencies", file=sys.stderr)
        sys.exit(1)

# ============================================================================
# Configuration Loading
# ============================================================================

_MODULE_DIR = Path(__file__).parent
_CONFIG_PATH = _MODULE_DIR / "integration_config.toml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from TOML file.

    Args:
        config_path: Optional path to config file (default: integration_config.toml in tool repo)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    path = config_path or _CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected location: {_CONFIG_PATH}"
        )

    try:
        with path.open('rb') as f:
            config = tomllib.load(f)
        return config
    except Exception as e:
        raise ValueError(f"Failed to parse {path}: {e}")


# Load configuration on import
_CONFIG = load_config()

# ============================================================================
# Path Resolution
# ============================================================================

_TARGET_ROOT: Path | None = None


def set_target_root(path: Path | str | None) -> None:
    """
    Set the target project root for path resolution.

    Args:
        path: Path to target project root, or None to use CWD
    """
    global _TARGET_ROOT
    _TARGET_ROOT = Path(path).resolve() if path else None


def get_target_root() -> Path:
    """Get the current target root (or CWD if not set)."""
    return _TARGET_ROOT or Path.cwd()


def resolve_path(
        config_value: str,
        relative_to_target: bool = True,
        allow_absolute: bool = True
) -> Path:
    """
    Resolve a config path value.

    Resolution priority:
    1. If absolute path and allowed → use as-is
    2. If relative_to_target → target_root / config_value
    3. Otherwise → relative to tool repo (module directory)
    """
    path = Path(config_value)

    if allow_absolute and path.is_absolute():
        return path

    if relative_to_target:
        return get_target_root() / path

    return _MODULE_DIR / path


# ============================================================================
# Path Configuration
# ============================================================================

def get_ledgers_root() -> Path:
    """Get the root directory for ledger discovery (relative to target project)."""
    return resolve_path(
        _CONFIG.get('paths', {}).get('ledgers_root', 'dist/ledgers'),
        relative_to_target=True
    )


def get_inventories_root() -> Path:
    """Get the root directory for inventory discovery (relative to target project)."""
    return resolve_path(
        _CONFIG.get('paths', {}).get('inventories_root', 'dist/inventory'),
        relative_to_target=True
    )


def get_integration_output_dir() -> Path:
    """Get the output directory for integration flow artifacts (relative to target project)."""
    return resolve_path(
        _CONFIG.get('paths', {}).get('integration_output_dir', 'dist/integration-output'),
        relative_to_target=True
    )


def get_spec_split_output_dir() -> Path:
    """Get the output directory for split spec files (relative to target project)."""
    return resolve_path(
        _CONFIG.get('paths', {}).get('spec_split_output_dir', 'dist/integration-output/split-specs'),
        relative_to_target=True
    )

# ============================================================================
# Discovery Configuration
# ============================================================================

def get_ledger_structure() -> str:
    """Get ledger directory structure: 'auto' | 'flat' | 'package'."""
    return _CONFIG.get('discovery', {}).get('ledger_structure', 'auto')


def get_namespace_anchor() -> str | None:
    """Get optional namespace anchor to strip."""
    anchor = _CONFIG.get('discovery', {}).get('namespace_anchor', '')
    return anchor if anchor else None


# ============================================================================
# Stage Output Files
# ============================================================================

def get_stage_output(stage: int) -> Path:
    """
    Get the output file path for a given stage.

    Args:
        stage: Stage number (1-4)

    Returns:
        Full path to stage output file (in target project)
    """
    output_dir = get_integration_output_dir()
    stages = _CONFIG.get('stages', {})

    filename_key = f'stage{stage}_output'
    filename = stages.get(filename_key, f'stage{stage}-output.yaml')

    return output_dir / filename


def get_stage_input(stage: int) -> Path:
    """
    Get the expected input file for a given stage (output of previous stage).

    Args:
        stage: Stage number (2-4)

    Returns:
        Path to expected input file
    """
    if stage < 2 or stage > 4:
        raise ValueError(f"Stage must be 2-4 (got {stage})")

    return get_stage_output(stage - 1)


# ============================================================================
# Output Format Configuration
# ============================================================================

def get_yaml_width() -> int:
    """Get YAML line width."""
    return _CONFIG.get('output_format', {}).get('yaml_width', 100)


def get_yaml_indent() -> int:
    """Get YAML indentation."""
    return _CONFIG.get('output_format', {}).get('yaml_indent', 2)


def get_yaml_sort_keys() -> bool:
    """Check if YAML keys should be sorted."""
    return _CONFIG.get('output_format', {}).get('yaml_sort_keys', False)


def include_metadata() -> bool:
    """Check if metadata sections should be included."""
    return _CONFIG.get('output_format', {}).get('include_metadata', True)


def debug_output() -> bool:
    """Check if debug information should be included."""
    return _CONFIG.get('output_format', {}).get('debug_output', False)


# ============================================================================
# Logging Configuration
# ============================================================================

def get_verbosity() -> int:
    """Get verbosity level: 0 (quiet) | 1 (normal) | 2 (verbose)."""
    return _CONFIG.get('logging', {}).get('verbosity', 1)


def show_progress() -> bool:
    """Check if progress should be displayed."""
    return _CONFIG.get('logging', {}).get('show_progress', True)


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_output_dir() -> None:
    """Ensure the integration output directory exists."""
    get_integration_output_dir().mkdir(parents=True, exist_ok=True)


def validate_config() -> list[str]:
    """
    Validate configuration settings.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if not get_ledgers_root().exists():
        errors.append(f"Ledgers root does not exist: {get_ledgers_root()}")

    if not get_inventories_root().exists():
        errors.append(f"Inventories root does not exist: {get_inventories_root()}")

    structure = get_ledger_structure()
    if structure not in {'auto', 'flat', 'package'}:
        errors.append(f"Invalid ledger_structure: {structure} (must be 'auto', 'flat', or 'package')")

    verbosity = get_verbosity()
    if verbosity not in {0, 1, 2}:
        errors.append(f"verbosity must be 0, 1, or 2 (got {verbosity})")

    if get_yaml_width() <= 0:
        errors.append(f"yaml_width must be > 0 (got {get_yaml_width()})")

    if get_yaml_indent() <= 0:
        errors.append(f"yaml_indent must be > 0 (got {get_yaml_indent()})")

    return errors


def print_config_summary() -> None:
    """Print a summary of current configuration."""
    print("Configuration Summary:")
    print(f"  Target root:      {get_target_root()}")
    print(f"  Ledgers root:     {get_ledgers_root()}")
    print(f"  Inventories root: {get_inventories_root()}")
    print(f"  Output dir:       {get_integration_output_dir()}")
    print(f"  Structure:        {get_ledger_structure()}")
    print(f"  Stage 1 output:   {get_stage_output(1)}")
    print(f"  Stage 2 output:   {get_stage_output(2)}")
    print(f"  Stage 3 output:   {get_stage_output(3)}")


# ============================================================================
# Automatic Validation on Import
# ============================================================================
def run_validation() -> None:
    validation_errors = validate_config()
    if validation_errors:
        print("Configuration validation failed:", file=sys.stderr)
        for error in validation_errors:
            print(f"  ✗ {error}", file=sys.stderr)
        print(f"\nCheck configuration in: {_CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
