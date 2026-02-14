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

# Find the config file relative to this module
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

# Target root can be set by scripts (e.g., via --target-root argument)
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
    """
    Get the current target root (or CWD if not set).

    Returns:
        Path to target root
    """
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
    2. If relative_to_target and target_root set → target_root / config_value
    3. If relative_to_target → cwd / config_value
    4. Otherwise → relative to tool repo (module directory)

    Args:
        config_value: Path from config file
        relative_to_target: If True, resolve relative to target project; if False, relative to tool repo
        allow_absolute: If True, absolute paths are used as-is

    Returns:
        Resolved absolute path
    """
    path = Path(config_value)

    # Absolute paths pass through (if allowed)
    if allow_absolute and path.is_absolute():
        return path

    # Relative to target project
    if relative_to_target:
        return get_target_root() / path

    # Relative to tool repo
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


def get_integration_output_dir() -> Path:
    """Get the output directory for integration flow artifacts (relative to target project)."""
    return resolve_path(
        _CONFIG.get('paths', {}).get('integration_output_dir', 'dist/integration-output'),
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
        stage: Stage number (1-5)

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
    Get the expected input file for a given stage.

    Args:
        stage: Stage number (2-5)

    Returns:
        Path to expected input file (output of previous stage)
    """
    if stage < 2 or stage > 5:
        raise ValueError(f"Stage must be 2-5 (got {stage})")

    return get_stage_output(stage - 1)


# ============================================================================
# Processing Configuration
# ============================================================================

def get_max_flow_depth() -> int:
    """Get maximum flow depth for cycle prevention."""
    return _CONFIG.get('processing', {}).get('max_flow_depth', 20)


def get_min_window_length() -> int:
    """Get minimum window length for sliding windows."""
    return _CONFIG.get('processing', {}).get('min_window_length', 2)


def get_max_window_length() -> int | None:
    """Get maximum window length (None = use full flow length)."""
    max_len = _CONFIG.get('processing', {}).get('max_window_length', 0)
    return max_len if max_len > 0 else None


def boundaries_are_terminal() -> bool:
    """Check if boundary integrations should be treated as terminal nodes."""
    return _CONFIG.get('processing', {}).get('boundaries_are_terminal', True)


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
# Validation Configuration
# ============================================================================

def get_schema_path() -> Path:
    """Get path to JSON schema file (relative to tool repo)."""
    return resolve_path(
        _CONFIG.get('validation', {}).get(
            'schema_path',
            'integration/specs/integration-flow-schema.json'
        ),
        relative_to_target=False  # Schema is in tool repo, not target project
    )


def validate_outputs() -> bool:
    """Check if outputs should be validated against schema."""
    return _CONFIG.get('validation', {}).get('validate_outputs', True)


# ============================================================================
# Utility Functions
# ============================================================================

def get_pattern_analysis_max_depth() -> int:
    return _CONFIG.get('pattern_analysis', {}).get('max_depth', 40)


def get_long_flow_threshold() -> int:
    return _CONFIG.get('pattern_analysis', {}).get('long_flow_threshold', 8)


def get_pattern_analysis_output() -> Path:
    return get_integration_output_dir() / 'stage4-pattern-analysis.yaml'


def ensure_output_dir() -> None:
    """Ensure the integration output directory exists."""
    get_integration_output_dir().mkdir(parents=True, exist_ok=True)


def validate_config() -> list[str]:
    """
    Validate configuration settings.

    Returns:
        List of error messages (empty if valid)

    Raises:
        ValueError: If critical validation fails
    """
    errors = []

    # Path existence checks
    ledgers_root = get_ledgers_root()
    if not ledgers_root.exists():
        errors.append(f"Ledgers root does not exist: {ledgers_root}")

    schema_path = get_schema_path()
    if not schema_path.exists():
        errors.append(f"Schema file does not exist: {schema_path}")

    # Enum validation
    structure = get_ledger_structure()
    if structure not in {'auto', 'flat', 'package'}:
        errors.append(f"Invalid ledger_structure: {structure} (must be 'auto', 'flat', or 'package')")

    # Integer constraints
    max_depth = get_max_flow_depth()
    if max_depth < 2:
        errors.append(f"max_flow_depth must be >= 2 (got {max_depth})")

    min_window = get_min_window_length()
    if min_window < 2:
        errors.append(f"min_window_length must be >= 2 (got {min_window})")

    max_window = get_max_window_length()
    if max_window is not None and min_window > max_window:
        errors.append(f"min_window_length ({min_window}) must be <= max_window_length ({max_window})")

    verbosity = get_verbosity()
    if verbosity not in {0, 1, 2}:
        errors.append(f"verbosity must be 0, 1, or 2 (got {verbosity})")

    # YAML format validation
    yaml_width = get_yaml_width()
    if yaml_width <= 0:
        errors.append(f"yaml_width must be > 0 (got {yaml_width})")

    yaml_indent = get_yaml_indent()
    if yaml_indent <= 0:
        errors.append(f"yaml_indent must be > 0 (got {yaml_indent})")

    return errors


def print_config_summary() -> None:
    """Print a summary of current configuration."""
    print("Configuration Summary:")
    print(f"  Target root: {get_target_root()}")
    print(f"  Ledgers root: {get_ledgers_root()}")
    print(f"  Output dir: {get_integration_output_dir()}")
    print(f"  Structure: {get_ledger_structure()}")
    print(f"  Stage 1 output: {get_stage_output(1)}")
    print(f"  Max flow depth: {get_max_flow_depth()}")
    print(f"  Window length: {get_min_window_length()}-{get_max_window_length() or 'N'}")


# ============================================================================
# Automatic Validation on Import
# ============================================================================

# Validate configuration immediately when module is imported
_VALIDATION_ERRORS = validate_config()
if _VALIDATION_ERRORS:
    print("Configuration validation failed:", file=sys.stderr)
    for error in _VALIDATION_ERRORS:
        print(f"  ✗ {error}", file=sys.stderr)
    print(f"\nCheck configuration in: {_CONFIG_PATH}", file=sys.stderr)
    sys.exit(1)