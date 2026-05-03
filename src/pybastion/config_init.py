from __future__ import annotations

from importlib import resources
from pathlib import Path


def config_init(
    dest_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Write the unified PyBastion configuration file."""
    output_dir = (dest_dir or Path.cwd()).resolve()
    dest_path = output_dir / "pybastion_config.toml"

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        raise FileExistsError(
            f"Configuration file already exists: {dest_path}. "
            "Use --force to overwrite it."
        )

    resource = resources.files("pybastion_common").joinpath("pybastion_config.toml")

    if not resource.is_file():
        raise FileNotFoundError(
            "Configuration template not found: pybastion_common:pybastion_config.toml"
        )

    dest_path.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path