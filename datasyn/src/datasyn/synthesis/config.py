"""
YAML configuration helpers for the synthesis runner.

Configs are self-contained YAML files (no bundled default, no merge). Path
values may reference environment variables as ``${VAR}``; these are expanded at
load time from the process environment and from a repo-root ``.env`` file, so
machine-specific data roots live in ``.env`` while the YAML stays shareable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_run_config(config_path: Path | str) -> dict[str, Any]:
    """
    Load a single standalone YAML config, expanding ``${VAR}`` in strings.

    ``config_path`` is required: there is no bundled default. Environment
    variables (and a repo-root ``.env``) supply machine-specific path roots.
    """
    _load_dotenv()
    data = _load_yaml(Path(config_path))
    return _expand_env(data)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load synthesis YAML configs. "
            "Install it in the active project environment or use an environment "
            "where `import yaml` works."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")
    return data


def _load_dotenv() -> None:
    """
    Load ``.env`` from the current working directory into ``os.environ``.

    Minimal parser: one ``KEY=VALUE`` per line, ``#`` comments, optional
    ``export`` prefix and surrounding quotes. Real environment variables win —
    ``.env`` only fills keys that are otherwise unset.
    """
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, val = line.partition("=")
        if not sep:
            continue
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _expand_env(obj: Any) -> Any:
    """
    Recursively expand ``${VAR}`` in every string leaf of ``obj``.

    Raises if any ``${...}`` placeholder is left unresolved, so a missing env
    var fails loudly instead of silently producing a bogus literal path.
    """
    if isinstance(obj, str):
        expanded = os.path.expandvars(obj)
        if "${" in expanded:
            raise ValueError(
                f"Unresolved environment variable in config value: {obj!r}. "
                "Set it in the environment or in a repo-root .env file."
            )
        return expanded
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj
