"""Load config.yaml, the single source of truth for which columns get embedded.

`config.yaml` is looked up in this order:
  1. an explicit path (``--config-path``)
  2. ``config.yaml`` packaged alongside this module (how jobs read it; see the
     ``force-include`` entry in pyproject.toml)
  3. the repo root, walking up from this file (how local tests read it)
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

CONFIG_FILENAME = "config.yaml"


def _candidate_paths() -> list[pathlib.Path]:
    here = pathlib.Path(__file__).resolve()
    candidates = [here.parent / CONFIG_FILENAME]
    candidates.extend(parent / CONFIG_FILENAME for parent in here.parents)
    return candidates


def find_config() -> pathlib.Path:
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(c) for c in _candidate_paths())
    raise FileNotFoundError(f"could not locate {CONFIG_FILENAME}; searched:\n  {searched}")


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Parse config.yaml into a dict."""
    resolved = pathlib.Path(path) if path else find_config()
    if not resolved.is_file():
        raise FileNotFoundError(f"config not found: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"{resolved} must parse to a mapping, got {type(loaded).__name__}")
    return loaded


def ai_search_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the ``ai_search`` section into a single mapping.

    ``ai_search`` is authored as a *list of single-key mappings* rather than a
    plain mapping, so it needs flattening before use:

        ai_search:
          - embedding_cols: [...]
          - label_col: PS_PRODUCT_ID
    """
    section = cfg.get("ai_search")
    if section is None:
        raise KeyError("config.yaml has no 'ai_search' section")

    if isinstance(section, dict):
        # Tolerate a plain mapping in case the shape is ever simplified.
        return dict(section)

    if not isinstance(section, list):
        raise TypeError(f"'ai_search' must be a list or mapping, got {type(section).__name__}")

    flat: dict[str, Any] = {}
    for entry in section:
        if not isinstance(entry, dict):
            raise TypeError(f"every 'ai_search' entry must be a mapping, got {type(entry).__name__}: {entry!r}")
        for key, value in entry.items():
            if key in flat:
                raise ValueError(f"duplicate key {key!r} in 'ai_search'")
            flat[key] = value
    return flat


def embedding_cols(cfg: dict[str, Any]) -> list[str]:
    """Columns concatenated into ``embed_text``. Order is preserved."""
    cols = ai_search_cfg(cfg).get("embedding_cols")
    if not cols:
        raise KeyError("ai_search.embedding_cols is missing or empty")
    if not isinstance(cols, list):
        raise TypeError(f"embedding_cols must be a list, got {type(cols).__name__}")
    return list(cols)


def filter_cols(cfg: dict[str, Any]) -> list[str]:
    """Low-cardinality columns carried for metadata filtering, not embedded."""
    cols = ai_search_cfg(cfg).get("filter_cols") or []
    if not isinstance(cols, list):
        raise TypeError(f"filter_cols must be a list, got {type(cols).__name__}")
    return list(cols)


def require(cfg: dict[str, Any], key: str) -> Any:
    """Read a required key out of the flattened ``ai_search`` section."""
    flat = ai_search_cfg(cfg)
    if key not in flat:
        raise KeyError(f"ai_search.{key} is missing (have: {sorted(flat)})")
    return flat[key]


def null_sentinels(cfg: dict[str, Any]) -> list[str]:
    """Literal strings that mean "missing" but are not empty."""
    return list((cfg.get("data_quality") or {}).get("null_sentinels") or [])
