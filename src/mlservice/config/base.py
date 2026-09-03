"""Shared plumbing for file-backed typed configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="YamlConfig")


class YamlConfig(BaseModel):
    """A configuration section that is loaded from a YAML document.

    Extra keys are rejected: a typo in a config file should fail loudly at load
    time rather than silently fall back to a default.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        """Load and validate this config section from a YAML file."""
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(
                f"Config file {resolved} must contain a YAML mapping, got {type(raw).__name__}"
            )
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation, suitable for embedding in artifact metadata."""
        return self.model_dump(mode="json")
