"""Policy training backends, behind the same presets and the same command line
(`aes train [--backend miles|verl] plan|run|presets`):

    miles/            Miles (DPPO, patched as in third_party/miles)
    verl/             verl v0.9.1 (DPPO, partial rollout and group filtering are native)
    bridge.py         the handshake the self-play loop drives either backend with
    carry/            the Pi episode, the recording proxy, and Miles' partial-rollout loop
    fuse_experts.py   rewrites a fine-tuned MoE checkpoint into the base layout Miles expects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    ok: bool
    returncode: int
    checkpoint: str | None
    iteration: int | None
    detail: dict[str, Any] = field(default_factory=dict)


class ConfigError(ValueError):
    """The configuration would fail or train something other than what it declares."""
