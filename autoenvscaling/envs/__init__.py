"""Multi-player text games for RL, played through TextArena.

Importing this package registers its games with ``textarena.make``::

    import textarena as ta
    import autoenvscaling.envs  # noqa: F401

    env = ta.make("UnstableUnicorns-v0")
    env.reset(num_players=4, seed=0)

Each id also comes as ``<id>-train`` (prompt, game log and current board) and ``<id>-raw``
(unwrapped), following TextArena's conventions.
"""

from textarena.envs.registration import register_with_versions
from textarena.wrappers import (
    ActionFormattingWrapper,
    GameMessagesAndCurrentBoardObservationWrapper,
    LLMObservationWrapper,
)

WRAPPERS = {"default": [LLMObservationWrapper, ActionFormattingWrapper],
            "-train": [GameMessagesAndCurrentBoardObservationWrapper, ActionFormattingWrapper]}

for env_id, kwargs in {"UnstableUnicorns-v0": {}, "UnstableUnicorns-v0-short": {"goal": 5}}.items():
    register_with_versions(id=env_id, wrappers=WRAPPERS, **kwargs,
                           entry_point="autoenvscaling.envs.unstable_unicorns.env:UnstableUnicornsEnv")
