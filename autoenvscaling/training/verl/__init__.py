"""Training with verl v0.9.1 (github.com/verl-project/verl), the same presets and command line
as Miles (`aes train --backend verl plan|run|presets`).

    config.py     VerlConfig and presets; the paper's settings are the defaults
    launcher.py   command and checkpoint discovery for a verl job
    main.py       verl's PPO entry point with AutoEnvScaling's trainer registered
    bridge.py     bridge mode: verl's colocated trainer fed by the self-play loop
    agent.py      the agent loops (Pi episodes, bridge replay) and the recording-proxy host
    episode.py    one Pi episode in a sandbox, as verl runs it in carry mode
    data.py       conversations as training tokens (bridge batches, SFT rows)

DPPO, group-mean advantages, partial rollout across weight updates, zero-variance group
filtering and the staleness bound are verl's own, so no patch is needed. Only config, launcher
and data import nothing from verl.
"""

from autoenvscaling.training.verl.config import BRIDGE, CARRY, PRESETS, SFT, VerlConfig
from autoenvscaling.training.verl.launcher import VerlLauncher, latest_checkpoint

__all__ = ["BRIDGE", "CARRY", "PRESETS", "SFT", "VerlConfig", "VerlLauncher", "latest_checkpoint"]
