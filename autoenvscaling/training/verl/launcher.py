"""Turn a `VerlConfig` into a command, run it, and read back what it produced.

A subprocess boundary, like the Miles launcher: nothing here imports verl, and `plan()` returns
the exact command so a run that costs GPU-days can be checked before it starts. Run it in an
environment with verl v0.9.1 and this package installed (`third_party/verl/README.md`).

RL jobs start `python -m autoenvscaling.training.verl.main`, which starts Ray itself and forwards
the `AES_*` and `MODAL_*` variables to the workers; SFT jobs start `torchrun -m
verl.trainer.sft_trainer`. Multi-node SFT reads NODE_RANK, MASTER_ADDR and MASTER_PORT.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from autoenvscaling.training import ConfigError, TrainingResult
from autoenvscaling.training.bridge import STOP_FILE
from autoenvscaling.training.verl.config import BRIDGE, CARRY, SFT, VerlConfig

LATEST_FILE = "latest_checkpointed_iteration.txt"


def latest_checkpoint(save_dir: Path) -> tuple[int, Path] | None:
    """The newest checkpoint verl recorded under `save_dir`, as (step, directory)."""
    save_dir = Path(save_dir)
    marker = save_dir / LATEST_FILE
    if marker.exists() and (text := marker.read_text().strip()).isdigit():
        return int(text), save_dir / f"global_step_{text}"
    found = [(int(m.group(1)), d) for d in save_dir.glob("global_step_*")
             if d.is_dir() and (m := re.fullmatch(r"global_step_(\d+)", d.name))]
    return max(found) if found else None


class VerlLauncher:
    def __init__(self, *, python: str | None = None, env: dict[str, str] | None = None):
        self.env = dict(os.environ if env is None else env)
        self.python = python or sys.executable

    def command(self, config: VerlConfig) -> list[str]:
        if config.mode != SFT:
            return [self.python, "-m", "autoenvscaling.training.verl.main", *config.overrides()]
        nodes = (["--standalone"] if config.num_nodes == 1 else
                 [f"--node-rank={self.env.get('NODE_RANK', '0')}",
                  f"--master-addr={self.env.get('MASTER_ADDR', '127.0.0.1')}",
                  f"--master-port={self.env.get('MASTER_PORT', '29500')}"])
        return ["torchrun", f"--nnodes={config.num_nodes}", f"--nproc-per-node={config.gpus_per_node}",
                *nodes, "-m", "verl.trainer.sft_trainer", *config.overrides()]

    def problems(self, config: VerlConfig) -> list[str]:
        out = list(config.problems())
        if config.mode == CARRY:
            from autoenvscaling.training.carry.rollout import CarrySettings
            out += [f"carry: {p}" for p in
                    CarrySettings.from_env(self.env).problems(config.n_samples_per_prompt)]
        return out

    def plan(self, config: VerlConfig) -> str:
        return shlex.join(self.command(config))

    def run(self, config: VerlConfig, *, log_path: Path | None = None) -> TrainingResult:
        if problems := self.problems(config):
            raise ConfigError("; ".join(problems))
        Path(config.save).mkdir(parents=True, exist_ok=True)
        if log_path is None:
            proc = subprocess.run(self.command(config), env=self.env, check=False)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log:
                proc = subprocess.run(self.command(config), env=self.env, stdout=log,
                                      stderr=subprocess.STDOUT, check=False)
        found = latest_checkpoint(Path(config.save))
        stopped = config.mode == BRIDGE and (Path(config.bridge_dir) / STOP_FILE).exists()
        return TrainingResult(ok=found is not None and proc.returncode == 0, returncode=proc.returncode,
                              checkpoint=str(found[1]) if found else None,
                              iteration=found[0] if found else None,
                              detail={"num_rollout": config.num_rollout, "stopped_by_loop": stopped,
                                      "command": self.plan(config)})
