"""verl's PPO entry point with AutoEnvScaling's trainer registered.

    python -m autoenvscaling.training.verl.main <Hydra overrides>    # rendered by VerlConfig

This is `python -m verl.trainer.main_ppo` with one difference: the Ray task runner imports
`training.verl.bridge` before verl looks its trainer up, so `trainer.v1.trainer_mode=aes_bridge`
resolves. The agent loops are registered by `actor_rollout_ref.rollout.agent.
agent_loop_manager_class`, which verl imports itself.

A Ray instance started here hands the caller's environment (AES_* settings, Modal and W&B
credentials) to every worker; a cluster started beforehand must have been started with it.
"""

from __future__ import annotations

import hydra
import ray
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@ray.remote
class TaskRunner(TaskRunnerV1.__ray_actor_class__):
    def run(self, config):
        import autoenvscaling.training.verl.bridge  # noqa: F401  registers the "aes_bridge" trainer
        return super().run(config)


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
