"""Bridge mode on verl: the self-play loop runs every episode, verl trains on its batches.

`BridgeTrainer` (`trainer.v1.trainer_mode=aes_bridge`) is verl's colocated V1 trainer with the
dataset replaced by the handshake in `training.bridge`. Before each update it announces an
OpenAI-compatible address for the current weights (`OpenAIFront`, over verl's SGLang replicas)
and waits for the loop's batch; the batch's groups are then submitted as prompts whose agent
loop (`aes_replay`, `training.verl.agent`) re-tokenizes each conversation, so verl's own code
builds every training field. One batch is one optimizer step whatever its size (Miles'
`--use-dynamic-global-batch-size`), and each sample trains on the advantage the loop computed
over its whole group. The behaviour log-probabilities are recomputed by the actor, as with
Miles, because the conversations are re-tokenized.

Between writing a batch and the next announcement the replicas are paused, not only asleep: a
request the loop sends late waits for the new weights instead of reaching a released engine.
When the loop writes `stop.json` the run ends cleanly after saving the last update.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import open_dict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict
from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, register_trainer
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.utils import tensordict_utils as tu

from autoenvscaling.training.bridge import wait_for_batch, write_ready
from autoenvscaling.training.verl.config import BRIDGE_TRAINER, REPLAY_AGENT, optimizer_batch

logger = logging.getLogger(__name__)

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "upgrade",
              "proxy-authorization", "proxy-authenticate", "te", "trailer"}


@ray.remote(num_cpus=1)
class OpenAIFront:
    """One OpenAI-compatible address in front of verl's rollout replicas.

    Each request goes, unchanged and streamed, to the replica with the fewest requests in flight;
    SGLang serves `/v1/chat/completions` itself, with the configured chat template, tool-call and
    reasoning parsers.
    """

    def __init__(self, replicas: list[str], port: int = 0):
        self.load = dict.fromkeys(replicas, 0)
        self.port = port

    async def serve(self) -> str:
        import aiohttp
        from aiohttp import web

        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                             auto_decompress=False)
        app = web.Application(client_max_size=1 << 30)
        app.router.add_route("*", "/{path:.*}", self.forward)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", self.port).start()
        return f"http://{ray.util.get_node_ip_address()}:{runner.addresses[0][1]}"

    async def forward(self, request: Any) -> Any:
        from aiohttp import web

        replica = min(self.load, key=self.load.get)
        self.load[replica] += 1
        try:
            async with self.session.request(request.method, f"http://{replica}{request.rel_url}",
                                            data=await request.read(), headers=_end_to_end(request.headers)) as up:
                response = web.StreamResponse(status=up.status, headers=_end_to_end(up.headers))
                await response.prepare(request)
                async for chunk in up.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        finally:
            self.load[replica] -= 1


def _end_to_end(headers: Any) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


class _NoDataset:
    """Stands in for the prompt dataloader in checkpoints: bridge batches come from the loop."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


@register_trainer(BRIDGE_TRAINER)
class BridgeTrainer(PPOTrainerColocateAsync):
    def __init__(self, config: Any):
        super().__init__(config)
        section = config.trainer.v1[BRIDGE_TRAINER]
        self.control_dir = Path(section.control_dir)
        self.poll_s = float(section.get("poll_s", 5.0))
        self.port = int(section.get("port", 0))
        self.rows: list[dict[str, Any]] = []
        self.first_step = 1
        # Every batch is complete when it is trained on: nothing is in flight across an update or
        # a checkpoint, so verl's synchronous sampling and checkpoint semantics apply.
        self.trainer_mode = "sync"

    def _build_replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(trainer_mode="sync", trainer_config={}, max_off_policy_threshold=None,
                            max_off_policy_strategy="drop", sampler_kwargs={})

    def _init_dataloader(self) -> None:
        self.train_dataloader = _NoDataset()
        self.total_training_steps = self.steps_per_epoch = int(self.config.trainer.total_training_steps)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps

    # ---- the handshake -----------------------------------------------------------------------
    def on_train_begin(self) -> None:
        node = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
        self.front = OpenAIFront.options(scheduling_strategy=node).remote(
            self.llm_server_manager.get_addresses(), self.port)
        self.front_url = ray.get(self.front.serve.remote())
        logger.info("[aes] bridge at %s serving %s", self.control_dir, self.front_url)
        self.first_step = self.global_steps
        self._receive(self.global_steps)

    def on_step_end(self) -> None:
        super().on_step_end()             # new weights in the replicas, generation resumed
        if self.global_steps < self.total_training_steps:
            self._receive(self.global_steps + 1)

    def on_train_end(self) -> None:
        # Reached only when the loop stopped before the planned budget (a proposer update was
        # skipped); verl saved every multiple of save_freq, so keep the updates made since.
        last, every = self.global_steps - 1, self.config.trainer.save_freq
        if every > 0 and last >= self.first_step and last % every:
            self.checkpoint_manager.sleep_replicas()
            self.global_steps = last
            self._save_checkpoint()

    def _receive(self, step: int) -> None:
        """Announce the weights for update `step` (rollout id `step - 1`) and wait for its batch."""
        write_ready(self.control_dir, step - 1, self.front_url)
        rows = wait_for_batch(self.control_dir, step - 1, poll_s=self.poll_s)
        if rows is None:
            logger.info("[aes] the self-play loop stopped before update %d", step)
            self.total_training_steps = step - 1
            return
        self.rows = rows
        samples, groups = len(rows), len({r["group_index"] for r in rows})
        mc = self.config.actor_rollout_ref.actor.megatron
        world = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        dp = world // (mc.tensor_model_parallel_size * mc.pipeline_model_parallel_size
                       * mc.context_parallel_size)
        with open_dict(self.config):
            self.config.data.train_batch_size = groups
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size = optimizer_batch(
                samples, self.config.actor_rollout_ref.rollout.n, dp)

    # ---- one update --------------------------------------------------------------------------
    def prepare_step(self) -> dict[str, float]:
        """Submit the loop's groups as prompts; `aes_replay` turns each member into a trajectory."""
        sizes = Counter(int(r["group_index"]) for r in self.rows)
        groups = sorted(sizes)
        batch = tu.get_tensordict({
            "uid": np.array([uuid4().hex for _ in groups], dtype=object),
            "index": np.array(groups, dtype=object),
            "group": np.array(groups, dtype=object),
            "rollout_id": np.array([self.global_steps - 1] * len(groups), dtype=object),
            "agent_name": np.array([REPLAY_AGENT] * len(groups), dtype=object),
            "__rollout_n__": np.array([sizes[g] for g in groups], dtype=object)})
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        self._submit_batch_to_rollout(batch)
        rewards = [float(r["reward"]) for r in self.rows]
        return {"aes/role_is_proposer": float({r["role"] for r in self.rows} == {"proposer"}),
                "aes/groups": float(len(groups)), "aes/samples": float(len(self.rows)),
                "aes/mean_reward": sum(rewards) / len(rewards)}

    def _compute_advantage(self, batch: Any, metrics: dict) -> Any:
        """The loop's advantage, computed over each whole group (members that could not be
        trained on included), broadcast over the response tokens."""
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["response_mask", "extra_fields"])
        advantages = [float(e.get("advantage", 0.0)) for e in data["extra_fields"].tolist()]
        masks = data["response_mask"].unbind()
        nested = torch.nested.as_nested_tensor([a * m.float() for a, m in zip(advantages, masks, strict=True)],
                                               layout=torch.jagged)
        fields = TensorDict({"advantages": nested, "returns": nested}, batch_size=len(batch))
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=fields)
