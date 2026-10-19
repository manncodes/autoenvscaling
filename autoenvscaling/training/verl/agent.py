"""AutoEnvScaling's agent loops in verl, and the manager that hosts the recording proxy.

    aes_pi       one terminal episode, Pi in a sandbox (carry mode, `training.verl.episode`)
    aes_replay   one sample of a bridge batch, re-tokenized (bridge mode, `training.verl.bridge`)

`AgentLoopManager` is verl's TransferQueue manager (`actor_rollout_ref.rollout.agent.
agent_loop_manager_class`) with two additions. Its workers are defined here, so every worker
process imports this module and both loops are registered where verl instantiates them. In
carry mode it starts the recording proxy (`training.carry.proxy`) as a Ray actor on the
trainer's node, with verl's own LLM client as the proxy's engine: requests are balanced over the
rollout replicas, an episode stays on one replica, and a generation cut by the weight update
continues under the new weights, so Pi never sees the update.

Sandboxes follow the carry's settings (`AES_CARRY_*`, `CarrySettings`); the proxy reads
`AES_PROXY_*` (port, API key, tool parser), and sandboxes reach it at AES_CARRY_PROXY_PUBLIC_URL.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import replace
from typing import Any
from uuid import uuid4

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, AgentLoopWorkerTQ

from autoenvscaling.sandbox.base import get_backend
from autoenvscaling.training.bridge import batch_path
from autoenvscaling.training.carry.proxy import (
    Generation,
    GenerationAborted,
    ProxyConfig,
    ProxyError,
    build_app,
    make_proxy,
)
from autoenvscaling.training.carry.rollout import CarrySettings, resolve_identity
from autoenvscaling.training.verl.config import BRIDGE_TRAINER, PI_AGENT, REPLAY_AGENT
from autoenvscaling.training.verl.data import encode_conversation, rows_by_group, split_sample
from autoenvscaling.training.verl.episode import instruction_of, run_episode
from autoenvscaling.trajectories.export import read_jsonl

logger = logging.getLogger(__name__)

PROXY_ACTOR = "aes_recording_proxy"


# ---- carry: the recording proxy on verl's engines ---------------------------------------------------------
class VerlEngine:
    """The proxy's engine: verl's LLM client, routed by episode (`session`)."""

    def __init__(self, client: Any, tokenizer: Any):
        self.client = client
        self.tokenizer = tokenizer

    async def generate(self, input_ids: list[int], max_new: int, temperature: float,
                       session: str | None = None) -> Generation:
        out = await self.client.generate(session or uuid4().hex, prompt_ids=input_ids, sampling_params={
            "temperature": temperature, "max_tokens": max_new, "logprobs": True})
        if out.stop_reason in ("abort", "aborted"):   # only without partial rollout
            raise GenerationAborted("engine request aborted")
        extra = out.extra_fields or {}
        return Generation(list(out.token_ids), list(out.log_probs),
                          self.tokenizer.decode(out.token_ids, skip_special_tokens=True),
                          weight_version=[extra.get("min_global_steps"), extra.get("max_global_steps")])


@ray.remote(num_cpus=1)
class RecordingProxy:
    """The recording proxy as a Ray actor: HTTP for Pi, method calls for the agent loops."""

    def __init__(self, config: ProxyConfig, client: Any):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.hf_dir, trust_remote_code=True)
        self.proxy = make_proxy(config, tokenizer=tokenizer, engine=VerlEngine(client, tokenizer))
        self.server: asyncio.Task | None = None

    async def serve(self) -> str:
        import uvicorn

        cfg = self.proxy.config
        server = uvicorn.Server(uvicorn.Config(build_app(self.proxy), host=cfg.host, port=cfg.port,
                                               log_level="warning"))
        self.server = asyncio.create_task(server.serve())
        while not server.started:
            if self.server.done():
                raise RuntimeError(f"the recording proxy could not listen on {cfg.host}:{cfg.port}")
            await asyncio.sleep(0.1)
        return f"http://{ray.util.get_node_ip_address()}:{cfg.port}"

    async def recording(self, episode_id: str) -> dict | None:
        try:
            return self.proxy.episode(episode_id, full=True)
        except ProxyError as exc:
            if exc.status == 404:
                return None
            raise

    async def drop(self, episode_id: str) -> None:
        self.proxy.drop(episode_id)


@functools.cache
def sandboxes(save_dir: str) -> tuple[CarrySettings, Any]:
    """`AES_CARRY_*` settings, with the run identity taken from the checkpoint directory, and the
    sandbox backend they name (one per process)."""
    settings = resolve_identity(CarrySettings.from_env(), save_dir)
    return settings, get_backend(settings.sandbox_backend)


@register(PI_AGENT)
class PiAgentLoop(AgentLoopBase):
    """One terminal episode: Pi in a sandbox, recorded by the proxy, scored by the task's tests."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        settings, backend = sandboxes(str(self.config.trainer.default_local_dir))
        proxy = ray.get_actor(PROXY_ACTOR)
        rollout = self.rollout_config
        sample = await run_episode(
            backend=backend, settings=settings, metadata=dict(kwargs.get("metadata") or {}),
            instruction=instruction_of(kwargs), context_window=rollout.prompt_length + rollout.response_length,
            max_tokens=rollout.response_length, name=settings.run_id,
            recording=lambda e: proxy.recording.remote(e), release=lambda e: proxy.drop.remote(e),
            version=int(kwargs["global_steps"]) - 1)
        return AgentLoopOutput(**sample, metrics=AgentLoopMetrics())


# ---- bridge: replaying the self-play loop's batch ---------------------------------------------------------
@functools.lru_cache(maxsize=2)
def _bridge_batch(control_dir: str, rollout_id: int) -> dict[int, list[dict[str, Any]]]:
    return rows_by_group(read_jsonl(batch_path(control_dir, rollout_id)))


@register(REPLAY_AGENT)
class ReplayAgentLoop(AgentLoopBase):
    """One sample of a bridge batch: the self-play loop's conversation, re-tokenized with the
    policy's chat template; its group advantage travels in `extra_fields`."""

    async def run(self, sampling_params: dict[str, Any], *, rollout_id: int, group: int, session_id: int,
                  global_steps: int, **_: Any) -> AgentLoopOutput:
        control_dir = str(self.config.trainer.v1[BRIDGE_TRAINER].control_dir)
        rows = await self.loop.run_in_executor(None, _bridge_batch, control_dir, int(rollout_id))
        row = rows[int(group)][int(session_id)]
        ids, mask = await self.loop.run_in_executor(None, functools.partial(
            encode_conversation, row["messages"], row.get("tools"), self.tokenizer,
            **self.apply_chat_template_kwargs))
        limit = self.rollout_config.prompt_length + self.rollout_config.response_length
        prompt, response, response_mask = split_sample(ids, mask, limit)
        version = int(global_steps) - 1
        return AgentLoopOutput(
            prompt_ids=prompt, response_ids=response, response_mask=response_mask,
            reward_score=float(row["reward"]), num_turns=len(row["messages"]), metrics=AgentLoopMetrics(),
            extra_fields={"advantage": float(row["advantage"]), "role": row["role"],
                          "trajectory_id": row["trajectory_id"], "truncated": len(ids) > limit,
                          "min_global_steps": version, "max_global_steps": version})


# ---- verl's manager and workers, with the loops above registered --------------------------------------------
@ray.remote
class AgentLoopWorker(AgentLoopWorkerTQ.__ray_actor_class__):
    """verl's TransferQueue worker, defined here so its process registers aes_pi and aes_replay."""


class AgentLoopManager(AgentLoopManagerTQ):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = AgentLoopWorker
        self.proxy = None

    async def _init_agent_loop_workers(self) -> None:
        if self.rollout_config.agent.default_agent_loop == PI_AGENT:
            await self._start_proxy()
        await super()._init_agent_loop_workers()

    async def _start_proxy(self) -> None:
        settings, backend = sandboxes(str(self.config.trainer.default_local_dir))
        # One job per checkpoint directory: sandboxes under this run id belong to a job that died.
        if (reap := getattr(backend, "remove_by_prefix", None)) and (removed := await reap(f"{settings.run_id}-")):
            logger.warning("[aes] removed %d sandboxes left by a previous job of run %s", removed, settings.run_id)
        rollout = self.rollout_config
        config = replace(ProxyConfig.from_env(), hf_dir=str(self.model_config.path), served_id=settings.served_model,
                         max_turns=settings.max_turns, max_context=int(rollout.prompt_length + rollout.response_length))
        node = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
        self.proxy = RecordingProxy.options(name=PROXY_ACTOR, scheduling_strategy=node).remote(config, self.llm_client)
        address = await self.proxy.serve.remote()
        logger.info("[aes] recording proxy at %s; sandboxes reach it at %s", address, settings.proxy_public_url)
