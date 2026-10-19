"""One terminal episode under verl: Pi in a sandbox of the task's image, its model calls through
the recording proxy, its reward from the task's own tests.

The Miles carry parks an unfinished episode at the weight update and resumes it under the new
weights (`training.carry`). verl's colocated asynchronous trainer does the same one level down:
it aborts in-flight generations at the update and continues them under the new weights
(`FullyAsyncLLMServerClient`), so Pi's request is merely slow and the episode here runs straight
through. Everything else is the carry's: the Pi install, the proxy's recording (the engine's
token ids and log-probabilities, never recomputed), the terminal checks and the verifier
contract (`training.carry.driver`).

Nothing here imports verl; `training.verl.agent` wraps `run_episode` as an agent loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from autoenvscaling.sandbox.base import SandboxBackend, SandboxSpec
from autoenvscaling.training.carry.driver import (
    INSTALL_CONCURRENCY,
    default_pi_version,
    install_pi,
    pi_env,
    pi_install_hosts,
    pi_launch,
    pi_models,
    pi_package_spec,
    pi_terminal_status,
    recording_failure,
    run_verifier,
    task_allowed_hosts,
    terminal_failure,
)
from autoenvscaling.training.carry.rollout import CarrySettings

logger = logging.getLogger(__name__)

Recording = Callable[[str], Awaitable[dict | None]]
Release = Callable[[str], Awaitable[Any]]


_installs = asyncio.Semaphore(INSTALL_CONCURRENCY)   # per process, as the carry driver bounds it


class EpisodeFailed(RuntimeError):
    """The episode ended without a trainable recording; verl drops its whole group."""


async def run_episode(*, backend: SandboxBackend, settings: CarrySettings, metadata: dict[str, Any],
                      instruction: str, context_window: int, max_tokens: int, name: str,
                      recording: Recording, release: Release, version: int) -> dict[str, Any]:
    """Run one episode and return its trainable sample (`sample_from_recording`).

    `recording`/`release` read and drop the proxy's record of an episode id; `name` prefixes the
    sandbox name (the run id, so a restarted job can remove what a dead one left behind);
    `version` is the weight version the episode was admitted under.
    """
    image = metadata.get("image")
    if not image:
        raise EpisodeFailed("prompt row carries no image; cannot start a sandbox")
    episode_id = f"ep-{uuid.uuid4().hex[:20]}"
    # verl holds Pi's request across weight updates without telling the episode, so the budget
    # is wall-clock: the carry's active budget plus its paused backstop.
    deadline = time.monotonic() + settings.episode_timeout_s + settings.max_paused_s

    def remaining(cap: float = float("inf")) -> float:
        left = min(deadline - time.monotonic(), cap)
        if left <= 0:
            raise TimeoutError(f"episode {episode_id} exhausted its time budget")
        return left

    sandbox = await backend.create(SandboxSpec(
        image=image, cpu=settings.sandbox_cpu, memory_gib=settings.sandbox_memory_gib,
        timeout_s=int(remaining() + 600), workdir=metadata.get("workdir") or settings.sandbox_workdir,
        domain_allowlist=allowed_hosts(settings, metadata), name=f"{name}-{episode_id}"))
    try:
        base_url = f"{settings.proxy_public_url.rstrip('/')}/v1/ep/{episode_id}"
        models = pi_models(base_url, settings.served_model, context_window=context_window, max_tokens=max_tokens)
        package = pi_package_spec(settings.pi_version or default_pi_version(), settings.pi_package)

        async def install() -> None:
            async with _installs:
                await install_pi(sandbox, package, models, label=episode_id)

        await asyncio.wait_for(install(), remaining())
        prompt_path = f"/tmp/aes_pi_prompt_{episode_id}.txt"
        await sandbox.write_file(prompt_path, instruction.encode("utf-8"))
        pi = await sandbox.exec("bash", "-lc", pi_launch(prompt_path, settings.served_model),
                                timeout_s=int(remaining()),
                                env=pi_env(os.environ.get(settings.proxy_api_key_env, "")))
        if pi.timed_out or pi.returncode:
            raise EpisodeFailed(f"Pi process failed (rc={pi.returncode}, timed_out={pi.timed_out}): "
                                f"{(pi.stderr or pi.stdout or '')[-300:]}")
        rec = await recording(episode_id)
        if failure := recording_failure(rec):
            raise EpisodeFailed(failure)
        if failure := terminal_failure(await pi_terminal_status(sandbox, remaining(60.0)), rec):
            raise EpisodeFailed(failure)
        verify_until = time.monotonic() + settings.verify_timeout_s
        reward = await run_verifier(sandbox, metadata.get("task_dir") or "",
                                    workdir=settings.sandbox_workdir,
                                    remaining=lambda: remaining(verify_until - time.monotonic()))
    finally:
        await _close(sandbox, episode_id, release)
    return sample_from_recording(rec, reward, version=version)


def sample_from_recording(rec: dict[str, Any], reward: float, *, version: int) -> dict[str, Any]:
    """`AgentLoopOutput` fields from a proxy recording.

    The prompt is the first render; the response is everything after it, generated tokens and
    tool results alike, with `response_mask` deciding what trains and the engine's
    log-probabilities as the behaviour policy. The weight versions the generations ran under
    become verl's off-policy span; a recording without them counts as `version`.
    """
    tokens, logprobs, mask = rec["tokens"], rec["logprobs"], rec["loss_mask"]
    cut = int(rec.get("prompt_len") or 0)
    if not len(tokens) == len(logprobs) == len(mask):
        raise EpisodeFailed(f"recording is internally inconsistent: tokens={len(tokens)} "
                            f"logprobs={len(logprobs)} mask={len(mask)}")
    if not 0 < cut < len(tokens):
        raise EpisodeFailed(f"recording has an unusable prompt_len={cut} for {len(tokens)} tokens")
    versions = [int(v) for span in rec.get("weight_versions") or []
                for v in (span if isinstance(span, (list, tuple)) else [span]) if v is not None]
    return {"prompt_ids": tokens[:cut], "response_ids": tokens[cut:], "response_mask": mask[cut:],
            "response_logprobs": logprobs[cut:], "reward_score": float(reward),
            "num_turns": int(rec.get("turns") or 0),
            "extra_fields": {"min_global_steps": min(versions, default=version),
                             "max_global_steps": max(versions, default=version)}}


def allowed_hosts(settings: CarrySettings, metadata: dict[str, Any]) -> list[str] | None:
    """The solver's network, as everywhere else in the repository: the proxy's host, the hosts Pi
    installs from and whatever the task allows; None opens it (`AES_SOLVER_NETWORK=public`)."""
    if settings.network == "public":
        return None
    hosts = [h for h in (urlsplit(settings.proxy_public_url).hostname,) if h]
    hosts += [*pi_install_hosts(), *task_allowed_hosts(metadata.get("task_dir") or "")]
    return sorted(dict.fromkeys(hosts))


def instruction_of(kwargs: dict[str, Any]) -> str:
    """The task text: `metadata.instruction`, else the first user message of the prompt row."""
    metadata = kwargs.get("metadata") or {}
    if metadata.get("instruction"):
        return str(metadata["instruction"])
    for message in kwargs.get("raw_prompt") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    raise EpisodeFailed("prompt row has no instruction")


async def _close(sandbox: Any, episode_id: str, release: Release) -> None:
    """Terminate the sandbox and drop the recording; a failure here is logged, never raised."""
    for step in (sandbox.terminate, lambda: release(episode_id)):
        try:
            await asyncio.wait_for(asyncio.shield(step()), 60)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the episode's outcome
            logger.warning("[aes] cleanup of %s incomplete: %s", episode_id, exc)
