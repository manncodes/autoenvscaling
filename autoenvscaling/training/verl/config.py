"""verl run configuration. The defaults are the paper's shared-model settings, the same ones
`training.miles.config` encodes for Miles (Appendix, "Training and Calibration Settings"):

    learning rate                  1e-6 (constant)
    solver rollout batch           16 environments x 16 trajectories
    DPPO total-variation threshold 0.1
    KL / entropy coefficients      0 / 0

`overrides()` renders Hydra overrides for verl v0.9.1: `ppo_trainer` with the Megatron engine
for RL (run through `training.verl.main`, which registers AutoEnvScaling's trainer and agent
loops) and `sft_trainer_engine` for fine-tuning (`verl.trainer.sft_trainer`). Three modes:

* `bridge` (default): the self-play loop runs every episode and hands verl finished RL groups
  through `training.bridge`; verl trains one optimizer step per batch (`training.verl.bridge`).
* `carry`: verl runs the episodes itself, Pi in a sandbox per episode through the recording
  proxy (`training.verl.agent`), with partial rollout across weight updates, the zero-variance
  group filter and a staleness bound, all native to verl's V1 trainer.
* `sft`: supervised fine-tuning on `to_sft` rows (`training.verl.data.ConversationDataset`).

How the Miles settings map onto verl:

    Miles                                     verl
    --advantage-estimator dppo, tv, 0.1       actor.policy_loss.loss_mode=dppo_tv, clip_ratio=0.1
    hooks.center_rewards_by_group             algorithm.adv_estimator=grpo, no std normalization
    per-sample loss mean (Miles' default)     actor.loss_agg_mode=seq-mean-token-mean
    --use-rollout-logprobs                    algorithm.rollout_correction.bypass_mode=True
    --partial-rollout                         trainer.v1.trainer_mode=colocate_async
    check_reward_nonzero_std                  algorithm.filter_groups (metric=reward)
    --max-weight-staleness 4                  trainer.v1.sampler.max_off_policy_threshold=5
    --ref-load (torch_dist conversion)        none: mbridge loads the Hugging Face checkpoint

Paths are never defaulted: model, data and output locations come from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, lcm
from typing import Any

BRIDGE = "bridge"
CARRY = "carry"
SFT = "sft"
MODES = (BRIDGE, CARRY, SFT)

BRIDGE_TRAINER = "aes_bridge"
REPLAY_AGENT = "aes_replay"
PI_AGENT = "aes_pi"
AGENT_LOOP_MANAGER = "autoenvscaling.training.verl.agent.AgentLoopManager"
SFT_DATASET = ("pkg://autoenvscaling.training.verl.data", "ConversationDataset")


@dataclass
class VerlConfig:
    # ---- model and checkpoints ------------------------------------------------------------
    hf_checkpoint: str = ""           # Hugging Face weights of the starting policy
    save: str = ""                    # trainer.default_local_dir
    save_interval: int = 16           # one checkpoint per pool refresh
    load: str | None = None           # a global_step_<n> directory; default: the newest under `save`

    # ---- where rollouts come from ------------------------------------------------------------
    mode: str = BRIDGE
    prompt_data: str | None = None    # carry: a pool export; sft: `export sft` rows
    bridge_dir: str | None = None     # bridge: the control directory the self-play loop also uses
    bridge_port: int = 0              # bridge: port of the loop's OpenAI address (0: any free one)
    partial_rollout: bool = True      # carry: unfinished episodes continue across weight updates
    max_weight_staleness: int = 4     # carry: updates an unfinished episode may span
    inflight_batches: float = 3.0     # carry: batches generating at once (3 x 256 = 768 episodes)

    # ---- batch and schedule --------------------------------------------------------------------
    rollout_batch_size: int = 16      # environments per solver batch
    n_samples_per_prompt: int = 16    # attempts per environment
    num_rollout: int = 305            # trainer updates; see Schedule.planned_updates
    num_epoch: int | None = None      # sft mode

    # ---- lengths -------------------------------------------------------------------------------
    seq_length: int = 131072
    rollout_max_context_len: int = 131072
    rollout_max_response_len: int = 16384   # per model call
    rollout_temperature: float = 1.0

    # ---- algorithm -----------------------------------------------------------------------------
    dppo_divergence_type: str = "tv"
    dppo_divergence_threshold: float = 0.1
    dppo_ratio_cap: float = 20.0      # verl's truncated-IS cap; Miles' DPPO had none
    kl_loss_coef: float = 0.0
    entropy_coef: float = 0.0
    loss_agg_mode: str = "seq-mean-token-mean"

    # ---- optimizer -----------------------------------------------------------------------------
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    min_lr: float | None = None
    lr_warmup_fraction: float | None = None
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    clip_grad: float = 1.0

    # ---- parallelism (one 8-GPU node, colocated training and inference) -------------------------
    num_nodes: int = 1
    gpus_per_node: int = 8
    tensor_parallel: int = 2
    context_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 8
    expert_tensor_parallel: int = 1
    # Packed (THD) sequences and dynamic batch size. Qwen3.5/3.6's gated delta-net needs
    # Megatron-Core >= 0.18 for them; verl's tested Qwen3.5 recipe runs padded (bshd) instead.
    packed_sequences: bool = False
    offload: bool = True              # page Megatron parameters and optimizer out during rollout

    # ---- serving -------------------------------------------------------------------------------
    rollout_tensor_parallel: int = 8
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 262144             # proposer episodes can exceed 200k tokens
    served_model_name: str = "policy"
    sglang: dict[str, Any] = field(default_factory=lambda: {
        "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3",
        "mamba_scheduler_strategy": "extra_buffer", "moe_runner_backend": "triton"})

    # ---- logging -------------------------------------------------------------------------------
    project: str = "autoenvscaling"
    experiment: str = "verl"
    wandb: bool = False
    extra_overrides: list[str] = field(default_factory=list)

    def with_(self, **changes: Any) -> VerlConfig:
        return replace(self, **changes)

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def data_parallel(self) -> int:
        world = self.num_nodes * self.gpus_per_node
        return world // (self.tensor_parallel * self.context_parallel * self.pipeline_parallel)

    def problems(self) -> list[str]:
        """Everything that would make the run fail or train something other than intended."""
        out = []
        if self.mode not in MODES:
            out.append(f"mode must be one of {MODES}, got {self.mode!r}")
        for name in ("hf_checkpoint", "save"):
            if not getattr(self, name):
                out.append(f"{name} is required")
        if self.mode in (CARRY, SFT) and not self.prompt_data:
            out.append(f"{self.mode} mode needs prompt_data")
        if self.mode == BRIDGE and not self.bridge_dir:
            out.append("bridge mode needs bridge_dir (the control directory the self-play loop also uses)")
        if self.dppo_divergence_type not in ("tv", "kl"):
            out.append("dppo_divergence_type must be 'tv' or 'kl'")
        if self.mode == CARRY and self.max_weight_staleness < 0:
            out.append("max_weight_staleness must be >= 0")
        if self.mode == CARRY and self.inflight_batches < 1:
            out.append("inflight_batches must be >= 1 (the batch being trained is one of them)")
        world = self.num_nodes * self.gpus_per_node
        model_parallel = self.tensor_parallel * self.context_parallel * self.pipeline_parallel
        if world % model_parallel:
            out.append(f"TP*CP*PP = {model_parallel} does not divide {world} GPUs")
        elif self.mode != SFT and self.global_batch_size % self.data_parallel:
            out.append(f"{self.global_batch_size} samples per batch do not split over "
                       f"{self.data_parallel} data-parallel ranks")
        if world % self.rollout_tensor_parallel:
            out.append(f"rollout_tensor_parallel {self.rollout_tensor_parallel} does not divide {world} GPUs")
        if self.max_model_len < self.rollout_max_context_len:
            out.append("max_model_len must be >= rollout_max_context_len")
        return out

    # ---- rendering -----------------------------------------------------------------------------
    def overrides(self) -> list[str]:
        """Hydra overrides for verl's trainer config (`ppo_trainer`, or `sft_trainer_engine`)."""
        return _render(self._sft() if self.mode == SFT else self._rl()) + list(self.extra_overrides)

    def _rl(self) -> dict[str, Any]:
        actor, rollout = "actor_rollout_ref.actor", "actor_rollout_ref.rollout"
        o: dict[str, Any] = {
            "model_engine": "megatron",
            "actor_rollout_ref.model.path": self.hf_checkpoint,
            "actor_rollout_ref.model.trust_remote_code": True,
            "actor_rollout_ref.model.use_remove_padding": self.packed_sequences,
            # DPPO with mean-centred advantages computed per group (no std division).
            "algorithm.adv_estimator": "grpo",
            "algorithm.norm_adv_by_std_in_grpo": False,
            "algorithm.use_kl_in_reward": False,
            f"{actor}.policy_loss.loss_mode": f"dppo_{self.dppo_divergence_type}",
            f"{actor}.clip_ratio": self.dppo_divergence_threshold,
            f"{actor}.clip_ratio_low": self.dppo_divergence_threshold,
            f"{actor}.clip_ratio_high": self.dppo_divergence_threshold,
            f"{actor}.clip_ratio_c": self.dppo_ratio_cap,
            f"{actor}.loss_agg_mode": self.loss_agg_mode,
            f"{actor}.use_kl_loss": self.kl_loss_coef > 0,
            f"{actor}.kl_loss_coef": self.kl_loss_coef,
            f"{actor}.entropy_coeff": self.entropy_coef,
            # One optimizer step per batch: 16 environments x 16 attempts.
            f"{actor}.ppo_epochs": 1,
            f"{actor}.ppo_mini_batch_size": self.rollout_batch_size,
            f"{actor}.ppo_micro_batch_size_per_gpu": 1,
            f"{actor}.use_dynamic_bsz": self.packed_sequences,
            f"{actor}.ppo_max_token_len_per_gpu": self.seq_length,
            **self._optim(f"{actor}.optim"),
            **self._parallel(f"{actor}.megatron"),
            f"{actor}.megatron.param_offload": self.offload,
            f"{actor}.megatron.optimizer_offload": self.offload,
            f"{rollout}.name": "sglang",
            f"{rollout}.mode": "async",
            f"{rollout}.n": self.n_samples_per_prompt,
            f"{rollout}.temperature": self.rollout_temperature,
            f"{rollout}.tensor_model_parallel_size": self.rollout_tensor_parallel,
            f"{rollout}.gpu_memory_utilization": self.gpu_memory_utilization,
            f"{rollout}.max_model_len": self.max_model_len,
            f"{rollout}.log_prob_micro_batch_size_per_gpu": 1,
            f"{rollout}.log_prob_use_dynamic_bsz": self.packed_sequences,
            f"{rollout}.log_prob_max_token_len_per_gpu": self.seq_length,
            f"{rollout}.prometheus.served_model_name": self.served_model_name,
            **{f"++{rollout}.engine_kwargs.sglang.{k}": v for k, v in self.sglang.items()},
            f"++{rollout}.agent.agent_loop_manager_class": AGENT_LOOP_MANAGER,
            "data.train_batch_size": self.rollout_batch_size,
            "data.max_prompt_length": self.rollout_max_context_len - self.rollout_max_response_len,
            "data.max_response_length": self.rollout_max_response_len,
            "trainer.use_v1": True,
            "trainer.total_training_steps": self.num_rollout,
            "trainer.total_epochs": self.num_rollout,   # the step budget, not the epochs, ends the run
            "trainer.val_before_train": False,
            "trainer.test_freq": -1,
            **self._trainer(),
        }
        return o | (self._carry() if self.mode == CARRY else self._bridge())

    def _bridge(self) -> dict[str, Any]:
        # Batches arrive already grouped, scored and filtered; the actor recomputes the behaviour
        # log-probabilities, as with Miles, because the conversations are re-tokenized. SGLang
        # serves the loop's chat completions itself, so it keeps its tokenizer.
        return {"trainer.v1.trainer_mode": BRIDGE_TRAINER,
                f"++trainer.v1.{BRIDGE_TRAINER}.control_dir": self.bridge_dir,
                f"++trainer.v1.{BRIDGE_TRAINER}.port": self.bridge_port,
                "trainer.v1.sampler.max_off_policy_threshold": None,
                "actor_rollout_ref.rollout.agent.default_agent_loop": REPLAY_AGENT,
                "actor_rollout_ref.rollout.calculate_log_probs": False,
                "actor_rollout_ref.rollout.skip_tokenizer_init": False}

    def _carry(self) -> dict[str, Any]:
        # The final run's rollout: a shuffled pool, over-sampling against the zero-variance
        # filter, partial rollout with the engine's log-probabilities as the behaviour policy.
        # verl measures staleness as the number of weight versions a group spans, so k updates
        # of carry is a span of k + 1.
        o = {"trainer.v1.trainer_mode": "colocate_async" if self.partial_rollout else "sync",
             "trainer.v1.sampler.max_off_policy_threshold": self.max_weight_staleness + 1,
             "trainer.v1.sampler.max_off_policy_strategy": "drop",
             "algorithm.filter_groups.enable": True,
             "algorithm.filter_groups.metric": "reward",
             "algorithm.rollout_correction.bypass_mode": True,
             "actor_rollout_ref.rollout.calculate_log_probs": True,
             "actor_rollout_ref.rollout.agent.default_agent_loop": PI_AGENT,
             "data.train_files": self.prompt_data, "data.val_files": self.prompt_data,
             "data.prompt_key": "prompt", "data.shuffle": True}
        if self.partial_rollout:
            # One batch is consumed per step and one is added; the rest is the prefetch window.
            o["trainer.v1.colocate_async.num_warmup_batches"] = self.inflight_batches - 1
        return o

    def _sft(self) -> dict[str, Any]:
        # slime's standard recipe as used in the paper: per-token loss over the whole batch,
        # which is how verl's SFT loss is defined.
        path, name = SFT_DATASET
        o: dict[str, Any] = {
            "engine": "megatron", "optim": "megatron",
            "model.path": self.hf_checkpoint, "model.trust_remote_code": True,
            "model.use_remove_padding": self.packed_sequences,
            "data.train_files": self.prompt_data, "data.val_files": None,
            "data.train_batch_size": self.rollout_batch_size,
            "data.micro_batch_size_per_gpu": 1,
            "data.max_length": self.seq_length, "data.max_token_len_per_gpu": self.seq_length,
            "data.use_dynamic_bsz": self.packed_sequences, "data.pad_mode": "no_padding",
            "data.truncation": "right",
            "data.custom_cls.path": path, "data.custom_cls.name": name,
            **self._optim("optim"), **self._parallel("engine"),
            "trainer.test_freq": -1, **self._trainer()}
        if self.num_epoch is not None:
            o["trainer.total_epochs"] = self.num_epoch
        else:
            o |= {"trainer.total_epochs": self.num_rollout, "trainer.total_training_steps": self.num_rollout}
        return o

    def _optim(self, at: str) -> dict[str, Any]:
        o = {f"{at}.lr": self.lr, f"{at}.lr_decay_style": self.lr_decay_style,
             f"{at}.weight_decay": self.weight_decay, f"{at}.betas": [self.adam_beta1, self.adam_beta2],
             f"{at}.clip_grad": self.clip_grad}
        if self.min_lr is not None:
            o[f"{at}.min_lr"] = self.min_lr
        if self.lr_warmup_fraction is not None:
            o[f"{at}.lr_warmup_steps_ratio"] = self.lr_warmup_fraction
        return o

    def _parallel(self, at: str) -> dict[str, Any]:
        recompute = {"recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 1}
        return {f"{at}.tensor_model_parallel_size": self.tensor_parallel,
                f"{at}.context_parallel_size": self.context_parallel,
                f"{at}.pipeline_model_parallel_size": self.pipeline_parallel,
                f"{at}.expert_model_parallel_size": self.expert_parallel,
                f"{at}.expert_tensor_parallel_size": self.expert_tensor_parallel,
                f"{at}.sequence_parallel": self.tensor_parallel > 1,
                f"{at}.use_mbridge": True, f"{at}.vanilla_mbridge": True,
                f"{at}.use_remove_padding": self.packed_sequences,
                **{f"++{at}.override_transformer_config.{k}": v for k, v in recompute.items()}}

    def _trainer(self) -> dict[str, Any]:
        return {"trainer.nnodes": self.num_nodes, "trainer.n_gpus_per_node": self.gpus_per_node,
                "trainer.default_local_dir": self.save, "trainer.save_freq": self.save_interval,
                "trainer.resume_mode": "resume_path" if self.load else "auto",
                **({"trainer.resume_from_path": self.load} if self.load else {}),
                "trainer.project_name": self.project, "trainer.experiment_name": self.experiment,
                "trainer.logger": ["console", "wandb"] if self.wandb else ["console"]}


def optimizer_batch(samples: int, group_size: int, data_parallel: int) -> int:
    """Mini-batch size (in groups of `group_size`) that trains `samples` as one optimizer step.

    verl pads a batch to a multiple of `mini_batch x group_size` that also splits over the
    data-parallel ranks, and takes one optimizer step per mini-batch; a mini-batch covering the
    whole padded batch is Miles' dynamic global batch size."""
    unit = lcm(group_size, data_parallel)
    return ceil(samples / unit) * unit // group_size


def _render(overrides: dict[str, Any]) -> list[str]:
    return [f"{key}={_value(value)}" for key, value in overrides.items()]


def _value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_value(v) for v in value) + "]"
    text = repr(value) if isinstance(value, float) else str(value)
    return text if text and not set(text) & set(" ,=[]{}()'\"\\:") else "'" + text.replace("'", "\\'") + "'"


# ---------------------------------------------------------------------------------------------
# Presets (the same names as the Miles presets)
# ---------------------------------------------------------------------------------------------
def shared_policy(**paths: Any) -> VerlConfig:
    """Recursive self-improvement: Qwen3.6-35B-A3B as proposer and solver, 288 solver updates
    plus 17 proposer updates (`Schedule(288).planned_updates`)."""
    return VerlConfig(experiment="shared-policy", **paths)


def exp2_solver(model: str = "qwen3.5-9b", **paths: Any) -> VerlConfig:
    """Solver RL with a fixed-weight proposer or a static pool (Section 3.3 experiments):
    group size 16, 16 prompts per batch, 128k maximum sequence length."""
    common = {"num_rollout": 300, "experiment": f"exp2-{model}", "max_model_len": 131072}
    if model == "qwen3.5-9b":
        cfg = VerlConfig(tensor_parallel=4, expert_parallel=1, rollout_tensor_parallel=1,
                         sglang={k: v for k, v in VerlConfig().sglang.items() if k != "moe_runner_backend"},
                         **common)
    elif model == "qwen3.5-35b-a3b":
        cfg = VerlConfig(**common)
    else:
        raise ValueError(f"unknown exp2 solver {model!r} (qwen3.5-9b or qwen3.5-35b-a3b)")
    return cfg.with_(**paths)


def exp2_qwen35_9b(**paths: Any) -> VerlConfig:
    """Solver RL for Qwen3.5-9B with a fixed-weight proposer or a static pool."""
    return exp2_solver("qwen3.5-9b", **paths)


def exp2_qwen35_35b(**paths: Any) -> VerlConfig:
    """Solver RL for Qwen3.5-35B-A3B with a fixed-weight proposer or a static pool."""
    return exp2_solver("qwen3.5-35b-a3b", **paths)


def carry_solver(**paths: Any) -> VerlConfig:
    """Solver RL with partial rollout and over-sampling, as the paper's final Qwen3.6-35B-A3B run
    (16 environments x 16 attempts per update, 768 episodes generating, up to 4 updates of
    carry, 200 updates, 96k context, 16k per model call). `prompt_data` is a pool export."""
    return VerlConfig(mode=CARRY, num_rollout=200, rollout_max_context_len=98304,
                      experiment="carry-solver", **paths)


def cold_start_sft(**paths: Any) -> VerlConfig:
    """Cold-Start: fine-tune Qwen3.6-35B-A3B on the proposer trajectories from `to_sft`.

    slime's Qwen3.5-35B-A3B SFT recipe, as used in the paper: 3 epochs, batch 128, per-token
    loss, lr 1e-5 cosine to 1e-6 with 10% warmup, weight decay 0.1, Adam betas 0.9/0.98.
    """
    return VerlConfig(mode=SFT, rollout_batch_size=128, num_epoch=3, lr=1e-5, lr_decay_style="cosine",
                      min_lr=1e-6, lr_warmup_fraction=0.1, save_interval=20, offload=False,
                      experiment="cold-start", **paths)


PRESETS = {"shared-policy": shared_policy,
           "carry-solver": carry_solver,
           "exp2-qwen3.5-9b": exp2_qwen35_9b,
           "exp2-qwen3.5-35b-a3b": exp2_qwen35_35b,
           "cold-start": cold_start_sft}
