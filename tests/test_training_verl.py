"""The verl backend without GPUs: the paper's settings as verl overrides, the launcher and CLI,
the conversation encoder against a Qwen3.5-shaped template, the Pi episode with a fake sandbox,
and (when verl is installed) every preset composed and validated against verl's own config."""

from __future__ import annotations

import re
import subprocess

import jinja2
import pytest

from autoenvscaling.sandbox.base import ExecResult
from autoenvscaling.training import __main__ as train_cli
from autoenvscaling.training.carry.rollout import CarrySettings
from autoenvscaling.training.verl import launcher as launcher_mod
from autoenvscaling.training.verl.config import (
    PRESETS,
    VerlConfig,
    carry_solver,
    cold_start_sft,
    exp2_solver,
    optimizer_batch,
    shared_policy,
)
from autoenvscaling.training.verl.data import (
    ConversationDataset,
    encode_conversation,
    rows_by_group,
    split_sample,
)
from autoenvscaling.training.verl.episode import (
    EpisodeFailed,
    instruction_of,
    run_episode,
    sample_from_recording,
)
from autoenvscaling.training.verl.launcher import VerlLauncher, latest_checkpoint

PATHS = {"hf_checkpoint": "/models/qwen3.6", "save": "/runs/save"}


def overrides(config: VerlConfig) -> dict[str, str]:
    return dict(o.partition("=")[::2] for o in config.overrides())


# ---- configuration -------------------------------------------------------------------------
def test_defaults_are_the_paper_settings_in_verl_terms():
    o = overrides(shared_policy(bridge_dir="/runs/bridge", **PATHS))
    actor = "actor_rollout_ref.actor"
    assert o[f"{actor}.policy_loss.loss_mode"] == "dppo_tv"
    assert o[f"{actor}.clip_ratio"] == o[f"{actor}.clip_ratio_low"] == o[f"{actor}.clip_ratio_high"] == "0.1"
    assert o["algorithm.adv_estimator"] == "grpo" and o["algorithm.norm_adv_by_std_in_grpo"] == "False"
    assert o[f"{actor}.loss_agg_mode"] == "seq-mean-token-mean"
    assert o[f"{actor}.use_kl_loss"] == "False" and o[f"{actor}.entropy_coeff"] == "0.0"
    assert o[f"{actor}.optim.lr"] == "1e-06" and o[f"{actor}.optim.lr_decay_style"] == "constant"
    assert o[f"{actor}.optim.betas"] == "[0.9,0.98]" and o[f"{actor}.optim.weight_decay"] == "0.1"
    assert o["data.train_batch_size"] == "16" and o["actor_rollout_ref.rollout.n"] == "16"
    assert o[f"{actor}.ppo_mini_batch_size"] == "16" and o[f"{actor}.ppo_epochs"] == "1"
    assert o["trainer.total_training_steps"] == "305" and o["trainer.save_freq"] == "16"
    assert o["actor_rollout_ref.model.path"] == "/models/qwen3.6"
    assert o["trainer.default_local_dir"] == "/runs/save" and o["trainer.resume_mode"] == "auto"


def test_bridge_mode_feeds_verl_from_the_self_play_loop():
    o = overrides(shared_policy(bridge_dir="/runs/bridge", **PATHS))
    assert o["trainer.v1.trainer_mode"] == "aes_bridge"
    assert o["++trainer.v1.aes_bridge.control_dir"] == "/runs/bridge"
    assert o["actor_rollout_ref.rollout.agent.default_agent_loop"] == "aes_replay"
    assert o["++actor_rollout_ref.rollout.agent.agent_loop_manager_class"].endswith("verl.agent.AgentLoopManager")
    # Re-tokenized conversations: the actor recomputes the behaviour log-probabilities.
    assert o["actor_rollout_ref.rollout.calculate_log_probs"] == "False"
    assert "algorithm.rollout_correction.bypass_mode" not in o
    # SGLang must serve chat completions for the loop, with its tokenizer and parsers.
    assert o["actor_rollout_ref.rollout.skip_tokenizer_init"] == "False"
    assert o["++actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser"] == "qwen3_coder"


def test_carry_mode_maps_the_final_run_onto_verl():
    c = carry_solver(prompt_data="/data/pool.jsonl", **PATHS)
    o = overrides(c)
    assert o["trainer.v1.trainer_mode"] == "colocate_async"                  # --partial-rollout
    assert o["algorithm.rollout_correction.bypass_mode"] == "True"           # --use-rollout-logprobs
    assert o["algorithm.filter_groups.enable"] == "True" and o["algorithm.filter_groups.metric"] == "reward"
    assert o["trainer.v1.sampler.max_off_policy_threshold"] == "5"           # 4 updates of carry
    assert o["trainer.v1.sampler.max_off_policy_strategy"] == "drop"
    assert o["trainer.v1.colocate_async.num_warmup_batches"] == "2.0"        # 768 episodes generating
    assert o["actor_rollout_ref.rollout.agent.default_agent_loop"] == "aes_pi"
    assert o["data.train_files"] == o["data.val_files"] == "/data/pool.jsonl"
    assert o["data.max_prompt_length"] == str(98304 - 16384) and o["data.max_response_length"] == "16384"
    assert o["trainer.total_training_steps"] == "200"
    assert overrides(c.with_(partial_rollout=False))["trainer.v1.trainer_mode"] == "sync"


def test_sft_mode_and_presets():
    o = overrides(cold_start_sft(prompt_data="/data/sft.jsonl", **PATHS))
    assert o["engine"] == o["optim"] == "megatron"
    assert o["data.custom_cls.path"] == "'pkg://autoenvscaling.training.verl.data'"
    assert o["data.custom_cls.name"] == "ConversationDataset"
    assert o["optim.lr"] == "1e-05" and o["optim.min_lr"] == "1e-06" and o["optim.lr_decay_style"] == "cosine"
    assert o["optim.lr_warmup_steps_ratio"] == "0.1" and o["trainer.total_epochs"] == "3"
    assert o["data.train_batch_size"] == "128" and o["data.pad_mode"] == "no_padding"
    assert "trainer.total_training_steps" not in o
    small = overrides(exp2_solver("qwen3.5-9b", bridge_dir="/b", **PATHS))
    assert small["actor_rollout_ref.actor.megatron.tensor_model_parallel_size"] == "4"
    assert small["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "1"
    assert "++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend" not in small
    with pytest.raises(ValueError):
        exp2_solver("qwen9")
    from autoenvscaling.training.miles.config import PRESETS as MILES_PRESETS
    assert set(PRESETS) == set(MILES_PRESETS)
    for name, make in PRESETS.items():
        config = make(prompt_data="/d.jsonl", bridge_dir="/b", **PATHS)
        assert config.problems() == [], name


def test_problems_and_override_rendering():
    assert {"hf_checkpoint is required", "save is required"} <= set(VerlConfig().problems())
    assert "bridge_dir" in " ".join(VerlConfig(**PATHS).problems())
    assert "prompt_data" in " ".join(carry_solver(**PATHS).problems())
    assert "does not divide" in " ".join(shared_policy(bridge_dir="/b", tensor_parallel=3, **PATHS).problems())
    assert "data-parallel" in " ".join(shared_policy(bridge_dir="/b", rollout_batch_size=3, n_samples_per_prompt=3,
                                                      **PATHS).problems())
    o = overrides(shared_policy(bridge_dir="/runs/a b", load="/s/global_step_32", **PATHS))
    assert o["++trainer.v1.aes_bridge.control_dir"] == "'/runs/a b'"
    assert o["trainer.resume_mode"] == "resume_path" and o["trainer.resume_from_path"] == "/s/global_step_32"
    assert overrides(VerlConfig(mode="carry", **PATHS))["data.train_files"] == "null"


@pytest.mark.parametrize("samples,group,dp,groups", [(256, 16, 4, 16), (128, 16, 4, 8), (250, 16, 4, 16),
                                                     (128, 16, 3, 9), (16, 16, 8, 1)])
def test_one_optimizer_step_per_bridge_batch(samples, group, dp, groups):
    assert optimizer_batch(samples, group, dp) == groups
    mini = groups * group
    assert mini >= samples and mini % dp == 0 and mini - samples < group * dp


# ---- launcher and command line -------------------------------------------------------------------
def test_commands_for_rl_and_sft():
    launcher = VerlLauncher(python="python3", env={})
    rl = launcher.command(shared_policy(bridge_dir="/b", **PATHS))
    assert rl[:3] == ["python3", "-m", "autoenvscaling.training.verl.main"]
    sft = launcher.command(cold_start_sft(prompt_data="/d.jsonl", **PATHS))
    assert sft[:5] == ["torchrun", "--nnodes=1", "--nproc-per-node=8", "--standalone", "-m"]
    assert sft[5] == "verl.trainer.sft_trainer"
    multi = VerlLauncher(env={"NODE_RANK": "1", "MASTER_ADDR": "10.0.0.1"}).command(
        cold_start_sft(prompt_data="/d.jsonl", num_nodes=2, **PATHS))
    assert "--node-rank=1" in multi and "--master-addr=10.0.0.1" in multi
    carry = launcher.problems(carry_solver(prompt_data="/d.jsonl", **PATHS))
    assert any("proxy_public_url" in p for p in carry)


def test_run_reads_back_the_checkpoint(tmp_path, monkeypatch):
    save, bridge = tmp_path / "save", tmp_path / "bridge"

    def fake_run(cmd, env=None, check=False, **kw):
        (save / "global_step_16").mkdir(parents=True)
        (save / "global_step_32").mkdir()
        (save / "latest_checkpointed_iteration.txt").write_text("32")
        bridge.mkdir(exist_ok=True)
        (bridge / "stop.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(launcher_mod.subprocess, "run", fake_run)
    result = VerlLauncher(env={}).run(shared_policy(hf_checkpoint="/m", save=str(save), bridge_dir=str(bridge)))
    assert result.ok and result.iteration == 32 and result.checkpoint.endswith("global_step_32")
    assert result.detail["stopped_by_loop"]
    (save / "latest_checkpointed_iteration.txt").unlink()
    assert latest_checkpoint(save) == (32, save / "global_step_32")
    assert latest_checkpoint(tmp_path / "empty") is None


def test_train_command_picks_the_backend(capsys, monkeypatch):
    args = ["plan", "--preset", "shared-policy", "--hf-checkpoint", "/m", "--ref-load", "/ignored",
            "--save", "/s", "--bridge-dir", "/b", "--num-rollout", "33", "--set", "rollout_batch_size=8",
            "--override", "trainer.logger=[console,wandb]"]
    assert train_cli.main(["--backend", "verl", *args]) == 0
    out = capsys.readouterr().out
    assert "-m autoenvscaling.training.verl.main" in out and "trainer.total_training_steps=33" in out
    assert "data.train_batch_size=8" in out and out.rstrip().endswith("'trainer.logger=[console,wandb]'")
    assert "/ignored" not in out
    monkeypatch.setenv("AES_TRAIN_BACKEND", "verl")
    assert train_cli.main(["presets"]) == 0 and "carry-solver" in capsys.readouterr().out
    assert train_cli.main(["--backend=jax", "presets"]) == 2


# ---- conversations as tokens ---------------------------------------------------------------------
# Qwen3.5's shape: consecutive tool results fold into one user turn, the generation prompt opens
# a <think> block, and every turn ends with <|im_end|> and a newline.
QWEN_LIKE = (
    "{%- for m in messages %}"
    "{%- if m.role == 'tool' %}"
    "{%- if loop.previtem and loop.previtem.role != 'tool' %}{{- '<|im_start|>user' }}{%- endif %}"
    "{{- '\\n<tool_response>\\n' + m.content + '\\n</tool_response>' }}"
    "{%- if loop.last or loop.nextitem.role != 'tool' %}{{- '<|im_end|>\\n' }}{%- endif %}"
    "{%- elif m.role == 'assistant' %}"
    "{%- if drop_old_reasoning and not loop.last %}{{- '<|im_start|>assistant\\n' + m.content }}"
    "{%- else %}{{- '<|im_start|>assistant\\n<think>\\n' + (m.reasoning_content or '') + '\\n</think>\\n\\n'"
    " + m.content }}{%- endif %}"
    "{%- for c in m.tool_calls or [] %}{{- '<tool_call>' + c.function.arguments.command + '</tool_call>' }}"
    "{%- endfor %}{{- '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}")


class CharTokenizer:
    """Special tokens whole, every other character one token, offsets as a fast tokenizer gives."""

    PIECES = re.compile(r"<\|[a-z_]+\|>|</?think>|.", re.DOTALL)

    def __init__(self, **template_vars):
        self.template = jinja2.Environment().from_string(QWEN_LIKE)
        self.template_vars = template_vars
        self.vocab: dict[str, int] = {}

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kw):
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                    **self.template_vars, **kw)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        pieces = [(m.group(), m.span()) for m in self.PIECES.finditer(text)]
        return {"input_ids": [self.vocab.setdefault(p, len(self.vocab)) for p, _ in pieces],
                "offset_mapping": [span for _, span in pieces]}

    def decode(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return "".join(inverse[i] for i in ids)


def call(command):
    return {"id": command, "type": "function", "function": {"name": "bash", "arguments": {"command": command}}}


CONVERSATION = [
    {"role": "system", "content": "You are Pi."},
    {"role": "user", "content": "Fix the build."},
    {"role": "assistant", "content": "", "reasoning_content": "look", "tool_calls": [call("ls"), call("cat")]},
    {"role": "tool", "content": "Makefile"},
    {"role": "tool", "content": "all: build"},
    {"role": "assistant", "content": "Done.", "reasoning_content": "fixed"},
]


def trained_runs(tok, ids, mask):
    runs, current = [], []
    for token, m in zip(ids, mask, strict=True):
        if m:
            current.append(token)
        elif current:
            runs.append(tok.decode(current))
            current = []
    return runs + ([tok.decode(current)] if current else [])


def test_assistant_turns_are_trained_and_nothing_else():
    tok = CharTokenizer()
    ids, mask = encode_conversation(CONVERSATION, None, tok)
    assert tok.decode(ids) == tok.apply_chat_template(CONVERSATION)
    assert trained_runs(tok, ids, mask) == [
        "look\n</think>\n\n<tool_call>ls</tool_call><tool_call>cat</tool_call><|im_end|>",
        "fixed\n</think>\n\nDone.<|im_end|>"]


def test_a_template_that_rewrites_history_is_refused():
    with pytest.raises(ValueError, match="prefix"):
        encode_conversation(CONVERSATION, None, CharTokenizer(drop_old_reasoning=True))


def test_split_sample_and_inert_samples():
    assert split_sample([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 0, 1], 100) == ([1, 2], [3, 4, 5, 6], [1, 1, 0, 1])
    assert split_sample([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 0, 1], 4) == ([1, 2, 3], [4], [0])   # too long
    assert split_sample([1, 2, 3], [0, 0, 0], 100) == ([1, 2], [3], [0])                     # nothing to train
    assert split_sample([1, 2, 3], [1, 1, 0], 100) == ([1], [2, 3], [1, 0])                  # keeps a prompt


def test_rows_by_group_orders_members():
    rows = [{"group_index": 1, "index": 1}, {"group_index": 0, "index": 0}, {"group_index": 1, "index": 0}]
    assert {g: [r["index"] for r in m] for g, m in rows_by_group(rows).items()} == {0: [0], 1: [0, 1]}


def test_conversation_dataset_reads_sft_rows(tmp_path):
    pytest.importorskip("torch")
    from autoenvscaling.trajectories.export import write_jsonl_atomic
    path = tmp_path / "sft.jsonl"
    write_jsonl_atomic(path, [{"messages": CONVERSATION, "tools": None, "metadata": {}}] * 2)
    tok = CharTokenizer()
    data = ConversationDataset(str(path), tok, {"max_length": 50, "truncation": "left"}, max_samples=1)
    item = data[0]
    assert len(data) == 1 and len(item["input_ids"]) == 50 and item["loss_mask"].sum() > 0
    assert item["position_ids"].tolist() == list(range(50))


# ---- the Pi episode ------------------------------------------------------------------------------
class FakeSandbox:
    def __init__(self, spec, *, pi_rc=0, reward="0.75"):
        self.spec, self.pi_rc, self.reward = spec, pi_rc, reward
        self.calls, self.files, self.terminated = [], {}, 0

    async def exec(self, *cmd, timeout_s=None, workdir=None, env=None):
        script = cmd[-1]
        self.calls.append((script, env))
        if "command -v curl" in script:
            return ExecResult(0, "/usr/bin/curl")
        if "npm install" in script:
            return ExecResult(0, "0.80.6")
        if "models.json" in script:
            return ExecResult(0, "PLANT_OK")
        if "pi --print" in script:
            return ExecResult(self.pi_rc, "EXIT=0")
        if cmd[0] == "python3":
            return ExecResult(0, '{"agent_ended": true, "stop_reason": "stop", "assistant_error": false, '
                                 '"content_blocks": 1, "http_status": null}')
        if script.startswith("rm -rf /tests"):
            return ExecResult(0)
        if "test.sh" in script:
            return ExecResult(0, f"RC=0\nREWARD_TXT\n{self.reward}\n")
        return ExecResult(1, "", f"unexpected: {script}")

    async def write_file(self, path, data):
        self.files[path] = data

    async def upload_dir(self, local_dir, remote_dir):
        pass

    async def terminate(self):
        self.terminated += 1


class FakeBackend:
    name = "fake"

    def __init__(self, **sandbox):
        self.sandbox, self.created = sandbox, []

    async def create(self, spec):
        self.created.append(FakeSandbox(spec, **self.sandbox))
        return self.created[-1]


RECORDING = {"tokens": list(range(10)), "logprobs": [0.0] * 4 + [-0.5] * 6,
             "loss_mask": [0, 0, 0, 0, 1, 1, 0, 0, 1, 1], "prompt_len": 4, "n_tokens": 10, "turns": 2,
             "tool_results_seen": 1, "stop_reason": None, "error": None, "weight_versions": [[3, 3], [3, 4]]}


@pytest.fixture
def task(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.sh").write_text("exit 0\n")
    (tmp_path / "task.toml").write_text('[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n')
    return tmp_path


def episode(backend, task, recording=RECORDING, **metadata):
    released = []

    async def record(episode_id):
        return recording

    async def release(episode_id):
        released.append(episode_id)

    settings = CarrySettings(proxy_public_url="https://proxy.example", network="allowlist",
                             sandbox_cpu=4, sandbox_memory_gib=8, run_id="aes-run")
    coro = run_episode(backend=backend, settings=settings, instruction="Fix the build.",
                       metadata={"image": "task:img", "task_dir": str(task), **metadata},
                       context_window=98304, max_tokens=16384, name=settings.run_id,
                       recording=record, release=release, version=2)
    return coro, released


async def test_episode_runs_pi_once_verifies_and_releases(task, monkeypatch):
    monkeypatch.setenv("AES_PROXY_API_KEY", "secret")
    backend = FakeBackend()
    coro, released = episode(backend, task)
    sample = await coro
    sandbox = backend.created[0]
    assert sample["prompt_ids"] == [0, 1, 2, 3] and sample["response_ids"] == list(range(4, 10))
    assert sample["response_mask"] == [1, 1, 0, 0, 1, 1] and sample["response_logprobs"] == [-0.5] * 6
    assert sample["reward_score"] == 0.75 and sample["num_turns"] == 2
    assert sample["extra_fields"] == {"min_global_steps": 3, "max_global_steps": 4}
    assert sandbox.spec.name.startswith("aes-run-ep-") and sandbox.spec.image == "task:img"
    assert {"proxy.example", "pypi.org"} <= set(sandbox.spec.domain_allowlist)
    assert sum("pi --print" in script for script, _ in sandbox.calls) == 1
    pi_env = next(env for script, env in sandbox.calls if "pi --print" in script)
    assert pi_env["OPENAI_API_KEY"] == "secret"
    assert sandbox.terminated == 1 and released == [sandbox.spec.name.removeprefix("aes-run-")]
    assert b"/v1/ep/" + released[0].encode() in sandbox.files["/tmp/aes-pi-agent/models.json"]


async def test_failed_episodes_are_dropped_and_cleaned_up(task):
    backend = FakeBackend(pi_rc=1)
    coro, released = episode(backend, task)
    with pytest.raises(EpisodeFailed, match="Pi process failed"):
        await coro
    assert backend.created[0].terminated == 1 and len(released) == 1
    backend = FakeBackend()
    coro, released = episode(backend, task, recording=None)
    with pytest.raises(EpisodeFailed, match="no tokens"):
        await coro
    backend = FakeBackend()
    coro, _ = episode(backend, task, image="")
    with pytest.raises(EpisodeFailed, match="no image"):
        await coro
    assert backend.created == []


def test_recordings_become_samples():
    sample = sample_from_recording({**RECORDING, "weight_versions": []}, 1.0, version=7)
    assert sample["extra_fields"] == {"min_global_steps": 7, "max_global_steps": 7}
    with pytest.raises(EpisodeFailed, match="inconsistent"):
        sample_from_recording({**RECORDING, "logprobs": [0.0]}, 1.0, version=0)
    with pytest.raises(EpisodeFailed, match="prompt_len"):
        sample_from_recording({**RECORDING, "prompt_len": 10}, 1.0, version=0)
    assert instruction_of({"metadata": {"instruction": "do it"}}) == "do it"
    assert instruction_of({"raw_prompt": [{"role": "user", "content": "row"}]}) == "row"
    with pytest.raises(EpisodeFailed):
        instruction_of({})


# ---- against verl itself (skipped unless verl is installed) -----------------------------------------
def test_every_preset_composes_and_validates_in_verl():
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_module
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import omega_conf_to_dataclass, validate_config

    for name, make in PRESETS.items():
        base = make(prompt_data="/d.jsonl", bridge_dir="/b", **PATHS)
        for config in [base, base.with_(mode="carry")] if base.mode == "bridge" else [base]:
            with initialize_config_module("verl.trainer.config", version_base=None):
                c = compose("sft_trainer_engine" if config.mode == "sft" else "ppo_trainer",
                            overrides=config.overrides())
            if config.mode == "sft":
                assert omega_conf_to_dataclass(c.engine).tensor_model_parallel_size == config.tensor_parallel
                continue
            validate_config(c, use_reference_policy=need_reference_policy(c), use_critic=need_critic(c))
            assert not need_reference_policy(c), name
            rollout = omega_conf_to_dataclass(c.actor_rollout_ref.rollout)
            assert rollout.agent.agent_loop_manager_class.endswith("verl.agent.AgentLoopManager")


def test_loops_and_trainer_register_in_verl():
    pytest.importorskip("verl")
    pytest.importorskip("transfer_queue")
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
    from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, get_trainer_cls

    import autoenvscaling.training.verl.agent
    import autoenvscaling.training.verl.bridge  # noqa: F401
    assert {"aes_pi", "aes_replay"} <= set(_agent_loop_registry)
    assert issubclass(get_trainer_cls("aes_bridge"), PPOTrainerColocateAsync)
