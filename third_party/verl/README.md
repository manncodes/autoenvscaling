# verl

AutoEnvScaling can train with [verl](https://github.com/verl-project/verl) instead of Miles. The
backend targets the latest release, **v0.9.1** (commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`,
20 September 2026), and uses verl unmodified: DPPO, group-mean advantages, partial rollout across
weight updates, zero-variance group filtering and the staleness bound are all native to its V1
trainer, so there is no patch.

## Installing

Use verl's v0.9.1 SGLang + Megatron image (`docker/` in that checkout; Qwen3.5/3.6 also need
`flash-linear-attention` and `mbridge`, as verl's `examples/grpo_trainer/run_qwen3_5_35b_megatron.sh`
lists), then install this repository into it:

```bash
pip install -e '/path/to/AutoEnvScaling[carry]'    # [carry]: the recording proxy (carry mode)
```

`pip install -e '.[verl]'` installs verl 0.9.1 itself, enough to plan jobs and run the tests that
compose every preset against verl's configuration (`tests/test_training_verl.py`), not to train.

## Running

The command line, presets and arguments are the same as for Miles; select the backend with
`--backend verl` or `AES_TRAIN_BACKEND=verl` (which the experiment launchers in `cmd/` honour):

```bash
aes train --backend verl plan --preset shared-policy --hf-checkpoint <hf dir> \
    --save <out dir> --bridge-dir <dir> --num-rollout 305
aes train --backend verl run ...                     # same arguments, launches
aes train --backend verl plan --preset carry-solver --prompt-data pool.jsonl ...
aes train --backend verl plan --preset cold-start --prompt-data sft.jsonl ...
```

`--set name=value` changes a `VerlConfig` field and `--override key=value` hands a Hydra override
to verl as is. verl reads the Hugging Face checkpoint directly (mbridge), so `--ref-load` is not
needed and is ignored. RL jobs run `python -m autoenvscaling.training.verl.main` (verl's
`main_ppo` with AutoEnvScaling's trainer registered); SFT runs `torchrun -m verl.trainer.sft_trainer`.
A restarted job resumes from the newest `global_step_<n>` under `--save`; `--load` names one.

## How AutoEnvScaling drives verl

- **bridge** (default; `training/verl/bridge.py`). `trainer.v1.trainer_mode=aes_bridge` is verl's
  colocated asynchronous trainer with the prompt dataset replaced by the self-play loop
  (`training/bridge.py`, the same handshake as Miles). Before each update it writes the address of
  an OpenAI-compatible front over verl's SGLang replicas to `ready_<id>.json` and waits for the
  loop's batch; each group becomes a prompt whose agent loop (`aes_replay`) re-tokenizes one
  conversation, so verl's own code builds every training field. A batch is one optimizer step
  whatever its size, each sample trains on the advantage the loop computed over its whole group,
  and the replicas are paused (not only asleep) while the trainer owns the GPUs, so a late request
  waits instead of reaching a released engine.
- **carry** (`training/verl/agent.py`, `episode.py`). verl runs the episodes: agent loop `aes_pi`
  starts a sandbox from the row's image, installs Pi and points it at the recording proxy, which
  runs as a Ray actor with verl's LLM client as its engine. `colocate_async` aborts in-flight
  generations at the weight update and continues them under the new weights, so Pi's request is
  merely slow; the recording keeps the engine's token ids and log-probabilities, which train as the
  behaviour policy (`bypass_mode`). The Pi install, terminal checks and verifier are the Miles
  carry's (`training/carry/driver.py`).
- **sft** (`training/verl/data.py`). `ConversationDataset` reads `export sft` rows as written and
  masks exactly the assistant turns of the full chat-template render.

| Miles | verl v0.9.1 |
|---|---|
| `--advantage-estimator dppo`, tv, 0.1 | `actor.policy_loss.loss_mode=dppo_tv`, `clip_ratio=0.1` |
| `hooks.center_rewards_by_group` | `algorithm.adv_estimator=grpo`, `norm_adv_by_std_in_grpo=False` |
| per-sample loss mean (Miles' default) | `actor.loss_agg_mode=seq-mean-token-mean` |
| `--use-rollout-logprobs` | `algorithm.rollout_correction.bypass_mode=True` |
| `--partial-rollout` | `trainer.v1.trainer_mode=colocate_async` |
| `check_reward_nonzero_std` | `algorithm.filter_groups` (metric `reward`), evicted groups refilled |
| `--max-weight-staleness 4` | `trainer.v1.sampler.max_off_policy_threshold=5` (a span of 5 versions) |
| 768 episodes generating | `trainer.v1.colocate_async.num_warmup_batches=2` (3 batches in flight) |
| `--use-dynamic-global-batch-size` (bridge) | mini-batch sized per batch to cover it once |
| `--ref-load` torch_dist conversion | none: mbridge loads the Hugging Face checkpoint |
| Miles patch: router guard, flush wait | verl pauses replicas during the update |

## Differences from the Miles runs

- **Truncated importance ratio.** verl's DPPO caps the importance ratio at `clip_ratio_c`
  (`dppo_ratio_cap`, 20, the DPPO paper's recommendation); the Miles patch had no cap.
- **Parallel layout.** verl's tested Qwen3.5 recipe runs padded (bshd) sequences with TP 2, EP 8,
  CP 1, because Qwen3.5/3.6's gated delta-net supports packed (THD) sequences only from
  Megatron-Core 0.18; the presets follow it (`packed_sequences=False`, `context_parallel=1`). The
  paper's Miles runs used TP 2, CP 4 with THD; set `packed_sequences=True` and `context_parallel=4`
  with a Megatron-Core that supports it.
- **Partial rollout granularity.** Miles discarded an aborted turn and regenerated it after the
  update; verl continues the same turn under the new weights, each token keeping the
  log-probability of the policy that sampled it.
- **Episode budget.** verl does not tell an episode when it is held across an update, so the carry
  budget is wall-clock (`AES_CARRY_EPISODE_TIMEOUT_S` + `AES_CARRY_MAX_PAUSED_S`) rather than
  active time plus a paused backstop. Stale groups are dropped when they finish, not evicted while
  parked.
- **Restart.** Sandboxes are named `<run id>-<episode>` (the run id comes from `--save`); a new job
  removes those of its run on start, so two jobs must not share a checkpoint directory.

## Known limitations

- Not run on GPUs yet. Every preset composes and passes verl's `validate_config` and typed config
  conversion against v0.9.1, the trainer and agent loops register inside verl, and the bridge
  batch, advantage broadcast, OpenAI front and recording proxy were exercised against verl's own
  code with fakes for the engines and sandboxes; a paid run is still needed.
- Bridge mode re-tokenizes conversations, as with Miles: the trained tokens are the chat
  template's render, not necessarily the exact tokens the engine produced.
- Evaluation does not run inside verl; evaluate saved checkpoints separately.
