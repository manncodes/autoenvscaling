# Shared helpers for the experiment launchers. Source it from a launcher:
#
#   source "$(dirname "$0")/../common.sh"
#
# Every command is echoed and its output is appended to one log file per launch, so a run can be
# followed live with `tail -f <log>`. Set DRY_RUN=1 to print the commands without running them.

set -euo pipefail

AES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$AES_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || command -v python)"; fi
DRY_RUN="${DRY_RUN:-0}"
AES_CLI=("$PYTHON" -m autoenvscaling.cli)
export AES_TRAIN_BACKEND="${AES_TRAIN_BACKEND:-miles}"   # or verl: `aes train` reads it
TRAIN_PIDS=()

start_log() {  # start_log <experiment name>
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  AES_RUN_DIR="${AES_RUN_DIR:-$AES_ROOT/runs/$1/$stamp}"
  mkdir -p "$AES_RUN_DIR"
  LOG="$AES_RUN_DIR/launcher.log"
  echo "run dir: $AES_RUN_DIR"
  echo "log:     $LOG   (follow with: tail -f $LOG)"
}

run() {  # run <command...>: echo, then run with output teed to the log
  echo "+ $*" | tee -a "$LOG"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  "$@" 2>&1 | tee -a "$LOG"
}

trace() {  # AES_TRACE=<file> records each CLI call's exact arguments, one shell-quoted line each
  if [[ -n "${AES_TRACE:-}" ]]; then printf '%q ' "$@" >> "$AES_TRACE"; echo >> "$AES_TRACE"; fi
}

aes() {  # aes <command> [args...]: the package CLI (`autoenvscaling ...`)
  echo "+ autoenvscaling $*" | tee -a "$LOG"
  trace "$@"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  "${AES_CLI[@]}" "$@" 2>&1 | tee -a "$LOG"
}

require_for_miles() {  # require_for_miles VAR...: torch_dist weights; verl reads Hugging Face ones
  if [[ "$AES_TRAIN_BACKEND" == "miles" ]]; then require_env "$@"; fi
}

not_in_release() {  # not_in_release <step> <what is missing>: say so and carry on
  echo "- $1: not in this release yet ($2)" | tee -a "$LOG"
}

note() {
  echo "# $*" | tee -a "$LOG"
}

require_env() {  # require_env VAR...: fail early instead of producing a wall of failed trials
  local missing=0 name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then echo "error: $name is not set" >&2; missing=1; fi
  done
  if [[ "$DRY_RUN" != "1" && "$missing" == "1" ]]; then exit 2; fi
}

num_rollout() {  # num_rollout <config> <steps> <schedule mode>: `train --num-rollout` for a loop
  "${AES_CLI[@]}" schedule --config "$1" --steps "$2" --mode "$3" \
    | sed -n 's/.*--num-rollout): *//p'
}

train_start() {  # train_start <preset> <run dir> <num rollouts> [train args...]
  # Checks the training job (Miles or verl, AES_TRAIN_BACKEND) with `train plan`, which stops the
  # launcher if a backend setting or a path is missing, then starts it in bridge mode in the
  # background; the self-play loop that feeds it runs in the foreground. The job's own output
  # goes to <run dir>/<backend>.log.
  local preset="$1" dir="$2" n="$3"
  shift 3
  local args=(--preset "$preset" --bridge-dir "$dir/bridge" --save "$dir/checkpoints"
              --num-rollout "$n" --log "$dir/$AES_TRAIN_BACKEND.log" "$@")
  aes train plan "${args[@]}"
  echo "+ autoenvscaling train run ${args[*]} &" | tee -a "$LOG"
  trace train run "${args[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  mkdir -p "$dir"
  "${AES_CLI[@]}" train run "${args[@]}" >>"$LOG" 2>&1 &
  TRAIN_PIDS+=("$!")
}

train_wait() {  # wait for the training jobs started by train_start
  local pid
  for pid in ${TRAIN_PIDS[@]+"${TRAIN_PIDS[@]}"}; do wait "$pid"; done
  TRAIN_PIDS=()
}

harness_for() {  # harness_for <model>: the coding agent each proposer runs in (Appendix D.2)
  case "$1" in
    claude-*) echo "claude-code" ;;
    gpt-*) echo "codex" ;;
    *) echo "pi" ;;
  esac
}
