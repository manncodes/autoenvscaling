"""Print or run a verl job from a preset (the same presets and arguments as Miles).

    python -m autoenvscaling.training.verl plan --preset shared-policy \\
        --hf-checkpoint <hf dir> --save <out dir> --bridge-dir <dir>
    python -m autoenvscaling.training.verl run ...        # same arguments, actually launches
    python -m autoenvscaling.training.verl presets

`--set name=value` overrides any VerlConfig field, e.g. `--set rollout_batch_size=8`;
`--override key=value` hands a Hydra override to verl unchanged. `--ref-load` is accepted so
that Miles command lines work as they are, and ignored: verl reads `--hf-checkpoint` directly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from autoenvscaling.training import ConfigError
from autoenvscaling.training.verl.config import CARRY, PRESETS, VerlConfig
from autoenvscaling.training.verl.launcher import VerlLauncher


def _coerce(current: Any, raw: str) -> Any:
    if raw.lower() == "none":
        return None
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(current, (int, float)):
        return type(current)(raw)
    if isinstance(current, list):
        return raw.split()
    return raw


def build_config(args: argparse.Namespace) -> VerlConfig:
    paths = {k: v for k, v in {"hf_checkpoint": args.hf_checkpoint, "save": args.save, "load": args.load,
                               "prompt_data": args.prompt_data,
                               "bridge_dir": args.bridge_dir and str(args.bridge_dir)}.items() if v}
    config = PRESETS[args.preset](**paths)
    if args.carry:
        config = config.with_(mode=CARRY)
    if args.num_rollout is not None:
        config = config.with_(num_rollout=args.num_rollout)
    known = {f.name for f in fields(VerlConfig)}
    for item in args.set or []:
        name, _, raw = item.partition("=")
        if name not in known:
            raise SystemExit(f"unknown VerlConfig field {name!r}")
        config = config.with_(**{name: _coerce(getattr(config, name), raw)})
    return config.with_(extra_overrides=[*config.extra_overrides, *(args.override or [])])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autoenvscaling.training.verl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("presets")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--preset", choices=sorted(PRESETS), default="shared-policy")
        p.add_argument("--hf-checkpoint")
        p.add_argument("--ref-load", help="ignored: verl loads --hf-checkpoint through mbridge")
        p.add_argument("--save")
        p.add_argument("--load", help="a global_step_<n> directory to resume from")
        p.add_argument("--prompt-data")
        p.add_argument("--carry", action="store_true",
                       help="switch to carry mode (partial rollout and over-sampling; needs --prompt-data)")
        p.add_argument("--num-rollout", type=int)
        p.add_argument("--bridge-dir", type=Path, help="bridge control directory (bridge mode)")
        p.add_argument("--set", action="append", metavar="NAME=VALUE")
        p.add_argument("--override", action="append", metavar="KEY=VALUE", help="a Hydra override for verl")
        p.add_argument("--log", type=Path, help="append the job's output to this file")
    args = parser.parse_args(argv)

    if args.command == "presets":
        for name, make in sorted(PRESETS.items()):
            print(f"{name}: {(make.__doc__ or '').strip().splitlines()[0]}")
        return 0

    config = build_config(args)
    launcher = VerlLauncher()
    problems = launcher.problems(config)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    print(launcher.plan(config))
    if args.command == "plan":
        return 1 if problems else 0
    try:
        result = launcher.run(config, log_path=args.log)
    except ConfigError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"ok={result.ok} returncode={result.returncode} checkpoint={result.checkpoint}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
