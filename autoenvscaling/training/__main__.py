"""`aes train [--backend miles|verl] plan|run|presets ...`

Hands the rest of the command line to the backend's own CLI (`python -m
autoenvscaling.training.miles` or `.verl`); both take the same presets and arguments. Without
`--backend`, AES_TRAIN_BACKEND decides, and Miles is the default.
"""

from __future__ import annotations

import importlib
import os
import sys

BACKENDS = {"miles": "autoenvscaling.training.miles.__main__",
            "verl": "autoenvscaling.training.verl.__main__"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    backend = os.environ.get("AES_TRAIN_BACKEND", "miles")
    if argv[:1] == ["--backend"] and len(argv) > 1:
        backend, argv = argv[1], argv[2:]
    elif argv and argv[0].startswith("--backend="):
        backend, argv = argv[0].partition("=")[2], argv[1:]
    if backend not in BACKENDS:
        print(f"unknown training backend {backend!r}; expected one of {sorted(BACKENDS)}", file=sys.stderr)
        return 2
    return importlib.import_module(BACKENDS[backend]).main(argv) or 0


if __name__ == "__main__":
    raise SystemExit(main())
