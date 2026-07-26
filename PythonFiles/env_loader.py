#!/usr/bin/env python3
"""Load a shell-style env file and emit shell-safe export statements."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path


def load_env_file(env_path: str | Path) -> str:
    exports: list[str] = []
    path = Path(env_path)

    if not path.is_file():
        raise FileNotFoundError(f"Environment file not found: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif value.startswith(('"', "'")):
            value = value[1:]

        exports.append(f"export {key}={shlex.quote(value)}")

    return "\n".join(exports)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: env_loader.py <env-file>")

    print(load_env_file(sys.argv[1]))
