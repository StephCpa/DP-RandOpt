#!/usr/bin/env python3
"""Retry a resumable score-table command after transient worker/OOM failures."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--sleep-seconds", type=float, default=30.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("provide a command after --")
    for attempt in range(1, args.max_retries + 2):
        print(f"[resumable] attempt {attempt}/{args.max_retries + 1}: {' '.join(command)}", flush=True)
        rc = subprocess.run(command).returncode
        if rc == 0:
            return 0
        if attempt > args.max_retries:
            print(f"[resumable] giving up after {attempt} attempts (rc={rc})", flush=True)
            return rc
        print(f"[resumable] command failed (rc={rc}); retrying in {args.sleep_seconds:g}s", flush=True)
        time.sleep(args.sleep_seconds)
    return 1


if __name__ == "__main__":
    sys.exit(main())
