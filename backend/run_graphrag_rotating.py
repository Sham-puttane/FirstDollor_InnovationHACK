#!/usr/bin/env python3
"""
Run GraphRAG indexing with API key rotation.
Patches the environment variable before each retry cycle.
GraphRAG has built-in caching, so it will skip already-completed chunks.
"""

import os
import sys
import time
import subprocess
import itertools

# 6 free keys to rotate
API_KEYS = [
    "AIzaSyAu5b6DScCd2x-KMv4kl88uEXJlltVw6WM",
    "AIzaSyAeW5k5qiD3C6vy5ojppmpUpll0W6KLYMY",
    "AIzaSyAnsPL7uEz6i_PuJAU6xoq9z8nDM-BwXrk",
    "AIzaSyDillXU2EzEcHanFbPFY3AWrTIuchDL1W8",
    "AIzaSyA2vt4reAGSLWL7yKFdKnn2cEBvRNEdwMw",
    "AIzaSyCkIHCzbs0KES_A9nS505AMP7AN4jfzNC4",
]

MAX_CYCLES = 10  # max retry cycles before giving up
WAIT_BETWEEN = 30  # seconds between cycles

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    key_cycle = itertools.cycle(enumerate(API_KEYS))

    for cycle in range(1, MAX_CYCLES + 1):
        key_idx, key = next(key_cycle)
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle}/{MAX_CYCLES} — Using key {key_idx + 1}/{len(API_KEYS)} ({key[:12]}...)")
        print(f"{'='*60}\n")

        # Set the env var for this run
        env = os.environ.copy()
        env["GRAPHRAG_API_KEY"] = key

        result = subprocess.run(
            [sys.executable, "-m", "graphrag", "index", "--root", root],
            env=env,
            cwd=root,
            timeout=1800,  # 30 min max per cycle
        )

        if result.returncode == 0:
            print(f"\n  GraphRAG indexing COMPLETED successfully on cycle {cycle}!")
            return 0

        print(f"\n  Cycle {cycle} exited with code {result.returncode}")
        print(f"  Waiting {WAIT_BETWEEN}s before next cycle with next key...")
        time.sleep(WAIT_BETWEEN)

    print(f"\n  Exhausted {MAX_CYCLES} cycles. Check logs for progress.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
