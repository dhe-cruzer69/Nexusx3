#!/usr/bin/env python3
"""Reproducible microbenchmark for the ARX model registry.

Measures expert-model lookup, the lookup the router performs for every request,
against a naive linear scan over the same collection.

    python bench/bench_models.py
    python bench/bench_models.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arx.ai.models import all_experts, get_expert  # noqa: E402

ITERATIONS = 20_000
SAMPLES = 7
GATES = {"expert_lookup": {"kind": "budget", "budget_ns": 8000}}


def time_call(fn, iterations: int, samples: int) -> float:
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        times.append((time.perf_counter() - start) / iterations)
    return statistics.median(times)


_experts = all_experts()
_TARGET = _experts[0].name if _experts else "unknown"


def subject_lookup():
    return get_expert(_TARGET)


def naive_lookup():
    for expert in _experts:
        if expert.name == _TARGET:
            return expert
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()

    fast = time_call(subject_lookup, args.iterations, SAMPLES)
    slow = time_call(naive_lookup, args.iterations, SAMPLES)
    speedup = slow / fast if fast else 0.0
    ns = fast * 1e9
    budget = GATES["expert_lookup"]["budget_ns"]
    ok = ns <= budget
    result = {
        "case": "expert_lookup",
        "iterations": args.iterations,
        "registered_experts": len(_experts),
        "subject_seconds_per_op": round(fast, 9),
        "naive_seconds_per_op": round(slow, 9),
        "subject_ops_per_second": round(1 / fast) if fast else None,
        "measured_speedup_vs_naive": round(speedup, 2),
        "gate": f"<= {budget} ns/op",
        "pass": ok,
    }
    report = {
        "repo": "arx-ai",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "results": [result],
        "passed": ok,
        "failures": [] if ok else [f"expert_lookup {ns:.0f} ns/op exceeds {budget} ns/op"],
        "note": "Speedup column is informational; the gate is an absolute latency budget.",
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"arx-ai benchmark · Python {report['python']} · {report['platform']}")
        print(f"{result['case']:<18}{result['subject_ops_per_second']:>12,} ops/s"
              f"{ns:>9.0f} ns  {'PASS' if ok else 'FAIL'}")
        print(f"registered experts: {len(_experts)} | vs naive scan: {speedup:.2f}x")
        print("\nGate met." if ok else report["failures"][0])

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
