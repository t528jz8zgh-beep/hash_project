#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bench_csv.py — Unified benchmark runner for Uniform / Elastic / Funnel hashing.

Design goals:
- Same measurement protocol across algorithms
- Stream results to CSV incrementally (file appears immediately)
- Report BOTH costs:
    * insertion probes (actual slot inspections during insert)
    * successful-query probe complexity (paper-defined for Elastic: φ(i,j))

Notes about Elastic:
- Elastic.search(key) returns φ(i,j) for present keys (paper metric).
- The paper states there is "no interesting notion" of probe complexity for missing keys.
  Therefore, by default we SKIP Elastic misses even when --include_miss is enabled.
  You can enable a limited engineering-only miss routine with --elastic_miss_mode limited.

Typical usage (quick, produces CSV fast):
  cd ~/Desktop/hash_project
  python3 bench_csv.py --algos uniform elastic funnel --n_list 20000 --delta_list 0.10 0.05 --trials 1 --q 2000 --out_csv quick.csv

Alpha sweep (fixed n, vary alpha=1-delta):
  python3 bench_csv.py --algos uniform elastic funnel --n_list 50000 --delta_list 0.20 0.10 0.05 0.02 --trials 3 --q 3000 --out_csv alpha_sweep.csv

N sweep (fixed alpha, vary n):
  python3 bench_csv.py --algos uniform elastic funnel --n_list 20000 40000 80000 160000 --delta_list 0.05 --trials 3 --q 3000 --out_csv n_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import algorithms (bench_csv.py should sit beside these files)
from uniform_hash import UniformProbingHash
from elastic_hash import ElasticHash
from funnel_hash import FunnelHashing


# ------------------------------ utils ------------------------------

def now_ns() -> int:
    return time.perf_counter_ns()

def pct(xs: Sequence[int], p: float) -> float:
    """Percentile with linear interpolation (p in [0,100])."""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if p <= 0:
        return float(ys[0])
    if p >= 100:
        return float(ys[-1])
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f]) * (c - k) + float(ys[c]) * (k - f)

def summarize_costs(costs: Sequence[int]) -> Dict[str, float]:
    if not costs:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(statistics.fmean(costs)),
        "p50": pct(costs, 50),
        "p90": pct(costs, 90),
        "p99": pct(costs, 99),
        "max": float(max(costs)),
    }

def ensure_parent(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def instantiate(cls, n: int, delta: float, seed: int) -> Any:
    """
    Instantiate an algorithm class with best-effort signature matching.
    Works even if your __init__ uses different parameter names.
    """
    sig = inspect.signature(cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if name in ("table_size", "n", "size", "m"):
            kwargs[name] = int(n)
        elif name in ("delta", "d", "eps"):
            kwargs[name] = float(delta)
        elif name == "seed":
            kwargs[name] = int(seed)
        # keep other parameters at defaults (e.g., c)
    try:
        return cls(**kwargs)
    except TypeError:
        # fallback: positional
        return cls(n, delta)

def algo_map() -> Dict[str, Any]:
    return {
        "uniform": UniformProbingHash,
        "elastic": ElasticHash,
        "funnel": FunnelHashing,
    }


def unpack_ok_cost(ret: Any, *, default_ok: bool = False, default_cost: int = 0) -> Tuple[bool, int]:
    """Unpack (ok, cost) from various possible return shapes.

    Some student implementations return extra debug fields, e.g.:
      - (ok, cost, pos)
      - (ok, cost, meta_dict)
    This helper keeps the benchmark runner robust by taking only the first two.
    """
    if isinstance(ret, (tuple, list)):
        if len(ret) == 0:
            return bool(default_ok), int(default_cost)
        ok = bool(ret[0])
        cost = int(ret[1]) if len(ret) >= 2 else int(default_cost)
        return ok, cost
    if isinstance(ret, bool):
        return bool(ret), int(default_cost)
    # If someone returns cost-only (int/float), treat as success.
    try:
        return True, int(ret)
    except Exception:
        return bool(default_ok), int(default_cost)


# ------------------------------ benchmark core ------------------------------

def run_one(algo_name: str, n: int, delta: float, trial: int, q: int, seed: int, include_miss: bool, elastic_miss_mode: str) -> Dict[str, Any]:
    cls = algo_map()[algo_name]
    rng = random.Random((seed & 0xFFFFFFFF) ^ (trial * 0x9E3779B9) ^ (n * 0x85EBCA6B))

    H = instantiate(cls, n=n, delta=delta, seed=seed + trial)

    max_elements = int(getattr(H, "max_elements", n - int(math.floor(delta * n))))
    keys: List[int] = []

    # deterministic keys (avoid Python's randomized hash on strings/objects)
    key_base = (seed + trial + 1) * 10**9

    insert_costs: List[int] = []
    insert_fail = 0

    t0 = now_ns()
    for i in range(max_elements):
        k = key_base + i
        ok, cost = unpack_ok_cost(H.insert(k))
        if not ok:
            insert_fail += 1
            break
        keys.append(k)
        insert_costs.append(int(cost))
    t1 = now_ns()

    inserted = len(keys)
    insert_ns = t1 - t0
    insert_ns_per_op = (insert_ns / inserted) if inserted > 0 else float("nan")
    insert_stats = summarize_costs(insert_costs)

    # Successful search (hits): sample from inserted keys
    hit_q = min(q, inserted)
    hit_costs: List[int] = []
    hit_found = 0

    t2 = now_ns()
    if hit_q > 0:
        sample_keys = rng.sample(keys, k=hit_q) if hit_q < inserted else list(keys)
        for k in sample_keys:
            found, cost = unpack_ok_cost(H.search(k))
            if found:
                hit_found += 1
                hit_costs.append(int(cost))
            else:
                # should not happen; treat as miss
                hit_costs.append(-1)
    t3 = now_ns()

    hit_ns = t3 - t2
    hit_ns_per_op = (hit_ns / hit_q) if hit_q > 0 else float("nan")
    hit_stats = summarize_costs([c for c in hit_costs if c >= 0])

    # Miss search (optional)
    miss_q = q if include_miss else 0
    miss_costs: List[int] = []
    miss_found = 0
    miss_ns_per_op = float("nan")
    miss_stats = {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    miss_notes = ""

    if include_miss and miss_q > 0:
        # Generate keys guaranteed not inserted (far away from key_base range)
        miss_base = (seed + trial + 12345) * 10**15
        miss_keys = [miss_base + i for i in range(miss_q)]

        # Elastic: default skip misses per paper
        if algo_name == "elastic" and elastic_miss_mode == "skip":
            miss_notes = "elastic_miss_skipped"
        else:
            t4 = now_ns()
            for mk in miss_keys:
                if algo_name == "elastic" and hasattr(H, "search_miss"):
                    found, cost = unpack_ok_cost(H.search_miss(mk) if elastic_miss_mode == "limited" else H.search(mk))
                else:
                    found, cost = unpack_ok_cost(H.search(mk))
                if found:
                    miss_found += 1
                miss_costs.append(int(cost) if isinstance(cost, int) else int(cost))
            t5 = now_ns()
            miss_ns = t5 - t4
            miss_ns_per_op = miss_ns / miss_q
            miss_stats = summarize_costs(miss_costs)

    row: Dict[str, Any] = {
        "algo": algo_name,
        "n": n,
        "delta": delta,
        "alpha": 1.0 - delta,
        "trial": trial,
        "seed": seed,
        "max_elements": max_elements,
        "inserted": inserted,
        "insert_fail": insert_fail,
        "insert_ns_total": insert_ns,
        "insert_ns_per_op": insert_ns_per_op,
        "insert_probes_mean": insert_stats["mean"],
        "insert_probes_p50": insert_stats["p50"],
        "insert_probes_p90": insert_stats["p90"],
        "insert_probes_p99": insert_stats["p99"],
        "insert_probes_max": insert_stats["max"],
        "hit_q": hit_q,
        "hit_found": hit_found,
        "hit_ns_total": hit_ns,
        "hit_ns_per_op": hit_ns_per_op,
        # For Elastic, these probes are φ(i,j) (paper-defined). For others, they are actual probes.
        "hit_probes_mean": hit_stats["mean"],
        "hit_probes_p50": hit_stats["p50"],
        "hit_probes_p90": hit_stats["p90"],
        "hit_probes_p99": hit_stats["p99"],
        "hit_probes_max": hit_stats["max"],
        "miss_q": miss_q,
        "miss_found": miss_found,
        "miss_ns_per_op": miss_ns_per_op,
        "miss_probes_mean": miss_stats["mean"],
        "miss_probes_p50": miss_stats["p50"],
        "miss_probes_p90": miss_stats["p90"],
        "miss_probes_p99": miss_stats["p99"],
        "miss_probes_max": miss_stats["max"],
        "notes": miss_notes,
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["uniform", "elastic", "funnel"], choices=["uniform", "elastic", "funnel"])
    ap.add_argument("--n_list", nargs="*", type=int, default=[], help="explicit n values, e.g. 20000 40000")
    ap.add_argument("--n_exp_list", nargs="*", type=int, default=[], help="exponents for n=2^k, e.g. 16 18 20")
    ap.add_argument("--delta_list", nargs="+", type=float, required=True, help="e.g. 0.2 0.1 0.05")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--q", type=int, default=2000, help="number of hit queries, and miss queries if --include_miss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include_miss", action="store_true", help="also run miss queries (Elastic skipped by default)")
    ap.add_argument("--elastic_miss_mode", choices=["skip", "limited"], default="skip")
    ap.add_argument("--out_csv", type=str, default="results.csv")
    args = ap.parse_args()

    n_values: List[int] = []
    n_values.extend(args.n_list)
    n_values.extend([2 ** k for k in args.n_exp_list])
    if not n_values:
        raise SystemExit("Please provide --n_list and/or --n_exp_list")

    ensure_parent(args.out_csv)

    fieldnames = [
        "algo", "n", "delta", "alpha", "trial", "seed",
        "max_elements", "inserted", "insert_fail",
        "insert_ns_total", "insert_ns_per_op",
        "insert_probes_mean", "insert_probes_p50", "insert_probes_p90", "insert_probes_p99", "insert_probes_max",
        "hit_q", "hit_found", "hit_ns_total", "hit_ns_per_op",
        "hit_probes_mean", "hit_probes_p50", "hit_probes_p90", "hit_probes_p99", "hit_probes_max",
        "miss_q", "miss_found", "miss_ns_per_op",
        "miss_probes_mean", "miss_probes_p50", "miss_probes_p90", "miss_probes_p99", "miss_probes_max",
        "notes",
    ]

    # Create CSV immediately (so you can "ls -la" and see it right away)
    new_file = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
            f.flush()

        for trial in range(args.trials):
            for n in n_values:
                for delta in args.delta_list:
                    for algo in args.algos:
                        print(f"[trial={trial}] algo={algo} n={n} delta={delta}")
                        row = run_one(
                            algo_name=algo,
                            n=n,
                            delta=float(delta),
                            trial=trial,
                            q=args.q,
                            seed=args.seed,
                            include_miss=args.include_miss,
                            elastic_miss_mode=args.elastic_miss_mode,
                        )
                        w.writerow(row)
                        f.flush()

    print(f"CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()
