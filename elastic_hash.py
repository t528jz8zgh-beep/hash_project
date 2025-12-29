#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
elastic_hash.py — Elastic Hashing (open addressing without reordering)

This implementation is designed for benchmarking / report-writing:
- Scales to large n (no per-op shuffle of [0..n))
- Deterministic probing (reproducible across runs/machines)
- Records BOTH:
    (1) insertion probes  = actual number of slot inspections during insert
    (2) search probes     = paper-defined probe complexity for present keys, i.e. φ(i,j)
        where the key is placed at h_{i,j}(x) inside subarray A_i and we map 2D probes
        into 1D indices via the injection φ from Lemma 1 (bit-encoding).

IMPORTANT:
- The paper intentionally decouples insertion probes from search probes.
- For successful queries, this file returns φ(i,j) (paper metric) in search().
- For negative queries (key not present), the paper states there is "no interesting notion"
  of probe complexity. We therefore expose search_miss() separately (optional) and the
  benchmark can skip Elastic misses by default.

Reference: "Optimal Bounds for Open Addressing Without Reordering" (Elastic Hashing section).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# Prefer src/common.py if user placed it under src/
try:
    from src.common import key_u64, splitmix64
except Exception:
    # fallback if common.py is at project root
    from common import key_u64, splitmix64  # type: ignore


MASK64 = (1 << 64) - 1


def _u64(x: int) -> int:
    return x & MASK64


def phi(i: int, j: int) -> int:
    """
    Injection φ: Z^+ × Z^+ → Z^+ from Lemma 1 in the paper.

    If i has binary a1 a2 ... ap (MSB→LSB), and j has binary b1 b2 ... bq,
    construct φ(i,j) binary representation as:
        1 0 b1 0 b2 0 b3 ... 0 bq 1 00 a1 a2 ... ap

    This is injective and satisfies φ(i,j) <= O(i · j^2).
    """
    if i <= 0 or j <= 0:
        raise ValueError("phi expects positive integers i,j")

    a = bin(i)[2:]  # bits of i
    b = bin(j)[2:]  # bits of j

    # Build bitstring exactly as in the paper.
    bits = ["1"]
    for bit in b:
        bits.append("0")
        bits.append(bit)
    bits.append("1")
    bits.append("0")
    bits.append("0")
    bits.append(a)
    return int("".join(bits), 2)


@dataclass
class Placement:
    level: int  # 1-indexed level i
    j: int      # 1-indexed probe number j within that level
    pos: int    # absolute position in the full table


class ElasticHash:
    """
    Elastic hashing table.

    Public API (used by bench_csv.py):
      - insert(key) -> (ok: bool, insertion_probes: int)
      - search(key) -> (found: bool, search_probes: int)
          * for found=True: returns φ(i,j) (paper-defined probe complexity)
          * for found=False: returns -1
      - search_miss(key, max_probes=...) -> (found: bool, miss_probe_count: int)
          * optional engineering-only miss search (NOT paper-defined)
    """

    def __init__(self, table_size: int, delta: float, c: int = 4, seed: int = 0):
        if table_size <= 0:
            raise ValueError("table_size must be positive")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1)")

        self.n = int(table_size)
        self.delta = float(delta)
        self.alpha = 1.0 - self.delta
        self.c = int(c)
        self.seed = int(seed)

        # Maximum number of insertions supported (paper uses n - floor(delta*n))
        self.max_elements = self.n - int(math.floor(self.delta * self.n))

        # Main slots array
        self.slots: List[Optional[Any]] = [None] * self.n

        # Build subarrays A1, A2, ... with geometric decay and disjoint ranges
        self.level_ranges: List[Tuple[int, int]] = self._build_levels()
        self.L = len(self.level_ranges)
        self.level_occ: List[int] = [0] * self.L

        # Batch plan: B0 fills A1 to 0.75|A1|; Bi uses Ai and Ai+1
        self.batch_plan: List[Tuple[int, int]] = self._compute_batch_plan()
        self.batch_idx = 0
        self.batch_used = 0

        # Placement map: key -> placement (level i, j, position)
        self.placement: Dict[Any, Placement] = {}

        # Total inserted
        self.element_count = 0

    # ----------------------------- structure -----------------------------

    def _build_levels(self) -> List[Tuple[int, int]]:
        """
        Build disjoint ranges for A1..A_L.

        Paper uses |A_{i+1}| = |A_i|/2 ± 1 and sum sizes = n.
        We approximate with halving sizes and put the remainder in the last level.
        """
        remaining = self.n
        sizes: List[int] = []
        s = max(1, self.n // 2)
        while remaining > 0:
            take = min(s, remaining)
            sizes.append(take)
            remaining -= take
            s = max(1, s // 2)

        ranges: List[Tuple[int, int]] = []
        cur = 0
        for sz in sizes:
            ranges.append((cur, cur + sz))
            cur += sz
        # ensure covers exactly n
        if ranges and ranges[-1][1] != self.n:
            ranges[-1] = (ranges[-1][0], self.n)
        return ranges

    def _level_size(self, level: int) -> int:
        s, e = self.level_ranges[level - 1]
        return e - s

    def _free_fraction(self, level: int) -> float:
        sz = self._level_size(level)
        return float(sz - self.level_occ[level - 1]) / float(sz) if sz > 0 else 0.0

    def _compute_batch_plan(self) -> List[Tuple[int, int]]:
        """
        Returns a list of (i, batch_size):
          - entry (0, k): B0 inserts k items into A1 only
          - entry (i, k) for i>=1: Bi inserts k items into Ai and Ai+1

        For Bi (i>=1), paper gives:
          |Bi| = |Ai| - floor(delta*|Ai|/2) - [0.75|Ai| + 0.75|A_{i+1}|]
        (up to floors). We follow this closely.

        The last batch may not fully run if we reach max_elements early.
        """
        if self.L == 0:
            return []

        plan: List[Tuple[int, int]] = []

        # B0: fill A1 to 0.75|A1|
        a1 = self._level_size(1)
        b0 = int(math.floor(0.75 * a1))
        plan.append((0, max(0, b0)))

        # Bi for i=1..L-1
        for i in range(1, self.L):
            ai = self._level_size(i)
            ai1 = self._level_size(i + 1)
            target_ai = ai - int(math.floor(self.delta * ai / 2.0))
            baseline = int(math.floor(0.75 * ai)) + int(math.floor(0.75 * ai1))
            bi = target_ai - baseline
            plan.append((i, max(0, bi)))

        return plan

    # ----------------------------- probing -----------------------------

    def _probe_pos(self, key64: int, level: int, j: int) -> int:
        """
        Deterministic "random slot" h_{i,j}(x) inside A_i.

        We do NOT generate a full permutation. Instead we map (key, level, j)
        to a pseudo-random slot via SplitMix64 and mod by level size.
        """
        s, e = self.level_ranges[level - 1]
        sz = e - s
        if sz <= 0:
            return s

        x = _u64(key64 ^ (level * 0x9E3779B97F4A7C15) ^ (j * 0xD1B54A32D192ED03) ^ self.seed)
        r = splitmix64(x)
        return s + int(r % sz)

    def _f(self, eps: float) -> int:
        """
        f(ε) = c · min( log^2(1/ε), log(1/δ) )   (paper, page 5)

        We use log base 2 (any constant base differs by constant factors).
        """
        if eps <= 0.0:
            return 1
        if eps >= 1.0:
            return 1

        log_eps = math.log2(1.0 / eps)
        log_del = math.log2(1.0 / self.delta)
        val = self.c * min((log_eps ** 2), log_del)
        return max(1, int(math.ceil(val)))

    # ----------------------------- insertion helpers -----------------------------

    def _advance_batch_if_needed(self) -> None:
        while self.batch_idx < len(self.batch_plan):
            _, bsz = self.batch_plan[self.batch_idx]
            if self.batch_used < bsz:
                return
            self.batch_idx += 1
            self.batch_used = 0

    def _insert_into_level(self, key: Any, key64: int, level: int, j_start: int = 1, j_limit: Optional[int] = None) -> Tuple[bool, int, int, int]:
        """
        Try inserting key into a given level A_level by probing h_{level,j}(key).

        Returns:
          (ok, probes_used, j_final, pos)
        """
        sz = self._level_size(level)
        if sz <= 0:
            return False, 0, -1, -1
        # If level is already full, fail fast
        if self.level_occ[level - 1] >= sz:
            return False, 0, -1, -1

        probes = 0
        j = j_start
        # safety bound: do not loop forever even if probing repeats positions
        max_iter = j_limit if j_limit is not None else (sz * 4)

        while (j - j_start) < max_iter:
            pos = self._probe_pos(key64, level, j)
            probes += 1
            if self.slots[pos] is None:
                self.slots[pos] = key
                self.level_occ[level - 1] += 1
                self.element_count += 1
                self.placement[key] = Placement(level=level, j=j, pos=pos)
                return True, probes, j, pos
            j += 1

        return False, probes, -1, -1

    def insert(self, key: Any) -> Tuple[bool, int]:
        """
        Insert key, returning (ok, insertion_probes).

        insertion_probes = actual slot inspections performed by this insert operation.
        """
        if self.element_count >= self.max_elements:
            return False, -1
        if key in self.placement:
            # already present
            return True, 0

        self._advance_batch_if_needed()
        key64 = key_u64(key)

        # If batch plan is exhausted, always insert into last level (fallback)
        if self.batch_idx >= len(self.batch_plan):
            ok, probes, _, _ = self._insert_into_level(key, key64, self.L)
            return ok, probes

        i, _ = self.batch_plan[self.batch_idx]
        self.batch_used += 1

        # B0: only A1
        if i == 0:
            ok, probes, _, _ = self._insert_into_level(key, key64, 1)
            return ok, probes

        # Bi: uses Ai and Ai+1
        Ai = i
        Ai1 = i + 1

        eps1 = self._free_fraction(Ai)
        eps2 = self._free_fraction(Ai1)

        probes_total = 0

        # Case 1: eps1 > delta/2 and eps2 > 0.25
        if (eps1 > (self.delta / 2.0)) and (eps2 > 0.25):
            # try up to f(eps1) probes in Ai
            limit = self._f(eps1)
            ok, probes, _, _ = self._insert_into_level(key, key64, Ai, j_start=1, j_limit=limit)
            probes_total += probes
            if ok:
                return True, probes_total

            # otherwise insert into Ai+1 (full uniform probing in Ai+1)
            ok2, probes2, _, _ = self._insert_into_level(key, key64, Ai1, j_start=1, j_limit=None)
            probes_total += probes2
            return ok2, probes_total

        # Case 2: eps1 <= delta/2  => must go to Ai+1
        if eps1 <= (self.delta / 2.0):
            ok, probes, _, _ = self._insert_into_level(key, key64, Ai1, j_start=1, j_limit=None)
            probes_total += probes
            return ok, probes_total

        # Case 3: eps2 <= 0.25  => must go to Ai
        ok, probes, _, _ = self._insert_into_level(key, key64, Ai, j_start=1, j_limit=None)
        probes_total += probes
        return ok, probes_total

    # ----------------------------- search -----------------------------

    def search(self, key: Any) -> Tuple[bool, int]:
        """
        Successful-query probe complexity (paper-defined):
          return φ(i,j) for a present key stored at h_{i,j}(key) within A_i.

        For keys not present, returns (False, -1).
        """
        plc = self.placement.get(key)
        if plc is None:
            return False, -1

        # Sanity: verify the stored position is still correct
        if self.slots[plc.pos] != key:
            # if something went wrong, fall back to "not found"
            return False, -1

        return True, phi(plc.level, plc.j)

    def search_miss(self, key: Any, max_level: Optional[int] = None, max_j: int = 8) -> Tuple[bool, int]:
        """
        Engineering-only miss search (NOT paper-defined).

        We probe a limited set of (level, j) pairs and return the number of slot inspections.
        This is useful only if you want to record something for negative queries;
        by default the benchmark should SKIP Elastic misses per the paper discussion.
        """
        key64 = key_u64(key)
        probes = 0
        L = self.L if max_level is None else max(1, min(self.L, int(max_level)))

        for lvl in range(1, L + 1):
            for j in range(1, max_j + 1):
                pos = self._probe_pos(key64, lvl, j)
                probes += 1
                if self.slots[pos] == key:
                    return True, probes
        return False, probes

    # ----------------------------- debug helpers -----------------------------

    def inserted_fraction(self) -> float:
        return float(self.element_count) / float(self.n)

    def __len__(self) -> int:
        return self.element_count
