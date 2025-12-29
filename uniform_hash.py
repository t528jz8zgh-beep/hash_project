# uniform_hash.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from src.common import key_u64, permute_0_n


class UniformProbingHash:
    """
    Traditional Yao-style "uniform probing":
      - For each key, define a (deterministic) random permutation of [0..n-1].
      - Insert: probe along that permutation; place at the first empty slot (greedy).
      - Search:
          * successful: probe until key found
          * unsuccessful: can stop at the first empty slot (greedy insertion property)
    """

    def __init__(self, table_size: int, delta: float):
        self.n = int(table_size)
        self.delta = float(delta)
        self.max_elements = self.n - int(np.floor(self.delta * self.n))

        self.slots: List[Optional[object]] = [None] * self.n
        self.element_count = 0

        # Unified stats (same names across algorithms)
        self.insert_inspections: List[int] = []
        self.insert_depths: List[int] = []
        self.search_hit_inspections: List[int] = []
        self.search_miss_inspections: List[int] = []
        self.fail_insert = 0
        self.fail_search = 0

    def _probe_pos(self, key: object, step0: int) -> int:
        # deterministic per-key seed
        seed = key_u64(key)
        return permute_0_n(step0, self.n, seed)

    def insert(self, key: object) -> Tuple[bool, int, int]:
        """
        Returns: (success, inspections, depth)
          inspections: number of slot checks performed
          depth: step index where we placed the key (here equals inspections)
        """
        if self.element_count >= self.max_elements:
            self.fail_insert += 1
            return False, 0, -1

        inspections = 0
        for step0 in range(self.n):
            pos = self._probe_pos(key, step0)
            inspections += 1
            if self.slots[pos] is None:
                self.slots[pos] = key
                self.element_count += 1
                depth = inspections
                self.insert_inspections.append(inspections)
                self.insert_depths.append(depth)
                return True, inspections, depth

        self.fail_insert += 1
        self.insert_inspections.append(inspections)
        self.insert_depths.append(-1)
        return False, inspections, -1

    def search(self, key: object) -> Tuple[bool, int]:
        """
        Returns: (found, inspections)
        """
        inspections = 0
        for step0 in range(self.n):
            pos = self._probe_pos(key, step0)
            inspections += 1
            v = self.slots[pos]
            if v == key:
                self.search_hit_inspections.append(inspections)
                return True, inspections
            if v is None:
                # greedy insertion => empty means key absent
                self.search_miss_inspections.append(inspections)
                return False, inspections

        self.fail_search += 1
        self.search_miss_inspections.append(self.n)
        return False, self.n

    def stats(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "delta": self.delta,
            "alpha": 1.0 - self.delta,
            "inserted": self.element_count,
            "fail_insert": self.fail_insert,
            "fail_search": self.fail_search,
            "insert_inspections_mean": float(np.mean(self.insert_inspections)) if self.insert_inspections else 0.0,
            "insert_inspections_max": int(np.max(self.insert_inspections)) if self.insert_inspections else 0,
            "insert_depth_mean": float(np.mean([d for d in self.insert_depths if d > 0])) if self.insert_depths else 0.0,
            "insert_depth_max": int(np.max(self.insert_depths)) if self.insert_depths else 0,
            "search_hit_mean": float(np.mean(self.search_hit_inspections)) if self.search_hit_inspections else 0.0,
            "search_miss_mean": float(np.mean(self.search_miss_inspections)) if self.search_miss_inspections else 0.0,
        }

    # Backward-compatible with your earlier report printing
    def get_performance_statistics(self) -> dict:
        xs = self.insert_inspections
        if not xs:
            return {
                "Number of Inserted Elements": 0,
                "Amortized Expected Probe Complexity": 0.0,
                "Worst-Case Expected Probe Complexity": 0,
                "Probe Complexity List": [],
            }
        return {
            "Number of Inserted Elements": int(self.element_count),
            "Amortized Expected Probe Complexity": float(np.mean(xs)),
            "Worst-Case Expected Probe Complexity": int(np.max(xs)),
            "Probe Complexity List": list(xs),
        }
