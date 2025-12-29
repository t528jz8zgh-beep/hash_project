# funnel_hash.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from src.common import key_u64, splitmix64, permute_0_n


class SubArray:
    """Represents an array Ai, subdivided into blocks (subarrays) of size beta."""
    def __init__(self, size: int):
        self.size = int(size)
        self.slots: List[Optional[object]] = [None] * self.size
        self.occ = 0  # occupied slots

    def is_full(self) -> bool:
        return self.occ >= self.size

    def update_occ(self) -> None:
        # O(size) fallback; normally you should maintain occ.
        self.occ = sum(1 for x in self.slots if x is not None)


class TwoChoiceArray:
    """
    Two-choice hashing structure C:
      - Partition into buckets of fixed size (bucket_size).
      - For a key, consider two buckets; insert into the less occupied one if possible,
        else try the other.
    """
    def __init__(self, size: int, bucket_size: int = 16):
        self.size = int(size)
        self.bucket_size = int(bucket_size)
        self.num_buckets = max(1, self.size // self.bucket_size)
        # if size is smaller than one bucket, still keep one bucket
        self.buckets: List[List[Optional[object]]] = [
            [None] * self.bucket_size for _ in range(self.num_buckets)
        ]
        self.occupancy: List[int] = [0] * self.num_buckets

    def _hash_to_bucket(self, key: object, salt: int) -> int:
        h = splitmix64(key_u64(key) ^ (salt * 0x9E3779B97F4A7C15))
        return int(h % self.num_buckets)

    def insert(self, key: object) -> Tuple[bool, int]:
        b1 = self._hash_to_bucket(key, 1)
        b2 = self._hash_to_bucket(key, 2)

        # choose target by current occupancy (insert-time rule)
        if self.occupancy[b1] <= self.occupancy[b2]:
            target, other = b1, b2
        else:
            target, other = b2, b1

        inspections = 0
        # scan target
        for i in range(self.bucket_size):
            inspections += 1
            if self.buckets[target][i] is None:
                self.buckets[target][i] = key
                self.occupancy[target] += 1
                return True, inspections
        # scan other
        for i in range(self.bucket_size):
            inspections += 1
            if self.buckets[other][i] is None:
                self.buckets[other][i] = key
                self.occupancy[other] += 1
                return True, inspections

        return False, inspections

    def search(self, key: object) -> Tuple[bool, int]:
        b1 = self._hash_to_bucket(key, 1)
        b2 = self._hash_to_bucket(key, 2)

        inspections = 0
        # must scan both buckets (bucket choice at insert depends on occupancy at that time)
        for i in range(self.bucket_size):
            inspections += 1
            if self.buckets[b1][i] == key:
                return True, inspections
        for i in range(self.bucket_size):
            inspections += 1
            if self.buckets[b2][i] == key:
                return True, inspections
        return False, inspections


class FunnelHashing:
    """
    Funnel Hashing (greedy multi-array + special array).

    Structure:
      - Main arrays A1..A_alpha, each mapped to a key-specific block (subarray) of size beta.
      - Special array A_{alpha+1} = B (limited uniform probing) + C (two-choice hashing).

    Unified return conventions:
      insert(key) -> (success, inspections, depth)
      search(key) -> (found, inspections)

    inspections counts actual slot checks (including failed arrays), accumulated.
    """

    def __init__(self, table_size: int, delta: float):
        self.n = int(table_size)
        self.delta = float(delta)
        self.max_elements = self.n - int(np.floor(self.delta * self.n))
        self.element_count = 0

        # parameters (keep your original formula)
        self.alpha = max(1, int(np.ceil(4 * np.log2(1.0 / self.delta) + 10)))
        self.beta = max(1, int(np.ceil(2 * np.log2(1.0 / self.delta))))

        # Special array size (heuristic from your docx code; robustified)
        min_special_size = max(100, int(np.ceil(self.delta * self.n / 4)))
        max_special_size = max(1, int(self.n // 4))
        self.special_size = int(min(min_special_size, max_special_size))

        main_size = self.n - self.special_size
        self.main_arrays: List[SubArray] = []

        remaining = main_size
        for i in range(self.alpha):
            if remaining <= 0:
                break
            # exponential decay sizes, but ensure enough room for remaining levels
            target = max(self.beta, int(np.ceil(remaining / 2)))
            if i < self.alpha - 1:
                target = max(self.beta, min(target, remaining - (self.alpha - i - 1) * self.beta))
            else:
                target = remaining

            if target >= self.beta:
                self.main_arrays.append(SubArray(target))
                remaining -= target
            else:
                break

        # adjust beta to fit actual arrays
        if self.main_arrays:
            self.beta = min(self.beta, min(arr.size for arr in self.main_arrays))

        # special array: B (uniform probing limited) + C (two-choice)
        self.uniform_array_size = max(1, self.special_size // 2)
        self.uniform_array: List[Optional[object]] = [None] * self.uniform_array_size

        self.two_choice_size = max(1, self.special_size - self.uniform_array_size)
        self.two_choice_array = TwoChoiceArray(self.two_choice_size, bucket_size=16)

        # Unified stats
        self.insert_inspections: List[int] = []
        self.insert_depths: List[int] = []
        self.search_hit_inspections: List[int] = []
        self.search_miss_inspections: List[int] = []
        self.fail_insert = 0
        self.fail_search = 0

    # ---------------- core helpers ----------------

    def _subarray_index(self, key: object, array_index: int, num_subarrays: int) -> int:
        h = splitmix64(key_u64(key) ^ (array_index * 0x9E3779B97F4A7C15))
        return int(h % max(1, num_subarrays))

    def _attempt_insert_in_array(self, key: object, array_index: int) -> Tuple[bool, int]:
        """
        Try insert into main array[array_index], scanning one block (subarray) of size beta.
        Returns: (success, inspections_used_in_this_array)
        """
        if array_index >= len(self.main_arrays):
            return False, 0
        arr = self.main_arrays[array_index]
        if arr.is_full():
            return False, 0

        num_sub = max(1, arr.size // self.beta)
        sidx = self._subarray_index(key, array_index, num_sub)
        base = sidx * self.beta

        inspections = 0
        for j in range(self.beta):
            pos = base + j
            if pos >= arr.size:
                break
            inspections += 1
            if arr.slots[pos] is None:
                arr.slots[pos] = key
                arr.occ += 1
                self.element_count += 1
                return True, inspections

        # failed (subarray full)
        return False, inspections

    def _attempt_search_in_array(self, key: object, array_index: int) -> Tuple[Optional[bool], int]:
        """
        Search inside one main array's block.
        Returns:
          (True, ins)  if found
          (False, ins) if miss and we can stop early because an empty slot was encountered
          (None, ins)  if not found and no empty encountered (block full) -> must continue to next array
        """
        if array_index >= len(self.main_arrays):
            return False, 0
        arr = self.main_arrays[array_index]
        num_sub = max(1, arr.size // self.beta)
        sidx = self._subarray_index(key, array_index, num_sub)
        base = sidx * self.beta

        inspections = 0
        for j in range(self.beta):
            pos = base + j
            if pos >= arr.size:
                break
            inspections += 1
            v = arr.slots[pos]
            if v == key:
                return True, inspections
            if v is None:
                # greedy inside this block => early stop for miss
                return False, inspections
        # block full and key not found -> continue
        return None, inspections

    # ---------------- special array ----------------

    def _max_uniform_probes(self) -> int:
        # from your code: log2(log2(n)), at least 1
        if self.n <= 2:
            return 1
        return max(1, int(np.log2(max(2.0, np.log2(self.n)))))

    def _insert_in_special_array(self, key: object) -> Tuple[bool, int]:
        """
        Insert into B using limited uniform probing; if fails, insert into C (two-choice).
        Returns: (success, inspections)
        """
        inspections = 0
        m = self._max_uniform_probes()
        m = min(m, self.uniform_array_size)

        seed = key_u64(key) ^ 0xBADC0FFEE0DDF00D
        for step0 in range(m):
            pos = permute_0_n(step0, self.uniform_array_size, seed)
            inspections += 1
            if self.uniform_array[pos] is None:
                self.uniform_array[pos] = key
                self.element_count += 1
                return True, inspections

        ok, ins2 = self.two_choice_array.insert(key)
        inspections += ins2
        if ok:
            self.element_count += 1
        return ok, inspections

    def _search_in_special_array(self, key: object) -> Tuple[bool, int]:
        """
        Search in B then C. Early-stop in B if an empty is encountered (because insertion would have stopped).
        """
        inspections = 0
        m = self._max_uniform_probes()
        m = min(m, self.uniform_array_size)

        seed = key_u64(key) ^ 0xBADC0FFEE0DDF00D
        for step0 in range(m):
            pos = permute_0_n(step0, self.uniform_array_size, seed)
            inspections += 1
            v = self.uniform_array[pos]
            if v == key:
                return True, inspections
            if v is None:
                # would have inserted here instead of going to C
                return False, inspections

        found, ins2 = self.two_choice_array.search(key)
        inspections += ins2
        return found, inspections

    # ---------------- API ----------------

    def insert(self, key: object) -> Tuple[bool, int, int]:
        """
        Returns: (success, inspections, depth)
          depth here is defined as total inspections used until placement.
        """
        if self.element_count >= self.max_elements:
            self.fail_insert += 1
            return False, 0, -1

        inspections_total = 0

        # main arrays
        for a in range(len(self.main_arrays)):
            ok, ins = self._attempt_insert_in_array(key, a)
            inspections_total += ins
            if ok:
                depth = inspections_total
                self.insert_inspections.append(inspections_total)
                self.insert_depths.append(depth)
                return True, inspections_total, depth

        # special array
        ok, ins = self._insert_in_special_array(key)
        inspections_total += ins
        if ok:
            depth = inspections_total
            self.insert_inspections.append(inspections_total)
            self.insert_depths.append(depth)
            return True, inspections_total, depth

        self.fail_insert += 1
        self.insert_inspections.append(inspections_total)
        self.insert_depths.append(-1)
        return False, inspections_total, -1

    def search(self, key: object) -> Tuple[bool, int]:
        inspections_total = 0

        # main arrays with early-stop
        for a in range(len(self.main_arrays)):
            res, ins = self._attempt_search_in_array(key, a)
            inspections_total += ins
            if res is True:
                self.search_hit_inspections.append(inspections_total)
                return True, inspections_total
            if res is False:
                self.search_miss_inspections.append(inspections_total)
                return False, inspections_total
            # res is None -> continue

        # special array
        found, ins = self._search_in_special_array(key)
        inspections_total += ins
        if found:
            self.search_hit_inspections.append(inspections_total)
            return True, inspections_total
        self.search_miss_inspections.append(inspections_total)
        return False, inspections_total

    def stats(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "delta": self.delta,
            "alpha": 1.0 - self.delta,
            "alpha_param": self.alpha,
            "beta": self.beta,
            "special_size": self.special_size,
            "inserted": self.element_count,
            "fail_insert": self.fail_insert,
            "fail_search": self.fail_search,
            "insert_inspections_mean": float(np.mean(self.insert_inspections)) if self.insert_inspections else 0.0,
            "insert_inspections_max": int(np.max(self.insert_inspections)) if self.insert_inspections else 0,
            "search_hit_mean": float(np.mean(self.search_hit_inspections)) if self.search_hit_inspections else 0.0,
            "search_miss_mean": float(np.mean(self.search_miss_inspections)) if self.search_miss_inspections else 0.0,
        }

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
