# src/common.py
# -*- coding: utf-8 -*-
"""
Common utilities for hashing / deterministic pseudo-random permutations.

Goal:
- deterministic across processes/machines (avoid Python's randomized hash)
- fast and scalable (avoid per-operation shuffle of [0..n))
- simple, dependency-free (no numpy required here)

You can safely use integer keys in benchmarks for maximum determinism.
"""

from __future__ import annotations
from typing import Union

MASK64 = (1 << 64) - 1


def fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit hash (deterministic)."""
    h = 1469598103934665603  # offset basis
    for b in data:
        h ^= b
        h = (h * 1099511628211) & MASK64
    return h


def key_u64(key: Union[int, str, bytes, object]) -> int:
    """
    Convert a key to a deterministic 64-bit value.

    - int: used directly (masked)
    - bytes/str: hashed with FNV-1a
    - other objects: hashed via repr() (deterministic if repr is deterministic)
    """
    if isinstance(key, int):
        return key & MASK64
    if isinstance(key, bytes):
        return fnv1a_64(key)
    if isinstance(key, str):
        return fnv1a_64(key.encode("utf-8"))
    return fnv1a_64(repr(key).encode("utf-8"))


def splitmix64(x: int) -> int:
    """SplitMix64 generator / mixer."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    return (z ^ (z >> 31)) & MASK64


def feistel_prp_pow2(x: int, bits: int, k: int, rounds: int = 4) -> int:
    """
    A small Feistel PRP permutation on domain [0, 2^bits).

    Used as a building block for cycle-walking permutations on arbitrary n.
    """
    if bits <= 0:
        return 0
    mask = (1 << bits) - 1
    x &= mask

    left_bits = bits // 2
    right_bits = bits - left_bits
    left_mask = (1 << left_bits) - 1
    right_mask = (1 << right_bits) - 1

    L = (x >> right_bits) & left_mask
    R = x & right_mask

    for r in range(rounds):
        F = splitmix64((R ^ (k + r * 0x9E3779B97F4A7C15)) & MASK64) & left_mask
        L, R = R & right_mask, (L ^ F) & left_mask

    return (((L & left_mask) << right_bits) | (R & right_mask)) & mask


def permute_0_n(i: int, n: int, key_seed: int) -> int:
    """
    Deterministic permutation of i in [0..n-1] given key_seed.

    Implements cycle-walking on a PRP over [0,2^bits), where bits=ceil(log2(n)).
    """
    if n <= 1:
        return 0
    if not (0 <= i < n):
        raise ValueError(f"permute_0_n expects 0<=i<n; got i={i}, n={n}")
    bits = (n - 1).bit_length()
    x = i
    # cycle-walking until in range
    while True:
        x = feistel_prp_pow2(x, bits, key_seed)
        if x < n:
            return x
