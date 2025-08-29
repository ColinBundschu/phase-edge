"""Shared CLI helpers."""
from __future__ import annotations

from typing import Dict

__all__ = ["parse_counts_arg"]


def parse_counts_arg(s: str) -> Dict[str, int]:
    """Parse 'Fe:54,Mn:54' -> {'Fe': 54, 'Mn': 54} (whitespace-tolerant).

    The argument is a comma-separated list of ``element:count`` pairs. Whitespace
    around tokens is ignored. Duplicate elements, missing counts, and negative
    values raise ``ValueError``. At least one pair is required.
    """
    out: Dict[str, int] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if ":" not in kv:
            raise ValueError(f"Bad counts token '{kv}' (expected 'El:INT').")
        k, v = kv.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Empty element in counts token '{kv}'.")
        if k in out:
            raise ValueError(f"Duplicate element '{k}' in counts.")
        iv = int(v)
        if iv < 0:
            raise ValueError(f"Negative count for '{k}': {iv}")
        out[k] = iv
    if not out:
        raise ValueError("Empty counts string.")
    return out
