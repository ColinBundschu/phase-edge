__all__ = ["parse_counts_arg", "parse_cutoffs_arg", "parse_subl_token", "parse_mix_item"]

import re

from phaseedge.jobs.ensure_snapshots_compositions import MixtureElement
from phaseedge.schemas.sublattice import SublatticeSpec


def parse_counts_arg(s: str) -> dict[str, int]:
    """Parse 'Fe:54,Mn:54' -> {'Fe': 54, 'Mn': 54} (whitespace-tolerant)."""
    out: dict[str, int] = {}
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


def parse_cutoffs_arg(s: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if ":" not in kv:
            raise ValueError(f"Bad cutoffs token '{kv}' (expected 'ORDER:VALUE').")
        k, v = kv.split(":", 1)
        out[int(k.strip())] = float(v.strip())
    if not out:
        raise ValueError("Empty --cutoffs.")
    return out


def parse_subl_token(v: str) -> SublatticeSpec:
    """
    Parse one subl token like:
      subl=replace=Mg,counts=Fe:54,Mn:46

    Accepts ',' or ';' as intra-token separators, but *counts* may contain commas.
    Strategy: regex for replace=... and counts=... (counts consumes to end).
    """
    v = v.strip()

    m_replace = re.search(r'(?:^|[;,])\s*replace\s*=\s*([^,;]+)', v)
    m_counts = re.search(r'(?:^|[;,])\s*counts\s*=\s*(.+)$', v)

    if not m_replace or not m_counts:
        raise ValueError(
            "subl requires both 'replace' and 'counts'. "
            "Example: subl=replace=Mg,counts=Fe:64,Mg:192"
        )

    replace = m_replace.group(1).strip()
    counts_str = m_counts.group(1).strip()
    if not replace or not counts_str:
        raise ValueError("Empty replace or counts in subl=...")

    counts = parse_counts_arg(counts_str)
    return SublatticeSpec(replace_element=replace, counts=counts)


def parse_mix_item(s: str) -> MixtureElement:
    """
    Parse one --mix item like:
      "K=50;seed=123;subl=replace=Mg,counts=Fe:64,Mg:192; subl=replace=Al,counts=Al:256"
    Keys (semicolon-separated):
      - subl=... (repeatable): each defines one SublatticeSpec
      - K=...   (required): integer number of snapshots
      - seed=... (required): integer seed for this mixture element
    """
    parts = [p.strip() for p in s.split(";") if p.strip()]
    sublats: list[SublatticeSpec] = []
    K: int | None = None
    seed: int | None = None

    for p in parts:
        if p.startswith("subl="):
            sublats.append(parse_subl_token(p[len("subl="):]))
            continue
        if "=" not in p:
            raise ValueError(f"Bad --mix token '{p}' (expected key=value or subl=...)")
        k, v = p.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "k":
            K = int(v)
        elif k == "seed":
            seed = int(v)
        else:
            raise ValueError(f"Unknown key '{k}' in --mix item. Allowed: subl=..., K=..., seed=...")

    if not sublats:
        raise ValueError("Each --mix item must include at least one 'subl=...' block.")
    if K is None or K <= 0:
        raise ValueError("Each --mix item must include K>=1.")
    if seed is None:
        raise ValueError("Each --mix item must include 'seed=...'. No default is applied.")

    # Our public MixtureElement type across the codebase is dict-like:
    # {"sublattices": list[SublatticeSpec], "K": int, "seed": int}
    return {"sublattices": sublats, "K": int(K), "seed": int(seed)}
