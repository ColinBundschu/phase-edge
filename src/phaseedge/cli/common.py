from typing import Any
from phaseedge.schemas.mixture import Mixture, canonical_counts

__all__ = ["parse_counts_arg", "parse_cutoffs_arg", "parse_mix_item"]


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


def parse_counts_arg(s: str) -> dict[str, int]:
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


def _split_on_commas_outside_braces(s: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            if depth == 0:
                raise ValueError("Unmatched '}' in composition_map")
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        raise ValueError("Unmatched '{' in composition_map")
    last = "".join(buf).strip()
    if last:
        parts.append(last)
    return parts


def _parse_composition_map_arg(v: str) -> dict[str, dict[str, int]]:
    comp: dict[str, dict[str, int]] = {}
    tokens = _split_on_commas_outside_braces(v.strip())
    if not tokens:
        raise ValueError("composition_map must not be empty")

    for tok in tokens:
        if ":" not in tok:
            raise ValueError(f"Bad composition_map token '{tok}' (expected '<key>:{{...}}')")
        outer_key, inner = tok.split(":", 1)
        outer_key = outer_key.strip()
        inner = inner.strip()
        if not outer_key:
            raise ValueError(f"Empty outer key in composition_map token '{tok}'")
        if not (inner.startswith("{") and inner.endswith("}")):
            raise ValueError(
                f"composition_map entry for '{outer_key}' must be of the form {outer_key}:{{...}}"
            )
        inner_body = inner[1:-1].strip()
        counts = parse_counts_arg(inner_body)
        if outer_key in comp:
            raise ValueError(f"Duplicate outer key '{outer_key}' in composition_map")
        comp[str(outer_key)] = canonical_counts(counts)

    return comp


def parse_mix_item(s: str) -> Mixture:
    """
    Parse one --mix item like:
      "composition_map=Mg:{Fe:54,Mn:46},Co:{Fe:46,Mn:54};K=50;seed=123"

    Returns a Mixture.
    """
    item: dict[str, Any] = {}
    parts = [p.strip() for p in s.split(";") if p.strip()]

    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad --mix token '{p}' (expected key=value)")
        k_raw, v_raw = p.split("=", 1)
        k = k_raw.strip().lower()
        v = v_raw.strip()

        if k == "composition_map":
            item["composition_map"] = _parse_composition_map_arg(v)
        elif k == "k":
            item["K"] = int(v)
        elif k == "seed":
            item["seed"] = int(v)
        else:
            raise ValueError(f"Unknown key '{k}' in --mix item")

    if "composition_map" not in item or "K" not in item:
        raise ValueError("Each --mix item must include composition_map=... and K=...")

    return Mixture(**item)
