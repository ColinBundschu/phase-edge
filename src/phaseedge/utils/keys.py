import hashlib
import json
from typing import Any, Mapping, Sequence, cast
import numpy as np
from numpy.random import default_rng, Generator
from ase.atoms import Atoms
from pymatgen.io.ase import AseAtomsAdaptor


def compute_set_id_counts(
    *,
    prototype: str,
    prototype_params: dict[str, Any] | None,
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    counts: dict[str, int],
    seed: int,
    algo_version: str = "randgen-2-counts-1",
) -> str:
    """Deterministic identity for a logical (expandable) snapshot sequence using integer counts."""
    counts_sorted = {k: int(counts[k]) for k in sorted(counts)}
    payload = {
        "prototype": prototype,
        "proto_params": prototype_params or {},
        "diag": supercell_diag,
        "replace": replace_element,
        "counts": counts_sorted,
        "seed": seed,
        "algo": algo_version,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def seed_for(set_id: str, index: int, attempt: int = 0) -> int:
    h = hashlib.sha256(f"{set_id}:{index}:{attempt}".encode()).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def rng_for_index(set_id: str, index: int, attempt: int = 0) -> Generator:
    return default_rng(seed_for(set_id, index, attempt))


def occ_key_for_atoms(snapshot: Atoms) -> str:
    pmg = AseAtomsAdaptor.get_structure(snapshot)  # type: ignore[arg-type]
    payload = {
        "lattice": np.asarray(pmg.lattice.matrix).round(10).tolist(),
        "frac": np.asarray(pmg.frac_coords).round(10).tolist(),
        "species": [str(sp) for sp in pmg.species],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# -------------------- canonicalization helpers --------------------

def _json_canon(obj: Any) -> Any:
    try:
        import numpy as _np
    except Exception:  # pragma: no cover
        _np = None
    if isinstance(obj, dict):
        return {k: _json_canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_json_canon(x) for x in obj]
    if _np is not None and isinstance(obj, _np.ndarray):
        return _json_canon(obj.tolist())
    return obj


def canonical_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    """CE-style canonicalization: sort keys and cast values to int (no zero/neg checks)."""
    return {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])}


def _round_float(x: float, ndigits: int = 12) -> float:
    return float(f"{x:.{ndigits}g}")


def _canon_num(v: Any, ndigits: int = 12) -> Any:
    if isinstance(v, float):
        return _round_float(v, ndigits)
    if isinstance(v, (list, tuple)):
        return [_canon_num(x, ndigits) for x in v]
    if isinstance(v, dict):
        return {str(k): _canon_num(v[k], ndigits) for k in sorted(v)}
    return v


# -------------------- unified CE key over arbitrary sources --------------------

def _normalize_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize the 'sources' list into a canonical JSON-ready structure.

    Supported source types:
      - {"type": "composition", "elements": [{"counts": {...}, "K": int, "seed": int}, ...]}
      - {"type": "wl_chunked", "chunks": [{"wl_key": str, "step_end": int}, ...],
         "selection": {"policy": str, "k_per_bin": int | None, "dedupe": str | None}}
      - {"type": "specified", "occ_keys": [str, ...]}
    """
    norm: list[dict[str, Any]] = []

    for src in sources:
        t = str(src.get("type", "")).lower()

        if t == "composition":
            elems_in = list(src.get("elements", []))
            elems_norm: list[dict[str, Any]] = []
            for e in elems_in:
                cnts = canonical_counts(e.get("counts", {}))
                K = int(e.get("K", 0))
                seed = int(e.get("seed", 0))
                elems_norm.append({"counts": cnts, "K": K, "seed": seed})

            # stable sort: by (counts_json, seed, K)
            def _ekey(e: Mapping[str, Any]) -> tuple[str, int, int]:
                cj = json.dumps(e["counts"], sort_keys=True, separators=(",", ":"))
                return (cj, int(e["seed"]), int(e["K"]))
            elems_norm = sorted(elems_norm, key=_ekey)

            norm.append({"type": "composition", "elements": elems_norm})

        elif t == "wl_chunked":
            chunks_in = list(src.get("chunks", []))
            chunks_norm: list[dict[str, Any]] = []
            for ch in chunks_in:
                wl_key = str(ch["wl_key"])
                step_end = int(ch["step_end"])
                chunks_norm.append({"wl_key": wl_key, "step_end": step_end})
            # sort deterministically
            chunks_norm = sorted(chunks_norm, key=lambda c: (c["wl_key"], c["step_end"]))

            sel_in = dict(src.get("selection", {}))
            selection = {
                "policy": str(sel_in.get("policy", "")),
                **({"k_per_bin": int(sel_in["k_per_bin"])} if "k_per_bin" in sel_in and sel_in["k_per_bin"] is not None else {}),
                **({"dedupe": str(sel_in["dedupe"])} if "dedupe" in sel_in and sel_in["dedupe"] is not None else {}),
            }

            norm.append({"type": "wl_chunked", "chunks": chunks_norm, "selection": selection})

        elif t == "specified":
            occ_keys_in = [str(x) for x in src.get("occ_keys", [])]
            occ_keys_norm = sorted(set(occ_keys_in))
            norm.append({"type": "specified", "occ_keys": occ_keys_norm})

        else:
            raise ValueError(f"Unknown source type: {t!r}")

    # sort the sources list deterministically:
    #  - primary by type
    #  - then by full JSON of the inner payload
    def _src_key(s: Mapping[str, Any]) -> tuple[str, str]:
        t = str(s.get("type", ""))
        j = json.dumps(_json_canon(s), sort_keys=True, separators=(",", ":"))
        return (t, j)

    return sorted(norm, key=_src_key)


def compute_ce_key(
    *,
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    sources: Sequence[Mapping[str, Any]],
    algo_version: str,
    model: str,
    relax_cell: bool,
    dtype: str,
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any] | None = None,
    extra_hyperparams: Mapping[str, Any] | None = None,
    weighting: Mapping[str, Any] | None = None,
) -> str:
    """
    Deterministic key for a CE trained from an arbitrary set of sampling sources.
    Identity includes: system, canonicalized `sources`, engine, hyperparams (incl. weighting), and algo_version.
    """
    norm_sources = _normalize_sources(sources)

    payload = {
        "kind": "ce_key@sources",
        "system": {
            "prototype": prototype,
            "proto_params": _json_canon(prototype_params),
            "supercell": list(supercell_diag),
            "replace": replace_element,
        },
        "sampling": {
            "algo": algo_version,
            "sources": norm_sources,
        },
        "engine": {"model": model, "relax_cell": bool(relax_cell), "dtype": dtype},
        "hyperparams": {
            "basis": _json_canon(basis_spec),
            "regularization": _json_canon(regularization or {}),
            "extra": _json_canon(extra_hyperparams or {}),
            "weighting": _json_canon(weighting or {}),
        },
    }

    blob = json.dumps(_json_canon(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -------------------- Wang-Landau chain key (unchanged semantics) --------------------

def compute_wl_key(
    *,
    ce_key: str,
    bin_width: float,
    step_type: str,
    composition_counts: Mapping[str, int],
    check_period: int,
    update_period: int,
    seed: int,
    grid_anchor: float = 0.0,
    algo_version: str = "wl-grid-v1",
) -> str:
    """
    Idempotent identity for a canonical WL run based ONLY on the public contract:
    CE identity, *exact counts* (canonicalized like CE), binning contract, MC schedule (sans steps), and seed.
    Zero-count species are stripped; at least one positive count is required.
    """
    comp_counts = canonical_counts(composition_counts)
    comp_counts = {k: int(v) for k, v in comp_counts.items() if int(v) != 0}
    if not comp_counts:
        raise ValueError("composition_counts must include at least one species with a positive count.")

    payload = {
        "kind": "wl_key",
        "algo": algo_version,
        "ce_key": str(ce_key),
        "ensemble": {
            "type": "canonical",
            "composition_counts": comp_counts,
        },
        "grid": {
            "anchor": _round_float(grid_anchor),
            "bin_width": _round_float(float(bin_width)),
        },
        "mc": {
            "step_type": str(step_type),
            "check_period": int(check_period),
            "update_period": int(update_period),
            "seed": int(seed),
        },
    }
    blob = json.dumps(_json_canon(_canon_num(payload)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
