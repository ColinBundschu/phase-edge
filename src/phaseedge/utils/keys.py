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


# ---- Canonicalization helpers --------------------------------------------------------

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


def _canon_indices(elem: Mapping[str, Any]) -> list[int]:
    if "indices" in elem and elem["indices"] is not None:
        idx = [int(x) for x in cast(Sequence[Any], elem["indices"])]
        return sorted(dict.fromkeys(idx))
    K = int(elem.get("K", 0))
    if K <= 0:
        raise ValueError(f"Mixture element missing valid K/indices: {elem!r}")
    return [int(i) for i in range(K)]


def canonical_mixture_signature(mix: Sequence[Mapping[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for elem in mix:
        cnts = canonical_counts(elem.get("counts", {}))
        seed = int(elem.get("seed", 0))
        indices = _canon_indices(elem)
        normalized.append({"counts": cnts, "seed": seed, "indices": indices})

    def _key(e: Mapping[str, Any]) -> tuple[str, int, str]:
        c = json.dumps(e["counts"], sort_keys=True, separators=(",", ":"))
        i = json.dumps(e["indices"], sort_keys=True, separators=(",", ":"))
        return (c, int(e["seed"]), i)

    normalized_sorted = sorted(normalized, key=_key)
    return json.dumps(normalized_sorted, sort_keys=True, separators=(",", ":"))


# ---- CE key (counts-based, deterministic) -------------------------------------------

def compute_ce_key_mixture(
    *,
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    mixture: Sequence[Mapping[str, Any]],
    algo_version: str = "randgen-3-mix-1",
    model: str,
    relax_cell: bool,
    dtype: str,
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any] | None = None,
    extra_hyperparams: Mapping[str, Any] | None = None,
    weighting: Mapping[str, Any] | None = None,
) -> str:
    """
    Deterministic key for a CE trained on a union of snapshot families (mixture of compositions).
    Identity includes system, mixture signature, engine, hyperparams (including weighting), and algo_version.
    """
    mix_sig = canonical_mixture_signature(mixture)

    payload = {
        "kind": "ce_key@mixture",
        "system": {
            "prototype": prototype,
            "proto_params": _json_canon(prototype_params),
            "supercell": list(supercell_diag),
            "replace": replace_element,
        },
        "sampling": {"mixture_sig": mix_sig, "algo": algo_version},
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


# --- Wang-Landau keys (COUNTS-ONLY, canonical) ---------------------------------------

def _round_float(x: float, ndigits: int = 12) -> float:
    # 12 significant digits is a good balance: stable, but not overly lossy.
    return float(f"{x:.{ndigits}g}")


def _canon_num(v: Any, ndigits: int = 12) -> Any:
    if isinstance(v, float):
        return _round_float(v, ndigits)
    if isinstance(v, (list, tuple)):
        return [_canon_num(x, ndigits) for x in v]
    if isinstance(v, dict):
        return {str(k): _canon_num(v[k], ndigits) for k in sorted(v)}
    return v


def compute_wl_key(
    *,
    ce_key: str,
    bin_width: float,
    steps: int,
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
    CE identity, *exact counts* (canonicalized like CE), binning contract, MC schedule, and seed.
    Zero-count species are stripped; at least one positive count is required.
    """
    # Canonicalize counts identically to CE, then apply zero guardrail
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
            "steps": int(steps),
            "check_period": int(check_period),
            "update_period": int(update_period),
            "seed": int(seed),
        },
    }
    blob = json.dumps(_json_canon(_canon_num(payload)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
