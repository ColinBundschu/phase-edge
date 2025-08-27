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
    """
    Deterministic identity for a logical (expandable) snapshot sequence using integer counts.
    """
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

def compute_ce_key(
    *,
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    counts: Mapping[str, int],
    seed: int,
    indices: Sequence[int],
    algo_version: str = "randgen-2-counts-1",
    model: str,
    relax_cell: bool,
    dtype: str,
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any] | None = None,
    extra_hyperparams: Mapping[str, Any] | None = None,
) -> str:
    counts_sorted: dict[str, int] = {k: int(counts[k]) for k in sorted(counts)}
    indices_sorted: list[int] = sorted(int(i) for i in indices)

    payload = {
        "kind": "ce_key@counts",
        "system": {
            "prototype": prototype,
            "proto_params": _json_canon(prototype_params),
            "supercell": list(supercell_diag),
            "replace": replace_element,
        },
        "sampling": {
            "counts": counts_sorted,
            "seed": int(seed),
            "algo": algo_version,
            "indices": indices_sorted,
        },
        "engine": {"model": model, "relax_cell": bool(relax_cell), "dtype": dtype},
        "hyperparams": {
            "basis": _json_canon(basis_spec),
            "regularization": _json_canon(regularization or {}),
            "extra": _json_canon(extra_hyperparams or {}),
        },
    }

    blob = json.dumps(_json_canon(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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

# --- Wang–Landau keys -------------------------------------------------------------

from typing import Optional

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

def canonical_float_map(m: Optional[Mapping[str, Any]], ndigits: int = 12) -> dict[str, float]:
    if not m:
        return {}
    out: dict[str, float] = {}
    for k in sorted(m):
        v = m[k]
        if isinstance(v, (int, float)):
            out[str(k)] = _round_float(float(v), ndigits)
        else:
            raise TypeError(f"Non-numeric value for key '{k}' in float map: {v!r}")
    return out

def compute_wl_key(
    *,
    ce_key: str,
    bin_width: float,
    steps: int,
    step_type: str,
    ensemble: str,  # "canonical" | "semi_grand"
    composition: Optional[Mapping[str, float]] = None,          # canonical
    chemical_potentials: Optional[Mapping[str, float]] = None,  # semi_grand
    check_period: int,
    update_period: int,
    seed: int,
    grid_anchor: float = 0.0,
    algo_version: str = "wl-grid-v1",   # bump if the PUBLIC CONTRACT changes
) -> str:
    """
    Idempotent identity for a WL run based ONLY on the public contract:
    CE identity, ensemble spec, binning contract, MC schedule, and seed.
    NO derived window, NO pilot params, NO internal hacks included.
    """
    if ensemble not in ("canonical", "semi_grand"):
        raise ValueError(f"Invalid ensemble: {ensemble!r}")

    comp_canon = canonical_float_map(composition) if composition else {}
    mu_canon   = canonical_float_map(chemical_potentials) if chemical_potentials else {}

    payload = {
        "kind": "wl_key",
        "algo": algo_version,
        "ce_key": str(ce_key),
        "ensemble": {
            "type": ensemble,
            "composition": comp_canon,           # empty if not used
            "chemical_potentials": mu_canon,     # empty if not used
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
