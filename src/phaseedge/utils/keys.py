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
    # Stable, sorted counts to avoid dict-order nondeterminism
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
    """
    Canonicalize common Python containers for stable hashing:
    - dicts: sort keys recursively
    - lists/tuples: convert tuples to lists and recurse
    - numpy arrays: convert to (nested) lists
    - everything else: return as-is
    """
    try:
        import numpy as _np  # local import to avoid hard dependency at import time
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
    """
    Return a dict[str, int] with sorted keys and coerced integer values.
    """
    return {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])}


def _canon_indices(elem: Mapping[str, Any]) -> list[int]:
    """
    From a mixture element, produce a canonical indices list.
    If 'indices' is present, use sorted unique ints.
    Else, require 'K' and return [0..K-1].
    """
    if "indices" in elem and elem["indices"] is not None:
        idx = [int(x) for x in cast(Sequence[Any], elem["indices"])]
        return sorted(dict.fromkeys(idx))  # unique + sorted
    K = int(elem.get("K", 0))
    if K <= 0:
        raise ValueError(f"Mixture element missing valid K/indices: {elem!r}")
    return [int(i) for i in range(K)]


def canonical_mixture_signature(mix: Sequence[Mapping[str, Any]]) -> str:
    """
    Canonical, order-insensitive JSON string describing a mixture.
    Each element contributes: {counts (sorted), seed (int), indices (sorted list)}.
    Elements are sorted by (counts_json, seed, indices_json) to remove order sensitivity.
    """
    normalized: list[dict[str, Any]] = []
    for elem in mix:
        cnts = canonical_counts(elem.get("counts", {}))
        seed = int(elem.get("seed", 0))
        indices = _canon_indices(elem)
        normalized.append({"counts": cnts, "seed": seed, "indices": indices})

    # Sort elements deterministically
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
    # sublattice definition / replacement rule (keep consistent with snapshot generation)
    replace_element: str,
    # --- composition / sampling (EXACT COUNTS; NO RATIOS)
    counts: Mapping[str, int],
    seed: int,
    indices: Sequence[int],            # exact membership, e.g. [0,1,...,K-1]
    algo_version: str = "randgen-2-counts-1",
    # --- relax/engine identity
    model: str,
    relax_cell: bool,
    dtype: str,
    # --- CE hyperparameters (all knobs that distinguish models)
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any] | None = None,
    extra_hyperparams: Mapping[str, Any] | None = None,
) -> str:
    """
    Deterministic key for a CE trained on an EXACT set of snapshots (single composition).
    """
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
        "engine": {
            "model": model,
            "relax_cell": bool(relax_cell),
            "dtype": dtype,
        },
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
    # --- mixture: each element has counts, and either indices[] or K, plus seed
    mixture: Sequence[Mapping[str, Any]],
    algo_version: str = "randgen-3-mix-1",
    # --- relax/engine identity
    model: str,
    relax_cell: bool,
    dtype: str,
    # --- CE hyperparameters (all knobs that distinguish models)
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any] | None = None,
    extra_hyperparams: Mapping[str, Any] | None = None,
) -> str:
    """
    Deterministic key for a CE trained on a union of snapshot families (mixture of compositions).

    Identity includes:
      - system (prototype+params) + supercell + replace rule
      - mixture signature (order-insensitive):
          list of {counts (sorted), seed (int), indices (sorted list)}
      - relax engine (model/relax_cell/dtype)
      - CE hyperparameters (basis_spec, regularization, extra)
      - algo_version for the sampler family
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
        "sampling": {
            "mixture_sig": mix_sig,
            "algo": algo_version,
        },
        "engine": {
            "model": model,
            "relax_cell": bool(relax_cell),
            "dtype": dtype,
        },
        "hyperparams": {
            "basis": _json_canon(basis_spec),
            "regularization": _json_canon(regularization or {}),
            "extra": _json_canon(extra_hyperparams or {}),
        },
    }

    blob = json.dumps(_json_canon(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
