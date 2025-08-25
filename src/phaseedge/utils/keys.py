import hashlib
import json
from typing import Any, Mapping, Sequence
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

# ---- CE key (counts-based, deterministic) ------------------------------------------

def _json_canon(obj: Any) -> Any:
    """
    Canonicalize common Python containers for stable hashing:
    - dicts: sort keys recursively
    - tuples: convert to lists
    - numpy arrays: convert to (nested) lists if they sneak in
    - other objects: leave as-is (assuming they are JSON-serializable or simple scalars)
    """
    try:
        import numpy as _np  # local import to avoid hard dependency at import time
    except Exception:  # pragma: no cover
        _np = None

    if isinstance(obj, dict):
        return {k: _json_canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [ _json_canon(x) for x in obj ]
    if _np is not None and isinstance(obj, _np.ndarray):
        return _json_canon(obj.tolist())
    return obj

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
    Deterministic key for a CE trained on an EXACT set of snapshots.

    Identity includes:
      - system (prototype+params) + supercell + replace rule
      - exact integer counts, seed, algo_version, and the exact list of indices
      - relax engine (model/relax_cell/dtype)
      - CE hyperparameters (basis_spec, regularization, and any extra knobs)

    Returns a SHA256 hex digest over a canonically-ordered JSON payload.

    Notes:
      * counts are sorted by species key to avoid dict-order nondeterminism.
      * indices are sorted to avoid order sensitivity (membership defines identity).
      * all mappings are recursively key-sorted via _json_canon.
      * NO normalized ratios—composition identity is exact integer counts.
    """
    # Stable, integer-only counts (sorted by species)
    counts_sorted: dict[str, int] = {k: int(counts[k]) for k in sorted(counts)}

    # Stable indices list (membership only; order-insensitive)
    indices_sorted: list[int] = sorted(int(i) for i in indices)

    payload = {
        "kind": "ce_key@counts",  # explicit tag for future-proofing
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

    import json, hashlib
    blob = json.dumps(_json_canon(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
