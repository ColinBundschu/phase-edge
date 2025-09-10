# phaseedge/utils/keys.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.random import default_rng, Generator
from ase.atoms import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.schemas.sublattice import SublatticeSpec


# -------------------- Public dataclasses (universal CE key spec) --------------------

@dataclass(frozen=True, slots=True)
class SublatticeMixtureElement:
    """One mixture element: exact integer counts on one or more sublattices, plus K and seed."""
    sublattices: Sequence[SublatticeSpec]
    K: int
    seed: int


@dataclass(frozen=True, slots=True)
class CEKeySpec:
    """
    Universal, strongly-typed identity for a CE training run over sublattice compositions.
    This is the ONLY input type accepted by compute_ce_key.
    """
    # System
    prototype: str
    prototype_params: Mapping[str, Any]
    supercell_diag: tuple[int, int, int]

    # Sampling (only sublattice compositions are supported for CE identity)
    mixtures: Sequence[SublatticeMixtureElement]   # each has sublattices + K + seed

    # Engine identity
    model: str
    relax_cell: bool
    dtype: str

    # Hyperparameters
    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any] | None = None
    weighting: Mapping[str, Any] | None = None

    # Version tag for the sampling/key contract
    algo_version: str = "randgen-4-sublcomp-1"


# -------------------- Public helpers (unchanged) --------------------

def compute_set_id_counts(
    *,
    prototype: str,
    prototype_params: dict[str, Any] | None,
    supercell_diag: tuple[int, int, int],
    sublattices: Sequence[SublatticeSpec],
    seed: int,
    algo_version: str = "randgen-3-subl-1",
) -> str:
    """Deterministic identity for an RNG snapshot sequence (counts on one/more sublattices)."""
    canon_subl = _canon_sublattice_specs(sublattices)
    payload = {
        "prototype": str(prototype),
        "proto_params": prototype_params or {},
        "diag": list(supercell_diag),
        "subl": canon_subl,
        "seed": int(seed),
        "algo": str(algo_version),
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
    """Sort keys and cast to int (no zero/neg checks)."""
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


def _canon_sublattice_specs(sublattices: Sequence[SublatticeSpec]) -> list[dict[str, Any]]:
    """
    Canonicalize SublatticeSpec list into JSON-ready dicts:
      [{"replace": <str>, "counts": {elem: int, ...}}, ...]
    Deterministic order: sort by 'replace', then by counts JSON.
    """
    canon: list[dict[str, Any]] = []
    for sl in sublattices:
        replace = str(sl.replace_element)
        cnts = canonical_counts(sl.counts)
        canon.append({"replace": replace, "counts": cnts})

    def _key(d: Mapping[str, Any]) -> tuple[str, str]:
        cj = json.dumps(d["counts"], sort_keys=True, separators=(",", ":"))
        return (d["replace"], cj)

    return sorted(canon, key=_key)


def _canon_mixtures(mixtures: Sequence[SublatticeMixtureElement]) -> list[dict[str, Any]]:
    """
    Canonicalize mixture elements:
      [{"sublattices": [{"replace":..., "counts": {...}}, ...], "K": int, "seed": int}, ...]
    Stable order: by sublattices JSON, then seed, then K.
    """
    out: list[dict[str, Any]] = []
    for m in mixtures:
        subls = _canon_sublattice_specs(m.sublattices)
        out.append({"sublattices": subls, "K": int(m.K), "seed": int(m.seed)})

    def _ekey(e_: Mapping[str, Any]) -> tuple[str, int, int]:
        cj = json.dumps(e_["sublattices"], sort_keys=True, separators=(",", ":"))
        return (cj, int(e_["seed"]), int(e_["K"]))

    return sorted(out, key=_ekey)


def _derive_replace_elements(mixtures: Sequence[SublatticeMixtureElement]) -> list[str]:
    """Collect and sort the unique placeholder symbols appearing across all sublattices."""
    seen: set[str] = set()
    for m in mixtures:
        for sl in m.sublattices:
            seen.add(str(sl.replace_element))
    return sorted(seen)


# -------------------- Universal CE key --------------------

def compute_ce_key(*, spec: CEKeySpec) -> str:
    """
    Deterministic key for a CE trained from sublattice-composition mixtures.
    THIS is the sole public interface. All inputs must be provided via `CEKeySpec`.

    Identity includes:
      - system (prototype, params, supercell, derived replace_elements),
      - sampling (algo_version, canonical mixtures),
      - engine (model, relax_cell, dtype),
      - hyperparameters (basis_spec, regularization, weighting).
    """
    mixtures_canon = _canon_mixtures(spec.mixtures)
    replace_elements = _derive_replace_elements(spec.mixtures)

    payload = {
        "kind": "ce_key@sublattice_composition",
        "system": {
            "prototype": str(spec.prototype),
            "proto_params": _json_canon(spec.prototype_params),
            "supercell": [int(x) for x in spec.supercell_diag],
            "replace_elements": replace_elements,
        },
        "sampling": {
            "algo": str(spec.algo_version),
            "elements": mixtures_canon,
        },
        "engine": {
            "model": str(spec.model),
            "relax_cell": bool(spec.relax_cell),
            "dtype": str(spec.dtype),
        },
        "hyperparams": {
            "basis": _json_canon(spec.basis_spec),
            "regularization": _json_canon(spec.regularization or {}),
            "weighting": _json_canon(spec.weighting or {}),
        },
    }
    blob = json.dumps(_json_canon(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# -------------------- WL key (unchanged) --------------------

def compute_wl_key(
    *,
    ce_key: str,
    bin_width: float,
    step_type: str,
    composition_counts: Mapping[str, int],
    check_period: int,
    update_period: int,
    seed: int,
    algo_version: str = "wl-grid-v1",
) -> str:
    comp_counts = canonical_counts(composition_counts)
    comp_counts = {k: int(v) for k, v in comp_counts.items() if int(v) != 0}
    if not comp_counts:
        raise ValueError("composition_counts must include at least one species with a positive count.")
    payload = {
        "kind": "wl_key",
        "algo": algo_version,
        "ce_key": str(ce_key),
        "ensemble": {"type": "canonical", "composition_counts": comp_counts},
        "grid": {"bin_width": _round_float(float(bin_width))},
        "mc": {
            "step_type": str(step_type),
            "check_period": int(check_period),
            "update_period": int(update_period),
            "seed": int(seed),
        },
    }
    blob = json.dumps(_json_canon(_canon_num(payload)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
