import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.random import default_rng, Generator
from ase.atoms import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.schemas.mixture import Mixture, canonical_counts, sorted_composition_maps
from phaseedge.science.prototypes import PrototypeName


def compute_set_id(
    *,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any] | None,
    supercell_diag: tuple[int, int, int],
    composition_map: dict[str, dict[str, int]],
    seed: int,
    algo_version: str = "randgen-3-compmap-1",
) -> str:
    """
    Deterministic identity for a logical (expandable) snapshot sequence using a
    multi-sublattice composition map.

    Parameters
    ----------
    prototype
        Prototype name (e.g., "rocksalt").
    prototype_params
        Prototype parameters (e.g., {"a": 4.3}).
    supercell_diag
        Diagonal replication of the conventional/primitive cell (e.g., (3,3,3)).
    composition_map
        Map of each replaceable sublattice label (e.g., "Mg") to its integer
        counts mapping, e.g. {"Mg": {"Fe": 10, "Mg": 98}, "Al": {...}, ...}.
        Keys and nested keys are canonically sorted before hashing.
    seed
        RNG seed for the sequence of K snapshots under this logical set.
    algo_version
        Version tag for the identity algorithm.

    Returns
    -------
    str
        Stable SHA-256 hex digest identifying this logical snapshot set.
    """
    # Canonicalize: sort sublattice labels and, within each, sort element labels
    comp_norm: dict[str, dict[str, int]] = {}
    for sublat in sorted(composition_map):
        counts = composition_map[sublat] or {}
        comp_norm[sublat] = {el: int(counts[el]) for el in sorted(counts)}

    payload = {
        "prototype": prototype,
        "proto_params": dict(prototype_params) if prototype_params else {},
        "diag": list(map(int, supercell_diag)),
        "composition_map": comp_norm,
        "seed": int(seed),
        "algo": str(algo_version),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def seed_for(set_id: str, index: int) -> int:
    h = hashlib.sha256(f"{set_id}:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def rng_for_index(set_id: str, index: int) -> Generator:
    return default_rng(seed_for(set_id, index))


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


# -------------------- refined-WL INTENT normalization --------------------

def _normalize_wl_refined_intent(src: Mapping[str, Any]) -> dict[str, Any]:
    """
    Canonicalize a refined-WL *intent* source and REQUIRE a seed:

      {
        "type": "wl_refined_intent",
        "base_ce_key": str,
        "endpoints": [composition_map, ...],
        "wl_policy": {bin_width, step_type, check_period, update_period, seed},  # seed REQUIRED
        "ensure": {steps_to_run, samples_per_bin},
        "refine": {mode, n_total|null, per_bin_cap|null, strategy},
        "dopt": {budget, ridge, tie_breaker},
        "versions": {refine, dopt, sampler}
      }
    """
    base_ce_key = str(src["base_ce_key"])

    # wl_policy: must include a seed
    raw_wl_policy = dict(src.get("wl_policy", {}))
    if "seed" not in raw_wl_policy:
        raise ValueError("wl_refined_intent.wl_policy must include an integer 'seed'.")
    # Normalize/round numerics but keep exact int for seed after casting
    wl_policy = _json_canon(_canon_num(raw_wl_policy))
    wl_policy["seed"] = int(raw_wl_policy["seed"])

    ensure = _json_canon(_canon_num(dict(src.get("ensure", {}))))
    refine = _json_canon(_canon_num(dict(src.get("refine", {}))))
    dopt = _json_canon(_canon_num(dict(src.get("dopt", {}))))
    versions = _json_canon(_canon_num(dict(src.get("versions", {}))))

    return {
        "type": "wl_refined_intent",
        "base_ce_key": base_ce_key,
        "endpoints": sorted_composition_maps(src["endpoints"]),
        "wl_policy": wl_policy,
        "ensure": ensure,
        "refine": refine,
        "dopt": dopt,
        "versions": versions,
    }


# -------------------- unified CE key over arbitrary sources (intent only) --------------------

def normalize_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize the 'sources' list into a canonical JSON-ready structure.

    Supported source types:
      - {"type": "composition", "mixtures": list[Mixture]}
      - {"type": "wl_refined_intent", ...}

    BREAKING CHANGE: 'elements' replaced by 'mixtures'.
    """
    norm: list[dict[str, Any]] = []

    for src in sources:
        t = str(src.get("type", "")).lower()

        if t == "composition":
            mixtures_in = src.get("mixtures")
            if mixtures_in is None:
                raise ValueError("composition.mixtures is required")

            mixtures: list[Mixture] = list(mixtures_in)
            if not all(isinstance(m, Mixture) for m in mixtures):
                raise TypeError("composition.mixtures must be a list of Mixture objects")

            mixtures_sorted = sorted(mixtures, key=lambda m: m.sort_key())

            # Emit a plain, MSON-friendly payload (no @module/@class noise)
            mixtures_payload: list[dict[str, Any]] = [
                {"composition_map": m.composition_map, "K": m.K, "seed": m.seed}
                for m in mixtures_sorted
            ]

            norm.append({"type": "composition", "mixtures": mixtures_payload})

        elif t == "wl_refined_intent":
            norm.append(_normalize_wl_refined_intent(src))

        else:
            raise ValueError(f"Unknown source type: {t!r}")

    # sort sources deterministically by type then JSON payload
    def _src_key(s: Mapping[str, Any]) -> tuple[str, str]:
        tt = str(s.get("type", ""))
        j = json.dumps(_json_canon(s), sort_keys=True, separators=(",", ":"))
        return (tt, j)

    return sorted(norm, key=_src_key)


def compute_ce_key(
    *,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    sources: Sequence[Mapping[str, Any]],
    algo_version: str,
    model: str,
    relax_cell: bool,
    dtype: str,
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any] | None = None,
    weighting: Mapping[str, Any] | None = None,
) -> str:
    """
    Deterministic key for a CE trained from an arbitrary set of sampling sources.
    Identity includes: system, canonicalized `sources`, engine, hyperparams (incl. weighting), and algo_version.
    """
    norm_sources = normalize_sources(sources)

    payload = {
        "kind": "ce_key@sources",
        "system": {
            "prototype": prototype,
            "proto_params": _json_canon(prototype_params),
            "supercell": list(supercell_diag),
        },
        "sampling": {
            "algo": algo_version,
            "sources": norm_sources,
        },
        "engine": {"model": model, "relax_cell": bool(relax_cell), "dtype": dtype},
        "hyperparams": {
            "basis": _json_canon(basis_spec),
            "regularization": _json_canon(regularization or {}),
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
