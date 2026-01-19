import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.random import default_rng, Generator
from pymatgen.core import Structure
from numpy.typing import NDArray

from phaseedge.schemas.calc_spec import CalcSpec
from phaseedge.schemas.mixture import Mixture, canonical_comp_map, sorted_composition_maps
from phaseedge.science.prototype_spec import PrototypeSpec


def rng_for_index(set_id: str, index: int) -> Generator:
    h = hashlib.sha256(f"{set_id}:{index}".encode()).digest()
    return default_rng(int.from_bytes(h[:8], "big", signed=False))


def _quantize(a: NDArray[np.floating], scale: int) -> NDArray[np.int64]:
    """Round to nearest integer at the given scale (e.g., scale=10**10) and cast to int64."""
    # rint does bankers rounding; that’s fine here and deterministic.
    return np.rint(a * scale).astype(np.int64, copy=False)


# ---- canonical payload builders ----

def _canonical_payload_for_structure(s: Structure, *, ndigits: int = 10) -> dict[str, Any]:
    """
    Build a library-agnostic, order-invariant payload for hashing:
      - lattice: quantized 3x3 in row-major
      - sites  : sorted list of (species_str, fx_q, fy_q, fz_q), with frac coords quantized
    """
    scale = 10 ** int(ndigits)

    # Lattice 3x3
    L = np.asarray(s.lattice.matrix, dtype=float)
    L_q = _quantize(L, scale).reshape(3, 3)

    # Fractional coords: quantize first, THEN wrap by modulo on integers
    F_raw = np.asarray(s.frac_coords, dtype=float)
    F_q = np.rint(F_raw * scale).astype(np.int64, copy=False)
    F_q = np.mod(F_q, scale)  # map to [0, scale) exactly in integer space

    # Species labels as strings (keeps placeholders like 'X', and includes ox states if present)
    # The 'split/join' canonicalizes whitespace for any disordered string forms.
    species_labels = [" ".join(str(site.species).split()) for site in s.sites]

    # Stable sorting by (species_str, fx_q, fy_q, fz_q)
    keys = list(zip(species_labels, F_q[:, 0].tolist(), F_q[:, 1].tolist(), F_q[:, 2].tolist()))
    keys.sort()

    return {
        "L": L_q.flatten().tolist(),   # 9 ints
        "S": keys,                     # [(species_str, fx_q, fy_q, fz_q), ...]
        "nd": int(ndigits),            # bake precision into identity
    }


def occ_key_for_structure(pmg: Structure, *, ndigits: int = 10) -> str:
    payload = _canonical_payload_for_structure(pmg, ndigits=ndigits)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

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

def _normalize_dopt_sampling_intent(src: Mapping[str, Any]) -> dict[str, Any]:
    """
    Canonicalize a refined-WL *intent* source and REQUIRE a seed:

      {
        "type": "dopt_sampling_intent",
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
        raise ValueError("dopt_sampling_intent.wl_policy must include an integer 'seed'.")
    # Normalize/round numerics but keep exact int for seed after casting
    wl_policy = _json_canon(_canon_num(raw_wl_policy))
    wl_policy["seed"] = int(raw_wl_policy["seed"])

    ensure = _json_canon(_canon_num(dict(src.get("ensure", {}))))
    refine = _json_canon(_canon_num(dict(src.get("refine", {}))))
    dopt = _json_canon(_canon_num(dict(src.get("dopt", {}))))
    versions = _json_canon(_canon_num(dict(src.get("versions", {}))))

    return {
        "type": "dopt_sampling_intent",
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
      - {"type": "dopt_sampling_intent", ...}

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

        elif t == "dopt_sampling_intent":
            norm.append(_normalize_dopt_sampling_intent(src))

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
    prototype_spec: PrototypeSpec,
    supercell_diag: tuple[int, int, int],
    sources: Sequence[Mapping[str, Any]],
    algo_version: str,
    calc_spec: CalcSpec,
    basis_spec: Mapping[str, Any],
    partial: bool,
    regularization: Mapping[str, Any] | None = None,
    weighting: Mapping[str, Any] | None = None,
) -> str:
    """
    Deterministic key for a CE trained from an arbitrary set of sampling sources.
    Identity includes: system, canonicalized `sources`, engine, hyperparams (incl. weighting), and algo_version.
    """
    norm_sources = normalize_sources(sources)

    payload = {
        "kind": "partial_ce_key@sources" if partial else "ce_key@sources",
        "system": {
            "prototype_spec": _json_canon(prototype_spec.as_dict()),
            "supercell": list(supercell_diag),
        },
        "sampling": {
            "algo": algo_version,
            "sources": norm_sources,
        },
        "calc_spec": _json_canon(calc_spec.as_dict()),
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
    initial_comp_map: Mapping[str, Mapping[str, int]],
    reject_cross_sublattice_swaps: bool,
    check_period: int,
    update_period: int,
    seed: int,
    algo_version: str,
) -> str:
    # (unchanged)
    payload = {
        "kind": "wl_key",
        "algo_version": algo_version,
        "ce_key": str(ce_key),
        "ensemble": {
            "type": "canonical",
            "init_comp_map": canonical_comp_map(initial_comp_map),
        },
        "grid": {
            "bin_width": _round_float(float(bin_width)),
        },
        "mc": {
            "reject_cross_sublattice_swaps": bool(reject_cross_sublattice_swaps),
            "step_type": str(step_type),
            "check_period": int(check_period),
            "update_period": int(update_period),
            "seed": int(seed),
        },
    }
    blob = json.dumps(_json_canon(_canon_num(payload)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_wl_block_key(
    *,
    wl_key: str,
    parent_wl_block_key: str,
    state: Mapping[str, Any],
    occupancy: np.ndarray,
) -> str:
    """
    Compute a canonical WL chunk key used for immutable block identity.

    NOTE: Keep this stable. It must remain a function ONLY of
          (wl_key, parent_wl_block_key, state, occupancy).
    """
    payload = {
        "wl_key": wl_key,
        "parent_wl_block_key": parent_wl_block_key,
        "state": _json_canon(state),
        "occupancy": occupancy.astype(int).tolist(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
