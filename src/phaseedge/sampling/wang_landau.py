from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple, cast
import numpy as np

from pymatgen.io.ase import AseAtomsAdaptor
from smol.cofe import ClusterExpansion
from smol.moca.ensemble import Ensemble
from smol.moca import Sampler  # disorder-style API

from phaseedge.schemas.wl import WLSamplerSpec, WLResult
from phaseedge.storage.ce_store import lookup_ce_by_key
from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import (
    make_one_snapshot,
    validate_counts_for_sublattice,
)
from phaseedge.science.ce_training import featurize_structures, predict_from_features
from phaseedge.utils.grid import snap_floor, snap_ceil, ANCHOR


# ----------------------------- helpers ------------------------------------ #

def _rehydrate_ce(ce_key: str) -> Mapping[str, Any]:
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")
    return cast(Mapping[str, Any], doc)


def _pilot_enthalpies(
    doc: Mapping[str, Any],
    *,
    n: int,
    rng: np.random.Generator,
    counts_required: Mapping[str, int],
) -> np.ndarray:
    """
    Sample n random snapshots at the *exact requested counts* and return
    CE-predicted enthalpies (eV/supercell).
    """
    system = cast(Mapping[str, Any], doc["system"])
    payload = cast(Mapping[str, Any], doc["payload"])
    ce = ClusterExpansion.from_dict(dict(payload))
    subspace = ce.cluster_subspace

    prototype = cast(str, system["prototype"])
    prototype_params = cast(Mapping[str, Any], system["prototype_params"])
    supercell_diag = tuple(system["supercell_diag"])
    replace_element = cast(str, system["replace_element"])

    conv = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))

    counts_clean = {str(k): int(v) for k, v in counts_required.items()}
    validate_counts_for_sublattice(
        conv_cell=conv,
        supercell_diag=cast(Tuple[int, int, int], tuple(supercell_diag)),
        replace_element=replace_element,
        counts=counts_clean,
    )

    structures = []
    for _ in range(n):
        snap = make_one_snapshot(
            conv_cell=conv,
            supercell_diag=cast(Tuple[int, int, int], tuple(supercell_diag)),
            replace_element=replace_element,
            counts=counts_clean,
            rng=rng,
        )
        structures.append(AseAtomsAdaptor.get_structure(snap))  # type: ignore[arg-type]

    _, X = featurize_structures(
        subspace=subspace,
        structures=structures,
        supercell_diag=cast(Tuple[int, int, int], tuple(supercell_diag)),
    )
    coefs = np.asarray(getattr(ce, "parameters", getattr(ce, "coefs")))
    H_raw = predict_from_features(X, coefs)  # typically eV per *primitive cell*

    # Scale to eV per supercell so units match E0 and WL kernel
    sc_matrix = np.diag(cast(Sequence[int], supercell_diag))
    ensemble = Ensemble.from_cluster_expansion(ce, supercell_matrix=sc_matrix)
    H_supercell = np.asarray(H_raw, dtype=float) * ensemble.processor.size
    return H_supercell


def _initial_occupancy_from_counts(
    *,
    doc: Mapping[str, Any],
    counts: Mapping[str, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, Ensemble]:
    """
    Build an initial occupancy consistent with the CE's processor and exact counts.
    Returns (occupancy[int32], ensemble).
    """
    system = cast(Mapping[str, Any], doc["system"])
    prototype = cast(str, system["prototype"])
    prototype_params = cast(Mapping[str, Any], system["prototype_params"])
    supercell_diag = tuple(system["supercell_diag"])
    replace_element = cast(str, system["replace_element"])

    conv = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))
    counts_clean = {str(k): int(v) for k, v in counts.items()}
    validate_counts_for_sublattice(
        conv_cell=conv,
        supercell_diag=cast(Tuple[int, int, int], tuple(supercell_diag)),
        replace_element=replace_element,
        counts=counts_clean,
    )

    snap = make_one_snapshot(
        conv_cell=conv,
        supercell_diag=cast(Tuple[int, int, int], tuple(supercell_diag)),
        replace_element=replace_element,
        counts=counts_clean,
        rng=rng,
    )
    struct = AseAtomsAdaptor.get_structure(snap)  # type: ignore[arg-type]

    payload = cast(Mapping[str, Any], doc["payload"])
    ce = ClusterExpansion.from_dict(dict(payload))
    sc_matrix = np.diag(cast(Sequence[int], system["supercell_diag"]))
    ensemble = Ensemble.from_cluster_expansion(ce, supercell_matrix=sc_matrix)

    proc = ensemble.processor
    occ = proc.cluster_subspace.occupancy_from_structure(struct, encode=True)
    occ = np.asarray(occ, dtype=np.int32)

    n_sites = getattr(proc, "num_sites", occ.shape[0])
    if occ.shape[0] != n_sites:
        raise RuntimeError(f"Occupancy length {occ.shape[0]} != processor sites {n_sites}")

    return occ, ensemble


# ----------------------------- driver ------------------------------------- #

def run_wl(spec: WLSamplerSpec) -> WLResult:
    if not spec.composition_counts:
        raise ValueError("WLSamplerSpec.composition_counts is required for canonical WL.")

    doc = _rehydrate_ce(spec.ce_key)
    rng = np.random.default_rng(spec.seed)

    # Initial occupancy (int32) and ensemble
    occ, ensemble = _initial_occupancy_from_counts(
        doc=doc,
        counts=spec.composition_counts,
        rng=rng,
    )

    # Starting energy (eV per supercell) from CE
    E0 = float(np.dot(ensemble.compute_feature_vector(occ), ensemble.natural_parameters))

    # Pilot window using *exact same counts* as WL run (now guaranteed eV/supercell)
    H_pilot = _pilot_enthalpies(
        doc,
        n=spec.pilot_samples,
        rng=rng,
        counts_required=spec.composition_counts,
    )
    mu = float(np.median(H_pilot))
    sigma = float(np.std(H_pilot))
    if sigma <= 0:
        sigma = spec.bin_width or 1e-3

    raw_min = mu - spec.sigma_multiplier * sigma
    raw_max = mu + spec.sigma_multiplier * sigma
    H_min = snap_floor(raw_min, spec.bin_width, ANCHOR)
    H_max = snap_ceil(raw_max, spec.bin_width, ANCHOR)

    # Sanity: ensure we actually have multiple bins
    n_bins = int(np.floor((H_max - H_min) / float(spec.bin_width) + 1e-12))
    if n_bins < 2:
        raise RuntimeError(
            f"WL window too narrow or mis-binned: bins={n_bins} from "
            f"[{H_min}, {H_max}) with width={spec.bin_width}"
        )

    # Fail hard if the initial enthalpy is not within the pilot window
    if not (H_min <= E0 < H_max):
        raise RuntimeError(
            "Initial enthalpy E0 is outside the WL window derived from counts-only pilot. "
            f"E0={E0:.6f}, window=[{H_min:.6f}, {H_max:.6f}), mu={mu:.6f}, sigma={sigma:.6f}, "
            f"multiplier={spec.sigma_multiplier}, bin={spec.bin_width}."
        )

    # Build a Wang–Landau Sampler (disorder-style)
    sampler = Sampler.from_ensemble(
        ensemble,
        kernel_type="Wang-Landau",
        bin_size=spec.bin_width,
        step_type=spec.step_type,   # keep spec behavior; disorder hard-coded "swap"
        flatness=0.8,
        min_enthalpy=H_min,
        max_enthalpy=H_max,
        seeds=[int(spec.seed)],
    )

    # Run with your current thin_by policy (≈100 snapshots)
    sampler.run(spec.steps, occ, thin_by=max(1, spec.steps // 100), progress=False)

    # Pull results from the SampleContainer
    hist_all = np.asarray(sampler.samples.get_trace_value("histogram")[-1], dtype=int)
    ent_all  = np.asarray(sampler.samples.get_trace_value("entropy")[-1], dtype=float)

    # Modification-factor trace (exact key used in disorder)
    mod_factor_trace = sampler.samples.get_trace_value("mod_factor")
    mod_factor_trace = [float(x) for x in mod_factor_trace]  # ensure JSON serializable

    # Reconstruct levels exactly from our window and bin width
    levels_all = np.arange(H_min, H_max, spec.bin_width, dtype=float)

    mask = hist_all > 0
    levels    = levels_all[mask]
    entropy   = ent_all[mask]
    histogram = hist_all[mask]
    visited_mask = mask.copy()

    return WLResult(
        levels=levels,
        entropy=entropy,
        histogram=histogram,
        visited_mask=visited_mask,
        grid_anchor=ANCHOR,
        bin_width=spec.bin_width,
        window_used=(H_min, H_max),
        meta=dict(
            ce_key=spec.ce_key,
            steps=spec.steps,
            seed=spec.seed,
            step_type=spec.step_type,
            composition_counts={str(k): int(v) for k, v in spec.composition_counts.items()},
            check_period=spec.check_period,
            update_period=spec.update_period,
            pilot_samples=spec.pilot_samples,
            sigma_multiplier=spec.sigma_multiplier,
            E0=E0,
            pilot_mu=mu,
            pilot_sigma=sigma,
            mod_factor_trace=mod_factor_trace,
        ),
    )
