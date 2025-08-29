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
from phaseedge.utils.grid import ANCHOR

# Ensure our kernel class is imported/registered with smol's factory
# (Derived-class discovery requires the class to be imported into memory.)
from phaseedge.sampling.infinite_wang_landau import InfiniteWangLandau  # noqa: F401


# ----------------------------- helpers ------------------------------------ #

def _rehydrate_ce(ce_key: str) -> Mapping[str, Any]:
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")
    return cast(Mapping[str, Any], doc)


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

    # Starting energy (eV per supercell) from CE (used for logging/meta)
    E0 = float(np.dot(ensemble.compute_feature_vector(occ), ensemble.natural_parameters))

    # Build a Wang–Landau Sampler using our infinite-window kernel.
    # This mirrors Sampler.from_ensemble(...) usage so the driver remains unchanged.
    sampler = Sampler.from_ensemble(
        ensemble,
        kernel_type="InfiniteWangLandau",
        bin_size=spec.bin_width,
        step_type=spec.step_type,   # keep spec behavior
        flatness=0.8,
        seeds=[int(spec.seed)],
        check_period=spec.check_period,
        update_period=spec.update_period,
    )

    # Run with your current thin_by policy (≈100 snapshots)
    sampler.run(spec.steps, occ, thin_by=max(1, spec.steps // 100), progress=False)

    # Read final state directly from the kernel to avoid fixed-shape coupling.
    k = sampler.mckernels[0]
    levels_all = np.asarray(k.levels, dtype=float)
    ent_all = np.asarray(k.entropy, dtype=float)
    hist_all = np.asarray(k.histogram, dtype=int)
    bin_indices = np.asarray(getattr(k, "bin_indices", None), dtype=int)

    # Modification-factor trace (exact key used in disorder)
    mod_factor_trace = sampler.samples.get_trace_value("mod_factor")
    mod_factor_trace = [float(x) for x in mod_factor_trace]  # ensure JSON serializable

    return WLResult(
        levels=levels_all,
        entropy=ent_all,
        histogram=hist_all,
        bin_indices=bin_indices,
        grid_anchor=ANCHOR,
        bin_width=spec.bin_width,
        meta=dict(
            ce_key=spec.ce_key,
            steps=spec.steps,
            seed=spec.seed,
            step_type=spec.step_type,
            composition_counts={str(k): int(v) for k, v in spec.composition_counts.items()},
            check_period=spec.check_period,
            update_period=spec.update_period,
            E0=E0,
            mod_factor_trace=mod_factor_trace,
        ),
    )
