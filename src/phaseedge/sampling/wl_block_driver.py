from typing import Any, Mapping

import numpy as np

from smol.moca import Sampler
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.storage.wang_landau import WLBlockDoc, verify_wl_output_indexes, fetch_wl_tip
from phaseedge.schemas.wl_sampler_spec import WLSamplerSpec
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.sampling.infinite_wang_landau import InfiniteWangLandau  # ensure registered
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.jobs.store_ce_model import rehydrate_ensemble_by_ce_key
from phaseedge.utils.keys import compute_wl_block_key


# ---- shared helpers -------------------------------------------------------

def _occ_from_initial_comp_map(
    *,
    ce_key: str,
    initial_comp_map: Mapping[str, Mapping[str, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create ONE valid snapshot structure at the requested WL composition and
    the corresponding encoded occupancy for the ensemble.
    """
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")

    conv = make_prototype(doc["prototype"], **doc["prototype_params"])
    sx, sy, sz = (int(x) for x in doc["supercell_diag"])
    snap = make_one_snapshot(
        conv_cell=conv,
        supercell_diag=(sx, sy, sz),
        composition_map=initial_comp_map,
        rng=rng,
    )
    struct = AseAtomsAdaptor.get_structure(snap)  # type: ignore[arg-type]
    ensemble = rehydrate_ensemble_by_ce_key(ce_key)
    occ = ensemble.processor.cluster_subspace.occupancy_from_structure(struct, encode=True)
    occ = np.asarray(occ, dtype=np.int32)
    n_sites = getattr(ensemble.processor, "num_sites", occ.shape[0])
    if occ.shape[0] != n_sites:
        raise RuntimeError(f"Occupancy length {occ.shape[0]} != processor sites {n_sites}")
    return occ


def _build_sublattice_indices(*, ce_key: str, initial_comp_map: dict[str, dict[str, int]]) -> dict[str, tuple[np.ndarray, dict[int, str]]]:
    """
    Build label -> site-index map from the CE prototype+supercell used by WL.
    """
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")

    # Make a new rng for this operation (not part of the returned state)
    rng = np.random.default_rng(12345)
    conv = make_prototype(doc["prototype"], **doc["prototype_params"])
    sx, sy, sz = (int(x) for x in doc["supercell_diag"])
    sl_comp_map = {k: {k: sum(initial_comp_map[k].values())} for k in initial_comp_map.keys()}
    snap = make_one_snapshot(
        conv_cell=conv,
        supercell_diag=(sx, sy, sz),
        composition_map=sl_comp_map,
        rng=rng,
    )
    struct = AseAtomsAdaptor.get_structure(snap)  # type: ignore[arg-type]
    ensemble = rehydrate_ensemble_by_ce_key(ce_key)
    occ = ensemble.processor.cluster_subspace.occupancy_from_structure(struct, encode=False)
    sl_map: dict[str, tuple[np.ndarray, dict[int, str]]] = {}
    for sublattice in ensemble.active_sublattices:
        code_to_elem = {int(code): str(elem) for code, elem in zip(sublattice.encoding, sublattice.species)}
        for elem in [str(s) for s in sublattice.species]:
            if elem not in sl_comp_map:
                continue # this element is not a placeholder label for a sublattice
            idx = np.where([str(o.symbol) == elem for o in occ])[0]
            idx = idx[np.isin(idx, sublattice.sites)]
            if idx.size == 0:
                continue # This is a placeholder for a different sublattice
            if elem in sl_map:
                raise ValueError(f"Placeholder lattice identification element '{elem}' appears in multiple sublattices.")
            sl_map[elem] = (idx, code_to_elem)

    for placeholder in sl_comp_map.keys():
        if placeholder not in sl_map:
            raise ValueError(f"Could not find any sites for sublattice placeholder '{placeholder}' in the prototype structure.")

    return sl_map


# ---- Chunk runner ---------------------------------------------------------


def run_wl_block(spec: WLSamplerSpec) -> WLBlockDoc:
    verify_wl_output_indexes()

    """Extend the WL chain by `run_spec.steps` steps, idempotently, and write a block."""
    tip = fetch_wl_tip(spec.wl_key)
        
    ensemble = rehydrate_ensemble_by_ce_key(spec.ce_key)
    rng = np.random.default_rng(int(spec.seed))

    # Precompute sublattice site-index mapping for this WL key/spec
    sublattice_indices = _build_sublattice_indices(
        ce_key=spec.ce_key,
        initial_comp_map=spec.initial_comp_map,
    )

    sampler = Sampler.from_ensemble(
        ensemble,
        kernel_type="InfiniteWangLandau",
        bin_size=spec.bin_width,
        step_type=spec.step_type,
        flatness=0.8,
        seeds=[int(spec.seed)],
        check_period=spec.check_period,
        update_period=spec.update_period,
        samples_per_bin=int(spec.samples_per_bin),  # runtime capture policy (non-key)
        # ---- NEW runtime/statistics configuration passed to the kernel ----
        collect_cation_stats=spec.collect_cation_stats,
        production_mode=spec.production_mode,
        sublattice_indices=sublattice_indices,
        reject_cross_sublattice_swaps=spec.reject_cross_sublattice_swaps,
    )

    # Parent hash & restore point
    if tip is None:
        # Fresh initialization
        parent_wl_block_key = "GENESIS"
        step_start = 0
        occ = _occ_from_initial_comp_map(ce_key=spec.ce_key, initial_comp_map=spec.initial_comp_map, rng=rng)
    else:
        # Load kernel + occupancy from tip
        parent_wl_block_key = str(tip["wl_block_key"])
        step_start = int(tip["step_end"])
        occ = np.asarray(tip["occupancy"], dtype=np.int32)
        sampler.mckernels[0].load_state(tip["state"])

    # Minimize memory retention during the run (keep just one retained sample).
    thin_by = max(1, spec.steps)

    # Run the chunk
    sampler.run(spec.steps, occ, thin_by=thin_by, progress=False)

    # Capture state & occupancy (occupancy returned is last sample's)
    k = sampler.mckernels[0]
    end_state = k.state()
    occ_last = sampler.samples.get_occupancies(flat=False)[-1][0].astype(np.int32)

    # capture any per-bin samples harvested this chunk
    bin_samples: dict[int, list[list[int]]] = k.pop_bin_samples()

    bin_cation_counts = k.pop_bin_cation_counts()
    # Flatten for storage
    cation_counts_flat: list[dict[str, Any]] = [
        {
            "bin": int(b),
            "sublattice": sl,
            "element": elem,
            "n_sites": int(n_sites),
            "count": int(count),
        }
        for b, sl_map in bin_cation_counts.items()
        for sl, elem_map in sl_map.items()
        for elem, hist in elem_map.items()
        for n_sites, count in hist.items()
    ]

    wl_block_key = compute_wl_block_key(
        wl_key=spec.wl_key,
        parent_wl_block_key=parent_wl_block_key,
        state=end_state,
        occupancy=occ_last,
    )
    return {
        "kind": "WLBlockDoc",
        "wl_key": spec.wl_key,
        "wl_block_key": wl_block_key,
        "parent_wl_block_key": parent_wl_block_key,
        "samples_per_bin": spec.samples_per_bin,
        "block_steps": spec.steps,
        "step_end": step_start + spec.steps,
        "mod_updates": [{"step": int(st), "m": float(m)} for (st, m) in k.pop_mod_updates()],
        "bin_samples": [{"bin": int(b), "occ": occ} for b, occs in bin_samples.items() for occ in occs],
        "cation_counts": cation_counts_flat,
        "production_mode": spec.production_mode,
        "collect_cation_stats": spec.collect_cation_stats,
        "state": end_state,
        "occupancy": occ_last.tolist(),
    }
