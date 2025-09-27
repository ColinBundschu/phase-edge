from typing import Any, Mapping, Sequence, TypedDict, cast

import numpy as np

from smol.moca import Sampler
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from phaseedge.storage.wang_landau import WLCheckpointDoc, ensure_wl_output_indexes, fetch_wl_tip
from phaseedge.schemas.wl_sampler_spec import WLSamplerSpec
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.sampling.infinite_wang_landau import InfiniteWangLandau  # ensure registered
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.jobs.store_ce_model import rehydrate_ensemble_by_ce_key
from phaseedge.utils.keys import compute_wl_checkpoint_key


# ---- shared helpers -------------------------------------------------------

def _snapshot_struct_and_occ_from_counts(
    *,
    ce_key: str,
    sublattice_labels: Sequence[str],
    composition_counts: Mapping[str, int],
    rng: np.random.Generator,
) -> tuple[Structure, np.ndarray]:
    """
    Create ONE valid snapshot structure at the requested WL composition and
    the corresponding encoded occupancy for the ensemble.
    """
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")

    conv = make_prototype(doc["prototype"], **doc["prototype_params"])
    sx, sy, sz = (int(x) for x in doc["supercell_diag"])

    # Count the number of each sublattice label in the supercell
    sc = conv.repeat((sx, sy, sz))
    symbols = np.array(sc.get_chemical_symbols())
    sublattice_counts: dict[str, int] = {}
    for sl in sublattice_labels:
        n_sites = int(np.sum(symbols == sl))
        if n_sites <= 0:
            raise ValueError(f"Sublattice label '{sl}' not found in prototype structure.")
        sublattice_counts[sl] = n_sites

    # Evenly distribute counts to sublattices (deterministically assigning remainder)
    # composition_map: {placeholder_symbol -> {element -> count}}
    composition_map: dict[str, dict[str, int]] = {sl: {} for sl in sublattice_counts}

    # --- sanity checks
    total_sites = int(sum(int(v) for v in sublattice_counts.values()))
    total_requested = int(sum(int(v) for v in composition_counts.values()))
    if total_requested != total_sites:
        raise ValueError(
            f"composition_counts sum ({total_requested}) != total sublattice sites ({total_sites})"
        )

    # Precompute sizes in a deterministic order of sublattices
    # We'll always iterate keys in sorted order to keep results stable.
    subl_sorted = sorted(composition_map.keys())
    sizes = {sl: int(sublattice_counts[sl]) for sl in subl_sorted}

    # Allocate each element across sublattices
    for elem, N in sorted(composition_counts.items()):  # element order stable
        N = int(N)
        if N < 0:
            raise ValueError(f"Negative total for element '{elem}': {N}")
        if N == 0:
            continue

        # Quotas per sublattice
        quotas: dict[str, float] = {sl: (N * sizes[sl]) / float(total_sites) for sl in subl_sorted}
        base: dict[str, int] = {sl: int(np.floor(quotas[sl])) for sl in subl_sorted}
        assigned = sum(base.values())
        remainder = N - assigned
        if remainder < 0:
            # Shouldn't happen with floor; guard anyway.
            raise RuntimeError(f"Internal apportionment error for '{elem}' (remainder < 0).")

        # Fractional parts for tie-breaking
        fracs: dict[str, float] = {sl: (quotas[sl] - base[sl]) for sl in subl_sorted}

        # Rank sublattices by descending fractional part, then by sublattice key (stable)
        rank = sorted(subl_sorted, key=lambda sl: (-fracs[sl], sl))

        # Start with base, then hand out remainders
        for sl in subl_sorted:
            if base[sl] > 0:
                composition_map[sl][elem] = composition_map[sl].get(elem, 0) + base[sl]
        for i in range(remainder):
            sl = rank[i]
            composition_map[sl][elem] = composition_map[sl].get(elem, 0) + 1

    # Final check: per-sublattice totals must match sublattice size exactly
    for sl in subl_sorted:
        subtotal = sum(int(v) for v in composition_map[sl].values())
        if subtotal != sizes[sl]:
            raise RuntimeError(
                f"Sublattice '{sl}' assigned {subtotal} atoms, expected {sizes[sl]}."
            )

    snap = make_one_snapshot(
        conv_cell=conv,
        supercell_diag=(sx, sy, sz),
        composition_map=composition_map,
        rng=rng,
    )
    struct = AseAtomsAdaptor.get_structure(snap)  # type: ignore[arg-type]
    ensemble = rehydrate_ensemble_by_ce_key(ce_key)
    occ = ensemble.processor.cluster_subspace.occupancy_from_structure(struct, encode=True)
    occ = np.asarray(occ, dtype=np.int32)
    n_sites = getattr(ensemble.processor, "num_sites", occ.shape[0])
    if occ.shape[0] != n_sites:
        raise RuntimeError(f"Occupancy length {occ.shape[0]} != processor sites {n_sites}")
    return struct, occ


def _build_sublattice_indices(*, ce_key: str, sl_comp_map: dict[str, dict[str, int]]) -> tuple[dict[str, np.ndarray], dict[int, str]]:
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
    snap = make_one_snapshot(
        conv_cell=conv,
        supercell_diag=(sx, sy, sz),
        composition_map=sl_comp_map,
        rng=rng,
    )
    struct = AseAtomsAdaptor.get_structure(snap)  # type: ignore[arg-type]
    ensemble = rehydrate_ensemble_by_ce_key(ce_key)
    occ = ensemble.processor.cluster_subspace.occupancy_from_structure(struct, encode=True)
    [active_sl] = ensemble.active_sublattices
    code_to_elem = {int(code): str(elem) for code, elem in zip(active_sl.encoding, active_sl.species)}
    if any(len(v.keys()) != 1 for v in sl_comp_map.values()):
        raise NotImplementedError("This helper only supports single-element sublattices.")
    inverted_comp_map = {list(v.keys())[0]: k for k, v in sl_comp_map.items()}

    sl_map: dict[str, np.ndarray] = {}
    for sl in sl_comp_map.keys():
        # We use get here because just because an element can appear in the sublattice, does not mean its
        # used as part of the canonical mapping. Since each sublattice has one label, we use a single
        # element to identify it.
        idx = np.where([inverted_comp_map.get(code_to_elem[int(o)]) == sl for o in occ])[0]
        # exclude sites not in active_sl.sites
        idx = idx[np.isin(idx, active_sl.sites)]
        if idx.size == 0:
            raise ValueError(f"Sublattice label '{sl}' not found in prototype structure {occ}.")
        sl_map[sl] = idx
    return sl_map, code_to_elem


# ---- Chunk runner ---------------------------------------------------------


def run_wl_chunk(spec: WLSamplerSpec) -> WLCheckpointDoc:
    ensure_wl_output_indexes()

    """Extend the WL chain by `run_spec.steps` steps, idempotently, and write a checkpoint."""
    tip = fetch_wl_tip(spec.wl_key)
        
    ensemble = rehydrate_ensemble_by_ce_key(spec.ce_key)
    rng = np.random.default_rng(int(spec.seed))

    # Precompute sublattice site-index mapping for this WL key/spec
    sublattice_indices, active_codes_to_elems = _build_sublattice_indices(
        ce_key=spec.ce_key,
        sl_comp_map=spec.sl_comp_map,
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
        active_codes_to_elems=active_codes_to_elems,
    )

    # Parent hash & restore point
    if tip is None:
        # Fresh initialization
        parent_wl_checkpoint_key = "GENESIS"
        step_start = 0
        _, occ = _snapshot_struct_and_occ_from_counts(
            ce_key=spec.ce_key,
            sublattice_labels=list(spec.sl_comp_map.keys()),
            composition_counts=spec.composition_counts,
            rng=rng,
        )
    else:
        # Load kernel + occupancy from tip
        parent_wl_checkpoint_key = str(tip["wl_checkpoint_key"])
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

    wl_checkpoint_key = compute_wl_checkpoint_key(
        wl_key=spec.wl_key,
        parent_wl_checkpoint_key=parent_wl_checkpoint_key,
        state=end_state,
        occupancy=occ_last,
    )
    return {
        "kind": "WLCheckpointDoc",
        "wl_key": spec.wl_key,
        "wl_checkpoint_key": wl_checkpoint_key,
        "parent_wl_checkpoint_key": parent_wl_checkpoint_key,
        "samples_per_bin": spec.samples_per_bin,
        "checkpoint_steps": spec.steps,
        "step_end": step_start + spec.steps,
        "mod_updates": [{"step": int(st), "m": float(m)} for (st, m) in k.pop_mod_updates()],
        "bin_samples": [{"bin": int(b), "occ": occ} for b, occs in bin_samples.items() for occ in occs],
        "cation_counts": cation_counts_flat,
        "production_mode": spec.production_mode,
        "collect_cation_stats": spec.collect_cation_stats,
        "state": end_state,
        "occupancy": occ_last.tolist(),
    }
