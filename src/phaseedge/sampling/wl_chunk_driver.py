from typing import Any, Mapping, Sequence

import numpy as np
from pymongo.errors import DuplicateKeyError

from smol.moca import Sampler
from smol.moca.ensemble import Ensemble
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.sampling.infinite_wang_landau import InfiniteWangLandau  # ensure registered
from phaseedge.storage.wl_checkpoint_store import ensure_indexes, get_tip, insert_checkpoint
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.utils.rehydrators import rehydrate_ensemble_by_ce_key


# ---- shared helpers -------------------------------------------------------

def _initial_occupancy_from_counts(
    *, ce_key: str, ensemble: Ensemble, sublattice_labels: Sequence[str], composition_counts: Mapping[str, int], rng: np.random.Generator,
) -> np.ndarray:
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
    occ = ensemble.processor.cluster_subspace.occupancy_from_structure(struct, encode=True)
    occ = np.asarray(occ, dtype=np.int32)
    n_sites = getattr(ensemble.processor, "num_sites", occ.shape[0])
    if occ.shape[0] != n_sites:
        raise RuntimeError(f"Occupancy length {occ.shape[0]} != processor sites {n_sites}")
    return occ


# ---- Chunk runner ---------------------------------------------------------

def run_wl_chunk(spec: WLSamplerSpec) -> dict[str, Any]:
    """Extend the WL chain by `run_spec.steps` steps, idempotently, and write a checkpoint."""
    ensure_indexes()
    tip = get_tip(spec.wl_key)
    ensemble = rehydrate_ensemble_by_ce_key(spec.ce_key)
    rng = np.random.default_rng(int(spec.seed))
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
    )

    # Parent hash & restore point
    if tip is None:
        # Fresh initialization
        parent_hash = "GENESIS"
        step_start = 0
        occ = _initial_occupancy_from_counts(
            ce_key=spec.ce_key,
            ensemble=ensemble,
            sublattice_labels=spec.sublattice_labels,
            composition_counts=spec.composition_counts,
            rng=rng,
        )
    else:
        # Load kernel + occupancy from tip
        parent_hash = str(tip["hash"])
        step_start = int(tip["step_end"])
        occ = np.asarray(tip["occupancy"], dtype=np.int32)
        sampler.mckernels[0].load_state(tip["state"])

    # Minimize memory retention during the run (keep just one retained sample).
    thin_by = max(1, spec.steps)

    # Run the chunk
    sampler.run(spec.steps, occ, thin_by=thin_by, progress=False)

    # Capture state & occupancy (occupancy returned is last sample’s)
    k = sampler.mckernels[0]
    end_state = k.state()
    occ_last = sampler.samples.get_occupancies(flat=False)[-1][0].astype(np.int32)

    updates_local = k.pop_mod_updates()  # list[(step_abs, m_after)]
    mod_updates = [{"step": int(st), "m": float(m)} for (st, m) in updates_local]

    # capture any per-bin samples harvested this chunk
    bin_samples: dict[int, list[list[int]]] = k.pop_bin_samples()

    step_end = step_start + spec.steps

    # Defensive: fail fast if tip moved between our read and now
    latest_now = get_tip(spec.wl_key)
    if latest_now is not None and parent_hash != latest_now["hash"]:
        raise RuntimeError("Tip moved while running; aborting write to avoid fork.")

    # Try insert; uniqueness on (wl_key,parent_hash) ensures linear chain
    try:
        _id, doc_inserted = insert_checkpoint(
            wl_key=spec.wl_key,
            step_end=step_end,
            chunk_size=spec.steps,
            parent_hash=parent_hash,
            state=end_state,
            occupancy=occ_last,
            # --- first-class top-level metadata ---
            mod_updates=mod_updates,
            bin_samples=[{"bin": int(b), "occ": occ} for b, occs in bin_samples.items() for occ in occs],
            samples_per_bin=int(spec.samples_per_bin),
        )
    except DuplicateKeyError as e:
        raise RuntimeError("Checkpoint insert conflict (not on tip or duplicate). Retry from new tip.") from e

    return {
        "_id": _id,
        "wl_key": spec.wl_key,
        "step_end": step_end,
        "parent_hash": parent_hash,
        "hash": doc_inserted["hash"],
        "chunk_size": spec.steps,
    }
