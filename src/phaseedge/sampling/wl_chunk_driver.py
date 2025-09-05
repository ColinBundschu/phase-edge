from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
from pymongo.errors import DuplicateKeyError

from smol.moca import Sampler
from smol.moca.ensemble import Ensemble
from smol.cofe import ClusterExpansion
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.storage.ce_store import lookup_ce_by_key
from phaseedge.sampling.infinite_wang_landau import InfiniteWangLandau  # ensure registered
from phaseedge.storage.wl_checkpoint_store import ensure_indexes, get_tip, insert_checkpoint
from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import make_one_snapshot, validate_counts_for_sublattice


# ---- shared helpers -------------------------------------------------------

def _rehydrate_ce(ce_key: str) -> Mapping[str, Any]:
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")
    return cast(Mapping[str, Any], doc)

def _initial_occupancy_from_counts(
    *, doc: Mapping[str, Any], counts: Mapping[str, int], rng: np.random.Generator
) -> tuple[np.ndarray, Ensemble]:
    system = cast(Mapping[str, Any], doc["system"])
    prototype = cast(str, system["prototype"])
    prototype_params = cast(Mapping[str, Any], system["prototype_params"])
    supercell_diag = tuple(system["supercell_diag"])
    replace_element = cast(str, system["replace_element"])

    conv = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))
    counts_clean = {str(k): int(v) for k, v in counts.items()}
    validate_counts_for_sublattice(
        conv_cell=conv,
        supercell_diag=tuple(supercell_diag),
        replace_element=replace_element,
        counts=counts_clean,
    )
    snap = make_one_snapshot(
        conv_cell=conv,
        supercell_diag=tuple(supercell_diag),
        replace_element=replace_element,
        counts=counts_clean,
        rng=rng,
    )
    struct = AseAtomsAdaptor.get_structure(snap)  # type: ignore[arg-type]

    payload = cast(Mapping[str, Any], doc["payload"])
    ce = ClusterExpansion.from_dict(dict(payload))
    sc_matrix = np.diag(cast(tuple[int, int, int], tuple(supercell_diag)))
    ensemble = Ensemble.from_cluster_expansion(ce, supercell_matrix=sc_matrix)

    proc = ensemble.processor
    occ = proc.cluster_subspace.occupancy_from_structure(struct, encode=True)
    occ = np.asarray(occ, dtype=np.int32)
    n_sites = getattr(proc, "num_sites", occ.shape[0])
    if occ.shape[0] != n_sites:
        raise RuntimeError(f"Occupancy length {occ.shape[0]} != processor sites {n_sites}")
    return occ, ensemble


# ---- Chunk runner ---------------------------------------------------------

def run_wl_chunk(spec: WLSamplerSpec) -> dict[str, Any]:
    """Extend the WL chain by `run_spec.steps` steps, idempotently, and write a checkpoint."""
    ensure_indexes()
    tip = get_tip(spec.wl_key)

    # Parent hash & restore point
    if tip is None:
        parent_hash = "GENESIS"
        # Fresh initialization
        doc = _rehydrate_ce(spec.ce_key)
        rng = np.random.default_rng(int(spec.seed))
        occ, ensemble = _initial_occupancy_from_counts(doc=doc, counts=spec.composition_counts, rng=rng)
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
        step_start = 0
    else:
        parent_hash = str(tip["hash"])
        step_start = int(tip["step_end"])

        # Rehydrate ensemble and sampler
        doc = _rehydrate_ce(spec.ce_key)
        rng = np.random.default_rng(int(spec.seed))
        _, ensemble = _initial_occupancy_from_counts(doc=doc,
                                                     counts=spec.composition_counts,
                                                     rng=rng)
        sampler = Sampler.from_ensemble(
            ensemble,
            kernel_type="InfiniteWangLandau",
            bin_size=spec.bin_width,
            step_type=spec.step_type,
            flatness=0.8,
            seeds=[int(spec.seed)],
            check_period=spec.check_period,
            update_period=spec.update_period,
            samples_per_bin=int(spec.samples_per_bin),
        )
        # Load kernel + occupancy from tip
        k = sampler.mckernels[0]
        k.load_state(tip["state"])
        occ = np.asarray(tip["occupancy"], dtype=np.int32)

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
