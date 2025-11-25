from typing import Any, Mapping

import numpy as np

from smol.moca import Sampler
from smol.moca.ensemble import Ensemble
from pymatgen.core import Structure

from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.storage.wang_landau import WLBlockDoc, verify_wl_output_indexes, fetch_wl_tip
from phaseedge.schemas.wl_sampler_spec import WLSamplerSpec
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.sampling.infinite_wang_landau import InfiniteWangLandau  # ensure registered
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

    prototype_spec = PrototypeSpec.from_dict(doc["prototype_spec"])
    sx, sy, sz = (int(x) for x in doc["supercell_diag"])
    struct = make_one_snapshot(
        primitive_cell=prototype_spec.primitive_cell,
        supercell_diag=(sx, sy, sz),
        composition_map=initial_comp_map,
        rng=rng,
    )
    ensemble = rehydrate_ensemble_by_ce_key(ce_key)
    occ = ensemble.processor.cluster_subspace.occupancy_from_structure(struct, encode=True)
    occ = np.asarray(occ, dtype=np.int32)
    n_sites = getattr(ensemble.processor, "num_sites", occ.shape[0])
    if occ.shape[0] != n_sites:
        raise RuntimeError(f"Occupancy length {occ.shape[0]} != processor sites {n_sites}")
    return occ


def _build_sublattices(*, ce_key: str) -> tuple[dict[str, dict[str, int]], Structure]:
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")

    # Make a new rng for this operation (not part of the returned state)
    rng = np.random.default_rng(12345)
    prototype_spec = PrototypeSpec.from_dict(doc["prototype_spec"])
    sx, sy, sz = (int(x) for x in doc["supercell_diag"])
    multiplier = sx * sy * sz
    sl_comp_map = {k: {k: v * multiplier} for k, v in prototype_spec.active_sublattice_counts.items()}
    struct = make_one_snapshot(
        primitive_cell=prototype_spec.primitive_cell,
        supercell_diag=(sx, sy, sz),
        composition_map=sl_comp_map,
        rng=rng,
    )
    return sl_comp_map, struct


def _struct_to_occ_site_mapping(ensemble: Ensemble, structure: Structure):
    scmatrix = ensemble.processor.cluster_subspace.scmatrix_from_structure(structure)
    supercell = ensemble.processor.cluster_subspace.structure.copy()
    supercell.make_supercell(scmatrix)
    site_mapping = ensemble.processor.cluster_subspace.structure_site_mapping(supercell, structure)
    # Ensure that the mapping is a bijective list of indices from 0 to N-1
    if set(site_mapping) != set(range(len(site_mapping))):
        raise RuntimeError("Site mapping is not a bijection from 0 to N-1.")
    return site_mapping


def _build_sublattice_indices(*, ensemble: Ensemble, sl_struct: Structure, sl_comp_map: dict[str, dict[str, int]]) -> dict[str, tuple[np.ndarray, dict[int, str]]]:
    """
    Build label -> site-index map from the CE prototype+supercell used by WL.
    """
    occ = ensemble.processor.cluster_subspace.occupancy_from_structure(sl_struct, encode=False)
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


def _build_sublattice_nn_pairs(
    *,
    sl_struct: Structure,
    sublattice_indices: dict[str, tuple[np.ndarray, dict[int, str]]],
    rtol: float = 1e-3,
) -> dict[str, np.ndarray]:
    """
    For each sublattice placeholder label, build a nearest-neighbor edge list
    between *sites on that sublattice*.

    Strategy (per sublattice):
      - Take all sites belonging to that sublattice.
      - Compute the full pairwise distance matrix (with PBC).
      - For each site i:
          * find its minimum distance d_min(i) to any OTHER site on that sublattice
          * define a cutoff d_cut(i) = d_min(i) * (1 + rtol)
          * mark all j with d_ij <= d_cut(i) as nearest neighbors
      - Collect undirected edges (i, j) with i < j to avoid double-counting.

    Returns
    -------
    nn_pairs_by_sl : dict[str, np.ndarray]
        Mapping placeholder label -> array of shape (n_pairs, 2) of
        global site indices (wrt the WL supercell structure).
        If a sublattice has no NN pairs, the array is shape (0, 2).
    """
    nn_pairs_by_sl: dict[str, np.ndarray] = {}

    lattice = sl_struct.lattice

    for placeholder, (idx, _code_to_elem) in sublattice_indices.items():
        indices = np.asarray(idx, dtype=int)
        n = indices.size

        if n < 2:
            # Can't form any pairs
            nn_pairs_by_sl[placeholder] = np.empty((0, 2), dtype=np.int32)
            continue

        # Fractional coords for these sites
        fcoords = np.array([sl_struct[i].frac_coords for i in indices], dtype=float)

        # Full distance matrix with PBC
        dmat = lattice.get_all_distances(fcoords, fcoords)  # shape (n, n)
        # Ignore self-distances
        np.fill_diagonal(dmat, np.inf)

        # Per-site minimum NN distance
        min_dists = dmat.min(axis=1)  # shape (n,)

        pairs: list[tuple[int, int]] = []

        for row in range(n):
            d_min = float(min_dists[row])
            if not np.isfinite(d_min):
                # Pathological case: isolated site on this "sublattice"
                continue

            cutoff = d_min * (1.0 + float(rtol))
            neighbors = np.where(dmat[row] <= cutoff)[0]

            i_global = int(indices[row])
            for col in neighbors:
                if col <= row:
                    # ensure undirected i<j
                    continue
                j_global = int(indices[col])
                pairs.append((i_global, j_global))

        if pairs:
            nn_pairs_by_sl[placeholder] = np.asarray(pairs, dtype=np.int32)
        else:
            nn_pairs_by_sl[placeholder] = np.empty((0, 2), dtype=np.int32)

    return nn_pairs_by_sl



def _expected_random_unlike_pairs(
    *,
    num_edges: int,
    species_counts: Mapping[str, int],
) -> float:
    """
    Analytic expectation E[O_D] for a domain D under ideal random mixing
    with fixed composition.

    Parameters
    ----------
    num_edges : int
        M_D, the number of NN edges in the domain graph (e.g., len(nn_pairs)).
    species_counts : Mapping[str, int]
        Mapping element label -> number of sites of that element in the domain,
        e.g. {"Li": 32, "Mg": 16, "Ti": 16} for that domain.

        Must satisfy N = sum_a n_a >= 0.

    Returns
    -------
    float
        E[O_D^random] = M_D * P(unlike), where

          P(unlike) = [sum_{a != b} n_a n_b] / [N (N - 1)]

        for a finite system with fixed composition.
    """
    counts = np.array(list(species_counts.values()), dtype=int)
    N = int(counts.sum())
    M = int(num_edges)

    if M == 0 or N < 2:
        return 0.0

    counts_f = counts.astype(float)
    # N^2 - sum_a n_a^2 = sum_{a != b} n_a n_b
    numerator = float(N * N - np.sum(counts_f * counts_f))
    denom = float(N * (N - 1))
    p_unlike = numerator / denom

    return M * p_unlike


# ---- Chunk runner ---------------------------------------------------------


def run_wl_block(spec: WLSamplerSpec) -> WLBlockDoc:
    verify_wl_output_indexes()

    """Extend the WL chain by `run_spec.steps` steps, idempotently, and write a block."""
    tip = fetch_wl_tip(spec.wl_key)
        
    ensemble = rehydrate_ensemble_by_ce_key(spec.ce_key)
    rng = np.random.default_rng(int(spec.seed))

    # Precompute sublattice site-index mapping for this WL key/spec
    sl_comp_map, sl_struct = _build_sublattices(ce_key=spec.ce_key)
    sublattice_indices = _build_sublattice_indices(ensemble=ensemble, sl_struct=sl_struct, sl_comp_map=sl_comp_map)

    nn_pairs_by_sublattice = _build_sublattice_nn_pairs(
        sl_struct=sl_struct,
        sublattice_indices=sublattice_indices,
        rtol=1e-3,
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
        "algo_version": spec.algo_version,
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
