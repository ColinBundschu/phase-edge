from typing import Any, Mapping, Sequence, TypedDict, Literal, cast

import hashlib
import numpy as np
from jobflow.core.job import job
from smol.moca.ensemble import Ensemble

from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.science.prototypes import make_prototype
from phaseedge.schemas.mixture import composition_map_sig
from phaseedge.jobs.store_ce_model import rehydrate_ensemble_by_ce_key


class Candidate(TypedDict):
    # Required keys (present on all candidates)
    source: Literal["endpoint", "wl"]
    occ: list[int]
    occ_hash: str
    # Required-but-optional-value fields (always present; can be None)
    wl_key: str | None
    wl_block_key: str | None
    bin: int | None
    composition_map: Mapping[str, Mapping[str, int]]


def _occ_hash(occ: Sequence[int]) -> str:
    return hashlib.sha256(bytes(int(x) & 0xFF for x in occ)).hexdigest()


def _occ_for_counts(
    *,
    ensemble: Ensemble,
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    composition_map: Mapping[str, Mapping[str, int]],
) -> list[int]:
    conv = make_prototype(prototype, **dict(prototype_params))
    rng = np.random.default_rng(0)  # deterministic
    snap = make_one_snapshot(
        conv_cell=conv,
        supercell_diag=supercell_diag,
        composition_map=composition_map,
        rng=rng,
    )
    from pymatgen.io.ase import AseAtomsAdaptor as _Adaptor
    struct = _Adaptor.get_structure(snap)  # type: ignore[arg-type]

    proc = ensemble.processor
    occ = proc.cluster_subspace.occupancy_from_structure(struct, encode=True)
    return [int(x) for x in np.asarray(occ, dtype=np.int32)]


def _corr_from_occ(ensemble: Ensemble, occ: Sequence[int]) -> np.ndarray:
    """
    Fast path: compute feature vector directly from encoded occupancy
    using the ensemble's processor (reuses cached site map/supercell).
    """
    vec = ensemble.processor.compute_feature_vector(np.asarray(occ, dtype=np.int32))
    return np.asarray(vec, dtype=float).ravel()


@job
def select_d_optimal_basis(
    *,
    ce_key: str,
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    endpoints: Sequence[Mapping[str, Mapping[str, int]]],
    wl_compoisition_maps: Mapping[str, Mapping[str, Mapping[str, int]]],
    # each chain: {"wl_key": str, "wl_block_key": str, "samples": [{"bin": int, "occ": [...]}, ...]}
    chains: Sequence[Mapping[str, Any]],
    budget: int,
    ridge: float = 1e-10,
) -> Mapping[str, Any]:
    """
    Round-robin greedy D-opt selection by composition group.

    Seeds with the endpoint structures, then performs sweeps over composition
    groups; in each sweep we select at most one candidate per group, picking
    the candidate that maximizes Δ log det(XᵀX + ridge I), with deterministic
    tie-breaking by (bin, occ_hash). Uses Sherman–Morrison updates.

    Optimizations:
      - Precompute all correlation vectors X (M×p) once using the processor's
        compute_feature_vector(occ), avoiding Structure construction entirely.
      - Vectorized gain evaluation per group: lev = diag(X A_inv Xᵀ).
    """
    if budget <= 0:
        raise ValueError("budget must be positive.")

    ensemble = rehydrate_ensemble_by_ce_key(ce_key)

    # -------------------------
    # Build and deduplicate pool
    # -------------------------
    candidates: list[Candidate] = []

    # Endpoints (forced seed) — include counts for grouping
    for ep in endpoints:
        occ = _occ_for_counts(
            ensemble=ensemble,
            prototype=prototype,
            prototype_params=prototype_params,
            supercell_diag=supercell_diag,
            composition_map=ep,
        )
        candidates.append(
            Candidate(
                source="endpoint",
                occ=occ,
                occ_hash=_occ_hash(occ),
                wl_key=None,
                wl_block_key=None,
                bin=None,
                composition_map=ep,
            )
        )

    # WL candidates
    for ch in chains:
        wl_key = str(ch["wl_key"])
        ck_hash = str(ch["wl_block_key"])
        samples = cast(Sequence[Mapping[str, Any]], ch.get("samples", []))
        for rec in samples:
            b = int(rec["bin"])
            occ = [int(x) for x in cast(Sequence[int], rec["occ"])]
            candidates.append(
                Candidate(
                    source="wl",
                    occ=occ,
                    occ_hash=_occ_hash(occ),
                    wl_key=wl_key,
                    wl_block_key=ck_hash,
                    bin=b,
                    composition_map=wl_compoisition_maps[wl_key],
                )
            )

    # Deduplicate by occ_hash
    uniq: dict[str, Candidate] = {}
    for c in candidates:
        uniq.setdefault(c["occ_hash"], c)
    pool: list[Candidate] = list(uniq.values())
    if not pool:
        raise ValueError("No candidate configurations found (check endpoints/chains inputs).")

    # -------------------------
    # Precompute feature matrix X (M×p) in memory
    # -------------------------
    first_vec = _corr_from_occ(ensemble, pool[0]["occ"])
    p = int(first_vec.size)
    M = len(pool)
    X = np.empty((M, p), dtype=np.float64)
    X[0, :] = first_vec
    for i in range(1, M):
        X[i, :] = _corr_from_occ(ensemble, pool[i]["occ"])

    # -------------------------
    # Group indices by composition signature
    # -------------------------
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(pool):
        sig = composition_map_sig(c["composition_map"])
        groups.setdefault(sig, []).append(i)
    group_order = sorted(groups)

    # -------------------------
    # Initialize inverse info matrix; seed with endpoints only
    # -------------------------
    A_inv = np.eye(p, dtype=np.float64) / float(ridge)

    def sm_update(Ainv: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Sherman–Morrison: (A + xxᵀ)^{-1} = A^{-1} - A^{-1} x xᵀ A^{-1} / (1 + xᵀ A^{-1} x)
        Ax = Ainv @ x
        denom = 1.0 + float(x.T @ Ax)
        return Ainv - np.outer(Ax, Ax) / denom

    chosen_indices: list[int] = []

    # Seed with all endpoints (exactly as requested)
    for i, c in enumerate(pool):
        if c["source"] == "endpoint":
            x = X[i, :]
            A_inv = sm_update(A_inv, x)
            chosen_indices.append(i)

    if budget < len(chosen_indices):
        raise ValueError(f"budget={budget} is smaller than endpoint seed size {len(chosen_indices)}.")

    chosen_set = set(chosen_indices)
    remaining_by_group: dict[str, list[int]] = {
        g: [i for i in idxs if i not in chosen_set] for g, idxs in groups.items()
    }

    # -------------------------
    # Vectorized leverage/gain computation helpers
    # -------------------------
    def gains_for_indices(indices: list[int]) -> np.ndarray:
        """
        Compute log(1 + leverage_i) for candidates indexed by `indices`
        in a single BLAS-backed shot: lev = diag(Xi @ A_inv @ Xiᵀ).
        """
        if not indices:
            return np.empty((0,), dtype=np.float64)
        Xi = X[indices, :]  # (k, p)
        Zi = Xi @ A_inv     # (k, p)
        lev = np.einsum("ij,ij->i", Xi, Zi, dtype=np.float64)  # rowwise dot
        return np.log1p(lev)

    # -------------------------
    # Round-robin greedy sweeps
    # -------------------------
    tol = 1e-15  # tie threshold for gains
    while len(chosen_indices) < budget:
        picked_this_sweep = False
        for g in group_order:
            if len(chosen_indices) >= budget:
                break
            cand_idx_list = remaining_by_group.get(g, [])
            if not cand_idx_list:
                continue

            gains = gains_for_indices(cand_idx_list)
            if gains.size == 0:
                continue

            # select best by gain; tie-break by (bin asc, occ_hash asc)
            max_gain = float(gains.max())
            mask = np.abs(gains - max_gain) <= tol
            tied_positions = np.nonzero(mask)[0].tolist()
            if len(tied_positions) == 1:
                pick_pos = tied_positions[0]
            else:
                best_tb = ("~~~~", "~")
                pick_pos = tied_positions[0]
                for pos in tied_positions:
                    idx = cand_idx_list[pos]
                    c = pool[idx]
                    b = c["bin"]
                    tb = (str(b).zfill(12) if b is not None else "~~~~", c["occ_hash"])
                    if tb < best_tb:
                        best_tb = tb
                        pick_pos = pos

            best_i = cand_idx_list[pick_pos]
            # Commit pick
            x = X[best_i, :]
            A_inv = sm_update(A_inv, x)
            chosen_indices.append(best_i)
            picked_this_sweep = True

            # remove from the group's remaining list
            del cand_idx_list[pick_pos]

            if len(chosen_indices) >= budget:
                break

        if not picked_this_sweep:
            # No group had candidates left → done
            break

    # Assemble chosen candidates in the order they were selected
    chosen: list[Candidate] = [pool[i] for i in chosen_indices]

    return {
        "chosen": chosen,  # endpoints include "counts"; WL get composition via wl_counts_map downstream
        "seed_size": sum(1 for i in chosen_indices if pool[i]["source"] == "endpoint"),
        "budget": budget,
        "design_summary": {"p": p, "n_candidates": len(pool), "n_groups": len(group_order)},
    }
