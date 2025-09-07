# select_d_optimal_basis.py

from typing import Any, Mapping, Sequence, TypedDict, Literal, cast, Optional

import hashlib
import math
import numpy as np
from jobflow.core.job import job
from smol.cofe import ClusterExpansion
from smol.moca.ensemble import Ensemble

from phaseedge.science.prototypes import PrototypeName
from phaseedge.science.random_configs import make_one_snapshot, validate_counts_for_sublattice
from phaseedge.storage.ce_store import lookup_ce_by_key


class Candidate(TypedDict):
    # Required keys (present on all candidates)
    source: Literal["endpoint", "wl"]
    occ: list[int]
    occ_hash: str
    # Required-but-optional-value fields (always present; can be None)
    wl_key: str | None
    checkpoint_hash: str | None
    bin: int | None
    counts: Mapping[str, int] | None  # endpoints carry counts; WL entries set None


def _occ_hash(occ: Sequence[int]) -> str:
    return hashlib.sha256(bytes(int(x) & 0xFF for x in occ)).hexdigest()


def _rehydrate_ensemble(ce_doc: Mapping[str, Any]) -> Ensemble:
    payload = cast(Mapping[str, Any], ce_doc["payload"])
    ce = ClusterExpansion.from_dict(dict(payload))
    system = cast(Mapping[str, Any], ce_doc["system"])
    sc = tuple(int(x) for x in cast(Sequence[int], system["supercell_diag"]))
    sc_matrix = np.diag(sc)
    return Ensemble.from_cluster_expansion(ce, supercell_matrix=sc_matrix)


def _occ_for_counts(
    *,
    ensemble: Ensemble,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    counts: Mapping[str, int],
) -> list[int]:
    from phaseedge.science.prototypes import make_prototype
    from pymatgen.io.ase import AseAtomsAdaptor

    conv = make_prototype(prototype, **dict(prototype_params))
    counts_clean = {str(k): int(v) for k, v in counts.items()}
    validate_counts_for_sublattice(
        conv_cell=conv,
        supercell_diag=supercell_diag,
        replace_element=replace_element,
        counts=counts_clean,
    )
    rng = np.random.default_rng(0)  # deterministic
    snap = make_one_snapshot(
        conv_cell=conv,
        supercell_diag=supercell_diag,
        replace_element=replace_element,
        counts=counts_clean,
        rng=rng,
    )
    from pymatgen.io.ase import AseAtomsAdaptor as _Adaptor
    struct = _Adaptor.get_structure(snap)  # type: ignore[arg-type]

    proc = ensemble.processor
    occ = proc.cluster_subspace.occupancy_from_structure(struct, encode=True)
    return [int(x) for x in np.asarray(occ, dtype=np.int32)]


def _corr_from_occ(ensemble: Ensemble, occ: Sequence[int]) -> np.ndarray:
    struct = ensemble.processor.structure_from_occupancy(np.asarray(occ, dtype=np.int32))
    vec = ensemble.processor.cluster_subspace.corr_from_structure(struct)
    return np.asarray(vec, dtype=float).ravel()


def _sig_from_counts(counts: Mapping[str, int]) -> str:
    return ",".join(f"{k}:{int(v)}" for k, v in sorted(counts.items()))


@job
def select_d_optimal_basis(
    *,
    ce_key: str,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    endpoints: Sequence[Mapping[str, int]],
    # each chain: {"wl_key": str, "checkpoint_hash": str, "samples": [{"bin": int, "occ": [...]}, ...]}
    chains: Sequence[Mapping[str, Any]],
    budget: int,
    ridge: float = 1e-10,
    # NEW: map every WL chain to its composition counts so we can group by composition
    wl_counts_map: Mapping[str, Mapping[str, int]] | None = None,
) -> Mapping[str, Any]:
    """
    Round-robin greedy D-opt selection by composition group.

    Seeds with the endpoint structures, then performs sweeps over composition
    groups; in each sweep we select at most one candidate per group, picking
    the candidate that maximizes Δ log det(XᵀX + ridge I), with deterministic
    tie-breaking by (bin, occ_hash). Uses Sherman–Morrison updates.
    """
    if budget <= 0:
        raise ValueError("budget must be positive.")

    ce_doc = lookup_ce_by_key(ce_key)
    if not ce_doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")
    ensemble = _rehydrate_ensemble(ce_doc)

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
            replace_element=replace_element,
            counts=ep,
        )
        candidates.append(
            Candidate(
                source="endpoint",
                occ=occ,
                occ_hash=_occ_hash(occ),
                wl_key=None,
                checkpoint_hash=None,
                bin=None,
                counts={str(k): int(v) for k, v in ep.items()},
            )
        )

    # WL candidates
    for ch in chains:
        wl_key = str(ch["wl_key"])
        ck_hash = str(ch["checkpoint_hash"])
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
                    checkpoint_hash=ck_hash,
                    bin=b,
                    counts=None,  # will be resolved via wl_counts_map
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
    # Feature cache
    # -------------------------
    feat_cache: dict[str, np.ndarray] = {}

    def feat(occ_hash: str, occ: Sequence[int]) -> np.ndarray:
        vec = feat_cache.get(occ_hash)
        if vec is None:
            vec = _corr_from_occ(ensemble, occ)
            feat_cache[occ_hash] = vec
        return vec

    some_vec = feat(pool[0]["occ_hash"], pool[0]["occ"])
    p = int(some_vec.size)

    # -------------------------
    # Grouping by composition signature
    # -------------------------
    def comp_sig_for_candidate(c: Candidate) -> str:
        if c["source"] == "endpoint":
            assert c["counts"] is not None
            return _sig_from_counts(c["counts"])
        # WL: resolve via the wl_counts_map
        if wl_counts_map is None:
            return "unknown"
        wk = cast(Optional[str], c["wl_key"])
        if wk is None or wk not in wl_counts_map:
            return "unknown"
        return _sig_from_counts(wl_counts_map[wk])

    groups: dict[str, list[int]] = {}
    for i, c in enumerate(pool):
        sig = comp_sig_for_candidate(c)
        groups.setdefault(sig, []).append(i)

    group_order = sorted(groups)  # deterministic round-robin

    # -------------------------
    # Initialize inverse info matrix; seed with endpoints only
    # -------------------------
    A_inv = np.eye(p, dtype=np.float64) / float(ridge)

    def sm_update(Ainv: np.ndarray, x: np.ndarray) -> np.ndarray:
        Ax = Ainv @ x
        denom = 1.0 + float(x.T @ Ax)
        return Ainv - np.outer(Ax, Ax) / denom

    chosen_indices: list[int] = []

    # Seed with all endpoints (exactly as requested)
    for i, c in enumerate(pool):
        if c["source"] == "endpoint":
            x = feat(c["occ_hash"], c["occ"])
            A_inv = sm_update(A_inv, x)
            chosen_indices.append(i)

    if budget < len(chosen_indices):
        raise ValueError(f"budget={budget} is smaller than endpoint seed size {len(chosen_indices)}.")

    # Mark remaining per-group candidate lists (exclude already chosen)
    chosen_set = set(chosen_indices)
    remaining_by_group: dict[str, list[int]] = {
        g: [i for i in idxs if i not in chosen_set] for g, idxs in groups.items()
    }

    # -------------------------
    # Round-robin greedy sweeps
    # -------------------------
    def gain_for(i: int) -> tuple[float, tuple[str, str]]:
        c = pool[i]
        x = feat(c["occ_hash"], c["occ"])
        lev = float(x.T @ (A_inv @ x))
        gain = math.log1p(lev)
        # tie-break: (bin asc, hash asc). Endpoints have bin=None → place last.
        b = c["bin"]
        tb = (str(b).zfill(12) if b is not None else "~~~~", c["occ_hash"])
        return gain, tb

    while len(chosen_indices) < budget:
        picked_this_sweep = False
        for g in group_order:
            if len(chosen_indices) >= budget:
                break
            cand_idx_list = remaining_by_group.get(g, [])
            if not cand_idx_list:
                continue

            # Evaluate gain over this group's remaining candidates
            best_i = -1
            best_gain = -math.inf
            best_tb = ("~", "~")
            for i in cand_idx_list:
                gain, tb = gain_for(i)
                if gain > best_gain or (abs(gain - best_gain) < 1e-15 and tb < best_tb):
                    best_gain = gain
                    best_tb = tb
                    best_i = i

            if best_i < 0:
                continue

            # Commit pick
            x = feat(pool[best_i]["occ_hash"], pool[best_i]["occ"])
            A_inv = sm_update(A_inv, x)
            chosen_indices.append(best_i)
            picked_this_sweep = True
            # remove from the group's remaining list
            cand_idx_list.remove(best_i)

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
