from typing import Any, Mapping, Sequence, TypedDict, Literal, cast

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
    struct = AseAtomsAdaptor.get_structure(snap)  # type: ignore[arg-type]

    proc = ensemble.processor
    occ = proc.cluster_subspace.occupancy_from_structure(struct, encode=True)
    return [int(x) for x in np.asarray(occ, dtype=np.int32)]


def _corr_from_occ(ensemble: Ensemble, occ: Sequence[int]) -> np.ndarray:
    struct = ensemble.processor.structure_from_occupancy(np.asarray(occ, dtype=np.int32))
    vec = ensemble.processor.cluster_subspace.corr_from_structure(struct)
    return np.asarray(vec, dtype=float).ravel()


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
) -> Mapping[str, Any]:
    if budget <= 0:
        raise ValueError("budget must be positive.")

    ce_doc = lookup_ce_by_key(ce_key)
    if not ce_doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")
    ensemble = _rehydrate_ensemble(ce_doc)

    candidates: list[Candidate] = []

    # Endpoints (forced in seed) — include counts for grouping later
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
                    counts=None,
                )
            )

    # Deduplicate by occ_hash
    uniq: dict[str, Candidate] = {}
    for c in candidates:
        uniq.setdefault(c["occ_hash"], c)
    pool: list[Candidate] = list(uniq.values())

    if not pool:
        raise ValueError("No candidate configurations found (check endpoints/chains inputs).")

    # Seed: endpoints + lowest-bin per chain
    seeds: list[Candidate] = [c for c in pool if c["source"] == "endpoint"]

    by_chain: dict[str, list[Candidate]] = {}
    for c in pool:
        if c["source"] != "wl":
            continue
        # safe: for WL candidates wl_key is not None
        by_chain.setdefault(cast(str, c["wl_key"]), []).append(c)

    for arr in by_chain.values():
        # All candidates in arr are WL → bin is not None
        arr.sort(key=lambda x: (cast(int, x["bin"]), x["occ_hash"]))
        if arr:
            seeds.append(arr[0])

    seed_size = len(seeds)
    total_candidates = len(pool)
    if budget < seed_size:
        raise ValueError(f"budget={budget} is smaller than seed size {seed_size}.")
    if budget > total_candidates:
        raise ValueError(f"budget={budget} exceeds total available {total_candidates}.")

    feat_cache: dict[str, np.ndarray] = {}

    def feat(occ_hash: str, occ: Sequence[int]) -> np.ndarray:
        vec = feat_cache.get(occ_hash)
        if vec is None:
            vec = _corr_from_occ(ensemble, occ)
            feat_cache[occ_hash] = vec
        return vec

    some_vec = feat(pool[0]["occ_hash"], pool[0]["occ"])
    p = int(some_vec.size)
    A_inv = np.eye(p) / float(ridge)

    def sm_update(Ainv: np.ndarray, x: np.ndarray) -> np.ndarray:
        Ax = Ainv @ x
        denom = 1.0 + float(x.T @ Ax)
        return Ainv - np.outer(Ax, Ax) / denom

    chosen: list[Candidate] = []
    for s in seeds:
        x = feat(s["occ_hash"], s["occ"])
        A_inv = sm_update(A_inv, x)
        chosen.append(s)

    chosen_hashes = {c["occ_hash"] for c in chosen}
    remaining: list[Candidate] = [c for c in pool if c["occ_hash"] not in chosen_hashes]

    need = budget - len(chosen)
    for _ in range(need):
        best_idx = -1
        best_gain = -math.inf
        best_key = ("~", "~")
        for i, c in enumerate(remaining):
            x = feat(c["occ_hash"], c["occ"])
            lev = float(x.T @ (A_inv @ x))
            gain = math.log1p(lev)
            # Prefer lower bins, then hash. For endpoints (bin=None), use a sentinel.
            bval = c["bin"]
            tie = (str(bval).zfill(12) if bval is not None else "~~~~", c["occ_hash"])
            if gain > best_gain or (abs(gain - best_gain) < 1e-15 and tie < best_key):
                best_idx = i
                best_gain = gain
                best_key = tie
        if best_idx < 0:
            break
        cstar = remaining.pop(best_idx)
        chosen.append(cstar)
        A_inv = sm_update(A_inv, feat(cstar["occ_hash"], cstar["occ"]))

    return {
        "chosen": chosen,  # endpoints include "counts"
        "seed_size": seed_size,
        "budget": budget,
        "design_summary": {"p": p, "n_candidates": total_candidates},
    }
