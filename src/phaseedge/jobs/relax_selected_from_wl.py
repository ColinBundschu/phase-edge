from typing import Any, Mapping, Sequence, cast

import numpy as np
from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow
from smol.cofe import ClusterExpansion
from smol.moca.ensemble import Ensemble
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms

from phaseedge.storage.ce_store import lookup_ce_by_key
from phaseedge.jobs.decide_relax import check_or_schedule_relax
from phaseedge.schemas.mixture import canonical_counts, occ_key_for_atoms


def _counts_sig(counts: Mapping[str, int]) -> str:
    cc = canonical_counts(counts)
    return ",".join(f"{k}:{int(v)}" for k, v in cc.items())


def _rehydrate_ensemble(ce_doc: Mapping[str, Any]) -> Ensemble:
    payload = cast(Mapping[str, Any], ce_doc["payload"])
    ce = ClusterExpansion.from_dict(dict(payload))
    system = cast(Mapping[str, Any], ce_doc["system"])
    sc = tuple(int(x) for x in cast(Sequence[int], system["supercell_diag"]))
    sc_matrix = np.diag(sc)
    return Ensemble.from_cluster_expansion(ce, supercell_matrix=sc_matrix)


@job
def relax_selected_from_wl(
    *,
    ce_key: str,
    # From select_d_optimal_basis: list of {"source","wl_key","checkpoint_hash","bin","occ","occ_hash", ["counts" for endpoints]}
    selected: Sequence[Mapping[str, Any]],
    # Deterministic composition for each WL chain
    wl_counts_map: Mapping[str, Mapping[str, int]],
    model: str,
    relax_cell: bool,
    dtype: str,
    category: str = "gpu",
    set_id_prefix: str = "wlref",
) -> Mapping[str, Any] | Response:
    """
    Group selected occupancies by composition (WL & endpoints), map to structures, schedule
    relaxes in parallel, and return groups compatible with fetch_training_set_multi:
        groups = [{"set_id": str, "counts": dict[str,int], "occ_keys": [str, ...], "occs": [[int,...], ...]}, ...]
    """
    ce_doc = lookup_ce_by_key(ce_key)
    if not ce_doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")

    ensemble = _rehydrate_ensemble(ce_doc)

    relax_jobs: list[Job | Flow] = []
    # key -> {"set_id","counts","occ_keys","occs"}
    groups_map: dict[str, dict[str, Any]] = {}

    for rec in selected:
        src = str(rec.get("source", ""))
        if src == "wl":
            wl_key = str(rec["wl_key"])
            counts = wl_counts_map.get(wl_key)
            if counts is None:
                raise RuntimeError(f"relax_selected_from_wl: missing counts for wl_key={wl_key}")
        elif src == "endpoint":
            counts_raw = cast(Mapping[str, int], rec.get("counts", {}))
            if not counts_raw:
                raise RuntimeError("relax_selected_from_wl: endpoint record missing 'counts'.")
            counts = counts_raw
        else:
            # Future-proof: ignore unknown sources
            continue

        counts_canon = canonical_counts(counts)
        sig = _counts_sig(counts_canon)
        set_id = f"{set_id_prefix}::{ce_key}::{sig}"

        grp = groups_map.get(sig)
        if grp is None:
            grp = {"set_id": set_id, "counts": counts_canon, "occ_keys": [], "occs": []}
            groups_map[sig] = grp

        # Build structure from occupancy via CE processor
        occ_seq = cast(Sequence[int], rec["occ"])
        occ_arr = np.asarray([int(x) for x in occ_seq], dtype=np.int32)
        pmg_struct = ensemble.processor.structure_from_occupancy(occ_arr)

        # Compute a structure-based key (matches fetch_training_set_multi's scheme)
        atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
        occ_key = occ_key_for_atoms(cast(Atoms, atoms))

        # Schedule relax
        j_relax: Job = check_or_schedule_relax(
            set_id=set_id,
            occ_key=occ_key,
            structure=pmg_struct,
            model=model,
            relax_cell=relax_cell,
            dtype=dtype,
            category=category,
        )  # type: ignore[assignment]
        j_relax.name = f"relax[{sig}::{occ_key[:12]}]"
        j_relax.update_metadata({"_category": category})
        relax_jobs.append(j_relax)

        # Record keys and raw occupancies (for later exact reconstruction)
        grp["occ_keys"].append(occ_key)
        grp["occs"].append([int(x) for x in occ_seq])

    groups_out = list(groups_map.values())
    groups_out.sort(key=lambda g: _counts_sig(cast(Mapping[str, int], g["counts"])))

    subflow = Flow(relax_jobs, name="Relax selected (parallel)")
    return Response(replace=subflow, output={"groups": groups_out})
