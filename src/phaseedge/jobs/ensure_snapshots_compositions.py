from typing import Any, Mapping, Sequence, TypedDict, cast

from jobflow.core.flow import Flow
from jobflow.core.job import job, Job

from phaseedge.science.prototypes import PrototypeName
from phaseedge.jobs.ensure_snapshots_composition import make_ensure_snapshots_composition_flow

__all__ = ["make_ensure_snapshots_compositions"]


class SnapshotGroup(TypedDict):
    set_id: str
    occ_keys: list[str]
    counts: dict[str, int]
    seed: int


class GatheredGroups(TypedDict):
    groups: list[SnapshotGroup]


@job
def _gather_groups(groups: Sequence[Any], meta: Sequence[Mapping[str, Any]]) -> GatheredGroups:
    """
    Validate and normalize inner-gather outputs + aligned meta into:
      {
        "groups": [
          {"set_id": str, "occ_keys": [str, ...], "counts": {str:int}, "seed": int},
          ...
        ]
      }
    """
    if len(groups) != len(meta):
        raise ValueError(f"gather_groups: groups/meta length mismatch: {len(groups)} vs {len(meta)}")

    out: list[SnapshotGroup] = []
    bad_idxs: list[int] = []

    for i, (g, m) in enumerate(zip(groups, meta)):
        if not isinstance(g, Mapping):
            bad_idxs.append(i)
            continue

        sid = g.get("set_id")
        oks = g.get("occ_keys")
        counts = {str(k): int(v) for k, v in dict(m.get("counts", {})).items()}
        seed = int(m.get("seed", 0))

        if not isinstance(sid, str):
            bad_idxs.append(i)
            continue
        if not isinstance(oks, list) or not all(isinstance(x, str) for x in oks):
            bad_idxs.append(i)
            continue
        if not counts:
            bad_idxs.append(i)
            continue

        out.append(
            {
                "set_id": cast(str, sid),
                "occ_keys": cast(list[str], oks),
                "counts": counts,
                "seed": seed,
            }
        )

    if bad_idxs:
        raise ValueError(
            "gather_groups: Missing or invalid inner results at indices "
            f"{bad_idxs}. Upstream ensure_snapshots flows must return "
            "{{'set_id': str, 'occ_keys': [str, ...]}}, and meta must provide counts/seed."
        )

    return {"groups": out}


def make_ensure_snapshots_compositions(
    *,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    # Mixture: each element has counts, K (num snapshots), and optional seed override
    mixture: Sequence[Mapping[str, Any]],
    # Relax/engine identity
    model: str,
    relax_cell: bool,
    dtype: str,
    # Scheduling / defaults
    default_seed: int,
    category: str = "gpu",
) -> tuple[Flow, Job]:
    """
    For each mixture element, ensure K snapshots exist (barriered), then gather
    all groups into a single output:
        {"groups": [{"set_id": ..., "occ_keys": [...], "counts": {...}, "seed": ...}, ...]}.

    Thin wrapper around make_ensure_snapshots_composition_flow (single-composition).
    """
    if not mixture:
        raise ValueError("mixture must be a non-empty sequence.")

    subflows: list[Flow] = []
    inner_gathers: list[Job] = []
    canon_mix: list[dict[str, Any]] = []

    for mi, elem in enumerate(mixture):
        counts = {str(k): int(v) for k, v in dict(elem.get("counts", {})).items()}
        if not counts:
            raise ValueError(f"mixture[{mi}] has empty or missing 'counts'.")

        K = int(elem.get("K", 0))
        if K <= 0:
            raise ValueError(f"mixture[{mi}] must specify K >= 1, got {K}.")

        seed = int(elem.get("seed", default_seed))
        indices = [int(i) for i in range(K)]

        flow_i, j_gather_i = make_ensure_snapshots_composition_flow(
            prototype=prototype,
            prototype_params=prototype_params,
            supercell_diag=supercell_diag,
            composition_map={replace_element: counts},
            seed=seed,
            indices=indices,
            model=model,
            relax_cell=relax_cell,
            dtype=dtype,
            category=category,
        )
        flow_i.name = f"Ensure snapshots (comp#{mi})"
        j_gather_i.name = f"gather_occ_keys[comp#{mi}]"

        subflows.append(flow_i)
        inner_gathers.append(j_gather_i)
        canon_mix.append({"counts": counts, "K": K, "seed": seed})

    # Final gather depends on all inner gathers by referencing their outputs
    j_multi = _gather_groups(groups=[jg.output for jg in inner_gathers], meta=canon_mix)
    j_multi.name = "gather_groups"
    j_multi.metadata = {**(j_multi.metadata or {}), "_category": category}

    flow_all = Flow([*subflows, j_multi], name="Ensure snapshots (mixture)")
    return flow_all, j_multi
