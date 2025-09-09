from typing import Any, Mapping, Sequence, TypedDict, cast

from jobflow.core.flow import Flow
from jobflow.core.job import job, Job

from phaseedge.science.prototypes import PrototypeName
from phaseedge.jobs.ensure_snapshots_composition import make_ensure_snapshots_composition_flow
from phaseedge.schemas.sublattice import SublatticeSpec
# Reuse canonicalization helper from keys to avoid duplication
from phaseedge.utils.keys import _canon_sublattice_specs

__all__ = ["make_ensure_snapshots_compositions"]


class SnapshotGroup(TypedDict):
    set_id: str
    occ_keys: list[str]
    # Canonicalized sublattices (same shape as keys._canon_sublattice_specs):
    # [{"replace": "<placeholder>", "counts": {elem: int, ...}}, ...]
    sublattices: list[dict[str, Any]]
    seed: int


class GatheredGroups(TypedDict):
    groups: list[SnapshotGroup]


class MixtureElement(TypedDict):
    """
    One mixture entry specifying:
      - sublattices: exact integer counts per sublattice (SublatticeSpec)
      - K: number of snapshots to generate (>= 1)
      - seed: RNG seed for this mixture entry (required; no NotRequired)
    """
    sublattices: Sequence[SublatticeSpec]
    K: int
    seed: int


@job
def _gather_groups(groups: Sequence[Any], meta: Sequence[MixtureElement]) -> GatheredGroups:
    """
    Validate and normalize inner-gather outputs + aligned meta into:
      {
        "groups": [
          {
            "set_id": str,
            "occ_keys": [str, ...],
            "sublattices": [{"replace": str, "counts": {str:int}}, ...],
            "seed": int
          },
          ...
        ]
      }

    Notes:
      - Reuses phaseedge.utils.keys._canon_sublattice_specs to canonicalize sublattices.
      - This job assumes MixtureElement typing; no guess-and-check on dict shapes.
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

        # Strict typing: sublattices is Sequence[SublatticeSpec] and seed is required
        subl_specs: Sequence[SublatticeSpec] = m["sublattices"]
        if not isinstance(subl_specs, Sequence) or len(subl_specs) == 0:
            bad_idxs.append(i)
            continue

        canon_subls = _canon_sublattice_specs(subl_specs)
        seed = int(m["seed"])

        if not isinstance(sid, str):
            bad_idxs.append(i)
            continue
        if not isinstance(oks, list) or not all(isinstance(x, str) for x in oks):
            bad_idxs.append(i)
            continue

        out.append(
            {
                "set_id": cast(str, sid),
                "occ_keys": cast(list[str], oks),
                "sublattices": canon_subls,
                "seed": seed,
            }
        )

    if bad_idxs:
        raise ValueError(
            "gather_groups: Missing or invalid inner results at indices "
            f"{bad_idxs}. Upstream ensure_snapshots flows must return "
            "{{'set_id': str, 'occ_keys': [str, ...]}}, and meta must provide MixtureElement."
        )

    return {"groups": out}


def make_ensure_snapshots_compositions(
    *,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    # Mixture: each element is a MixtureElement with SublatticeSpec list
    mixture: Sequence[MixtureElement],
    # Relax/engine identity
    model: str,
    relax_cell: bool,
    dtype: str,
    # Scheduling / defaults (kept for API stability; not used when seed is required)
    default_seed: int,
    category: str = "gpu",
) -> tuple[Flow, Job]:
    """
    For each mixture element, ensure K snapshots exist (barriered), then gather
    all groups into a single output:
        {"groups": [{"set_id": ..., "occ_keys": [...], "sublattices": [...], "seed": ...}, ...]}.

    Each mixture element must be:
        {
          "sublattices": Sequence[SublatticeSpec],
          "K": int (>=1),
          "seed": int
        }
    """
    if not mixture:
        raise ValueError("mixture must be a non-empty sequence.")

    subflows: list[Flow] = []
    inner_gathers: list[Job] = []
    # Canonicalized meta passed to the final gather job—kept minimal and deterministic
    canon_meta: list[MixtureElement] = []

    for mi, elem in enumerate(mixture):
        subl_specs = elem["sublattices"]
        if not isinstance(subl_specs, Sequence) or len(subl_specs) == 0:
            raise ValueError(f"mixture[{mi}] has empty or missing 'sublattices'.")

        K = int(elem["K"])
        if K <= 0:
            raise ValueError(f"mixture[{mi}] must specify K >= 1, got {K}.")

        seed = int(elem["seed"])
        indices = [int(i) for i in range(K)]

        flow_i, j_gather_i = make_ensure_snapshots_composition_flow(
            prototype=prototype,
            prototype_params=prototype_params,
            supercell_diag=supercell_diag,
            sublattices=list(subl_specs),
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

        # Pass through typed meta for the final gather
        canon_meta.append({"sublattices": list(subl_specs), "K": K, "seed": seed})

    # Final gather depends on all inner gathers by referencing their outputs
    j_multi = _gather_groups(groups=[jg.output for jg in inner_gathers], meta=canon_meta)
    j_multi.name = "gather_groups"
    j_multi.metadata = {**(j_multi.metadata or {}), "_category": category}

    flow_all = Flow([*subflows, j_multi], name="Ensure snapshots (mixture)")
    return flow_all, j_multi
