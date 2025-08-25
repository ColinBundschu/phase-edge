from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from jobflow.core.flow import Flow
from jobflow.core.job import job, Job

from phaseedge.orchestration.jobs.random_config import RandomConfigSpec, make_random_config
from phaseedge.orchestration.jobs.decide_relax import check_or_schedule_relax

__all__ = ["make_ensure_snapshots_flow"]

@job
def _gather_occ_keys(set_id: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Barrier: depend on each relax/store result, then emit the ordered occ_keys.

    `results` are outputs of `check_or_schedule_relax`:
      - cache-hit: the stored doc directly
      - cache-miss: replaced by [relax -> store], so this ref waits for store
    """
    occ_keys = [cast(str, r["occ_key"]) for r in results]
    return {"set_id": set_id, "occ_keys": occ_keys}


def make_ensure_snapshots_flow(
    *,
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    counts: Mapping[str, int],
    seed: int,
    indices: Sequence[int],
    model: str,
    relax_cell: bool,
    dtype: str,
    category: str = "gpu",
) -> tuple[Flow, Job]:
    """
    Ensure an exact, ordered set of snapshots (by `indices`) have MACE relax results
    in the DB. Returns (flow, gather_job), where gather_job.output provides:
        - set_id
        - occ_keys (ordered to match `indices`)

    The final gather job *consumes* the per-index relax/store outputs so that any
    downstream job (e.g. fetch_training_set) waits until all relaxes are complete.
    """
    decide_outputs = []
    jobs: list[Job] = []

    first_set_id_ref = None

    for idx in indices:
        spec = RandomConfigSpec(
            prototype=prototype,
            prototype_params=dict(prototype_params),
            supercell_diag=tuple(supercell_diag),
            replace_element=replace_element,
            counts=dict(counts),
            seed=int(seed),
            index=int(idx),
        )

        j_gen = make_random_config(spec)
        j_gen.name = f"generate_random_config[{idx}]"
        j_gen.metadata = {**(j_gen.metadata or {}), "_category": category}

        if first_set_id_ref is None:
            first_set_id_ref = j_gen.output["set_id"]

        j_decide = check_or_schedule_relax(
            set_id=j_gen.output["set_id"],
            occ_key=j_gen.output["occ_key"],  # harmless input; the decision uses DB anyway
            structure=j_gen.output["structure"],
            model=model,
            relax_cell=relax_cell,
            dtype=dtype,
            category=category,
        )
        j_decide.name = f"check_or_schedule_relax[{idx}]"
        j_decide.metadata = {**(j_decide.metadata or {}), "_category": category}

        jobs.extend([j_gen, j_decide])
        decide_outputs.append(j_decide.output)  # barrier: reference each result doc

    assert first_set_id_ref is not None, "indices must be non-empty"

    j_gather = _gather_occ_keys(set_id=first_set_id_ref, results=decide_outputs)
    j_gather.name = "gather_occ_keys"
    j_gather.metadata = {**(j_gather.metadata or {}), "_category": category}

    flow = Flow([*jobs, j_gather], name="Ensure snapshots (barriered)")
    return flow, j_gather
