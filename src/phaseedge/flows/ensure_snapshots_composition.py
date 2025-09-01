from typing import Any, Mapping, Sequence, cast, TypedDict

from jobflow.core.flow import Flow
from jobflow.core.job import job, Job

from phaseedge.jobs.random_config import RandomConfigSpec, make_random_config
from phaseedge.jobs.decide_relax import check_or_schedule_relax
from phaseedge.science.prototypes import PrototypeName

__all__ = ["make_ensure_snapshots_composition_flow"]


class GatheredSnapshots(TypedDict):
    set_id: str
    occ_keys: list[str]


@job
def _gather_occ_keys(*, set_id: str, results: list[dict[str, Any] | None]) -> GatheredSnapshots:
    """
    Validate per-index results and gather their occ_keys into order, returning
    a payload that downstream jobs can consume (including the set_id that
    identifies this snapshot family).
    """
    bad_idxs: list[int] = []
    occ_keys: list[str] = []

    for i, r in enumerate(results):
        if not isinstance(r, dict) or "occ_key" not in r:
            bad_idxs.append(i)
            continue
        occ_keys.append(cast(str, r["occ_key"]))

    if bad_idxs:
        raise ValueError(
            "gather_occ_keys: Missing or invalid results at indices "
            f"{bad_idxs}. Upstream jobs likely returned None or a payload "
            "without 'occ_key'. Ensure decide_relax/store_mace_result always return "
            "{'occ_key': ..., ...}."
        )

    return {"set_id": set_id, "occ_keys": occ_keys}


def make_ensure_snapshots_composition_flow(
    *,
    prototype: PrototypeName,
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

    first_set_id_ref: Any | None = None

    for idx in indices:
        spec = RandomConfigSpec(
            prototype=prototype,
            prototype_params=dict(prototype_params),
            supercell_diag=supercell_diag,
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
            occ_key=j_gen.output["occ_key"],
            structure=j_gen.output["structure"],
            model=model,
            relax_cell=relax_cell,
            dtype=dtype,
            category=category,
        )
        j_decide.name = f"check_or_schedule_relax[{idx}]"
        j_decide.metadata = {**(j_decide.metadata or {}), "_category": category}

        jobs.extend([j_gen, j_decide])
        decide_outputs.append(j_decide.output)  # dependency barrier

    assert first_set_id_ref is not None, "indices must be non-empty"

    j_gather = _gather_occ_keys(set_id=cast(str, first_set_id_ref), results=decide_outputs)  # cast for Pylance
    j_gather.name = "gather_occ_keys"
    j_gather.metadata = {**(j_gather.metadata or {}), "_category": category}

    flow = Flow([*jobs, j_gather], name="Ensure snapshots (barriered)")
    return flow, j_gather
