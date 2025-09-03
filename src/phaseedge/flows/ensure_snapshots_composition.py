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
def _gather_occ_keys(
    *,
    set_id: str,
    occ_keys: list[str],
    # purely for dependency; ensures this job runs after all relax jobs complete
    wait_for: list[Any] | None = None,
) -> GatheredSnapshots:
    """
    Barrier + pass-through: return the set_id and the ordered list of occ_keys.
    The 'wait_for' argument is unused except to enforce that all upstream relax
    jobs have finished before this runs.
    """
    if not isinstance(occ_keys, list) or not all(isinstance(x, str) for x in occ_keys):
        raise ValueError("gather_occ_keys: 'occ_keys' must be a list[str].")
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
    Ensure an exact, ordered set of snapshots (by `indices`) have ForceField relax results
    in the DB. Returns (flow, gather_job), where gather_job.output provides:
        - set_id
        - occ_keys (ordered to match `indices`)

    The final gather job consumes the relax-job outputs via 'wait_for' so downstream
    work won’t start until all relaxes are completed, while the occ_keys are taken
    directly from the deterministic generator outputs.
    """
    jobs: list[Job] = []
    relax_barrier_outputs: list[Any] = []
    ordered_occ_keys: list[str] = []

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
        j_gen.update_metadata({"_category": category})

        if first_set_id_ref is None:
            first_set_id_ref = j_gen.output["set_id"]

        # record the generator's occ_key in order
        ordered_occ_keys.append(j_gen.output["occ_key"])

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
        j_decide.update_metadata({"_category": category})

        jobs.extend([j_gen, j_decide])
        # NOTE: we don’t inspect the relax output; we only depend on it to finish
        relax_barrier_outputs.append(j_decide.output)

    assert first_set_id_ref is not None, "indices must be non-empty"

    j_gather = _gather_occ_keys(
        set_id=cast(str, first_set_id_ref),
        occ_keys=ordered_occ_keys,
        wait_for=relax_barrier_outputs,
    )
    j_gather.name = "gather_occ_keys"
    j_gather.update_metadata({"_category": category})

    flow = Flow([*jobs, j_gather], name="Ensure snapshots (barriered)")
    return flow, j_gather
