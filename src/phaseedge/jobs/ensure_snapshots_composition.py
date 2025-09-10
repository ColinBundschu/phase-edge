from typing import Any, Mapping, Sequence, cast, TypedDict

from jobflow.core.flow import Flow, JobOrder
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
) -> GatheredSnapshots:
    """
    Pass-through gather: return the set_id and the ordered list of occ_keys.

    We no longer use an explicit 'wait_for' arg to create a barrier. Instead,
    the enclosing Flow enforces a barrier between the snapshot/relax subflow
    and this gather job by using JobOrder.LINEAR at the outer level. That
    preserves parallelism inside the subflow and guarantees gather runs last.
    """
    if not isinstance(occ_keys, list) or not all(isinstance(x, str) for x in occ_keys):
        raise ValueError("gather_occ_keys: 'occ_keys' must be a list[str].")
    return {"set_id": set_id, "occ_keys": occ_keys}


def make_ensure_snapshots_composition_flow(
    *,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    composition_map: dict[str, dict[str, int]],
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

    Implementation detail:
      - We keep parallelism across (generate → relax) per index inside an inner subflow
        (default AUTO ordering), and enforce a barrier to the final gather step by
        wrapping the inner subflow and the gather job in an OUTER flow with
        JobOrder.LINEAR. This replaces the previous 'wait_for' trick.
    """
    inner_jobs: list[Job | Flow] = []
    ordered_occ_keys: list[Any] = []

    first_set_id_ref: Any | None = None

    for idx in indices:
        spec = RandomConfigSpec(
            prototype=prototype,
            prototype_params=dict(prototype_params),
            supercell_diag=supercell_diag,
            composition_map=composition_map,
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

        # Keep per-index parallelism: only depend j_decide on its j_gen references
        inner_jobs.extend([j_gen, j_decide])

    assert first_set_id_ref is not None, "indices must be non-empty"

    # Final gather does NOT create extra data dependencies; it just collects the
    # deterministic set_id/occ_keys from j_gen outputs. The barrier is achieved by
    # outer Flow(order=LINEAR) placing this after the inner subflow.
    j_gather = _gather_occ_keys(
        set_id=cast(str, first_set_id_ref),
        occ_keys=ordered_occ_keys,
    )
    j_gather.name = "gather_occ_keys"
    j_gather.update_metadata({"_category": category})

    # Inner subflow: generation + relax per index, AUTO order for parallelism.
    inner = Flow(inner_jobs, name="Ensure snapshots (parallel)")

    # Outer flow: enforce barrier "inner → gather" without serializing inner jobs.
    flow = Flow([inner, j_gather], name="Ensure snapshots (barriered)", order=JobOrder.LINEAR)

    return flow, j_gather
