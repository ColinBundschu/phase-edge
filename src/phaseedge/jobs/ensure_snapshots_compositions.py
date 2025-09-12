from typing import Any, Mapping, Sequence, TypedDict
from jobflow.core.flow import Flow, JobOrder
from jobflow.core.job import Job, job

from phaseedge.schemas.mixture import Mixture
from phaseedge.science.prototypes import PrototypeName
from phaseedge.jobs.random_config import RandomConfigSpec, make_random_config
from phaseedge.jobs.decide_relax import check_or_schedule_relax


class SnapshotGroup(TypedDict):
    set_id: str
    occ_keys: list[str]
    composition_map: dict[str, dict[str, int]]
    seed: int


@job
def _emit_group(group: SnapshotGroup) -> SnapshotGroup:
    """
    Final barrier/emit: returns the group payload.
    Placed in an OUTER flow after the inner parallel flow so it only runs
    once all generate→relax jobs have completed.
    """
    return group


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
) -> Flow:
    """
    Ensure an exact, ordered set of snapshots (by `indices`) have relax results
    in the DB. Returns a Flow whose `output` is a SnapshotGroup.

    Structure:
      - INNER flow (AUTO order): per-index generate → relax in parallel.
      - OUTER flow (LINEAR): [INNER, _emit_group] so emit runs only after INNER completes.
    """
    inner_nodes: list[Flow | Job] = []
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
        ordered_occ_keys.append(j_gen.output["occ_key"])

        j_relax = check_or_schedule_relax(
            set_id=j_gen.output["set_id"],
            occ_key=j_gen.output["occ_key"],
            structure=j_gen.output["structure"],
            model=model,
            relax_cell=relax_cell,
            dtype=dtype,
            category=category,
        )
        j_relax.name = f"ff_check_or_schedule[{idx}]"
        j_relax.update_metadata({"_category": category})

        # Keep per-index parallelism: only depend relax on its generator
        inner_nodes.extend([j_gen, j_relax])

    assert first_set_id_ref is not None, "indices must be non-empty"

    # Inner parallel flow: lets all per-index pipelines run concurrently
    inner_parallel = Flow(inner_nodes, name="Ensure snapshots (parallel)")

    group_payload: SnapshotGroup = {
        "set_id": first_set_id_ref,
        "occ_keys": ordered_occ_keys,        # OutputReferences -> resolves to list[str]
        "composition_map": composition_map,  # already canonical upstream
        "seed": int(seed),
    }

    j_emit = _emit_group(group=group_payload)
    j_emit.name = "emit_group"
    j_emit.update_metadata({"_category": category})

    # OUTER flow: enforce barrier INNER -> EMIT via LINEAR order
    outer = Flow([inner_parallel, j_emit], name="Ensure snapshots (barriered)", order=JobOrder.LINEAR)
    outer.output = j_emit.output
    return outer


def make_ensure_snapshots_compositions(
    *,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    mixtures: Sequence[Mixture],
    model: str,
    relax_cell: bool,
    dtype: str,
    category: str = "gpu",
) -> Flow:
    """
    For each mixture, ensure K snapshots exist (barriered), then publish:
      {"groups": [SnapshotGroup, ...]} as the outer flow's output.
    """
    if not mixtures:
        raise ValueError("mixtures must be a non-empty sequence.")

    subflows: list[Job | Flow] = []
    for mi, mixture in enumerate(mixtures):
        flow_i = make_ensure_snapshots_composition_flow(
            prototype=prototype,
            prototype_params=prototype_params,
            supercell_diag=supercell_diag,
            composition_map=mixture.composition_map,
            seed=mixture.seed,
            indices=[int(i) for i in range(mixture.K)],
            model=model,
            relax_cell=relax_cell,
            dtype=dtype,
            category=category,
        )
        flow_i.name = f"Ensure snapshots (mix#{mi})"
        subflows.append(flow_i)

    flow_all = Flow(subflows, name="Ensure snapshots (mixtures)")
    flow_all.output = {"groups": [f.output for f in subflows]}
    return flow_all
