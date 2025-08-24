from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.orchestration.jobs.random_config import RandomConfigSpec, make_random_config
from phaseedge.orchestration.jobs.decide_relax import check_or_schedule_relax

def make_mace_relax_workflow(
    *,
    snapshot: RandomConfigSpec,
    model: str = "MACE-MPA-0",
    relax_cell: bool = True,
    dtype: str = "float64",
    category: str = "gpu",
):
    j_gen = make_random_config(snapshot)
    j_gen.name = "generate_random_config"
    j_gen.metadata = {**(j_gen.metadata or {}), "_category": category}

    j_decide = check_or_schedule_relax(
        set_id=j_gen.output["set_id"],
        occ_key=j_gen.output["occ_key"],
        structure=j_gen.output["structure"],
        model=model,
        relax_cell=relax_cell,
        dtype=dtype,
        category=category,
    )
    j_decide.name = "check_or_schedule_relax"
    j_decide.metadata = {**(j_decide.metadata or {}), "_category": category}

    flow = Flow([j_gen, j_decide], name="Random→MACE relax (idempotent)")
    wf = flow_to_workflow(flow)
    for fw in wf.fws:
        spec = dict(fw.spec or {})
        spec["_category"] = category
        fw.spec = spec
    return wf
