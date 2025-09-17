from phaseedge.jobs.decide_relax import lookup_ff_task


def lookup_total_energy_eV(
    *, set_id: str, occ_key: str, model: str, relax_cell: bool, dtype: str
) -> float:
    """
    Modern schema only: lookup_ff_task returns the inner 'output' document.
    Energy is doc['output']['energy'].
    """
    doc = lookup_ff_task(
        set_id=set_id, occ_key=occ_key, model=model, relax_cell=relax_cell, dtype=dtype
    )
    if doc is None:
        raise RuntimeError(f"No FF task output found for set_id={set_id} occ_key={occ_key}")
    try:
        return float(doc["output"]["energy"])
    except Exception as exc:
        raise RuntimeError(
            f"FF task output missing energy for set_id={set_id} occ_key={occ_key}"
        ) from exc

def fmt_mev(x: float | str) -> str:
    if isinstance(x, str):
        return x
    return f"{1e3 * float(x):.3f}"