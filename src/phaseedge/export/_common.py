import dataclasses

from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.jobs.ensure_dopt_ce import ensure_dopt_ce
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.ensure_dopt_ce_spec import EnsureDoptCESpec
from phaseedge.schemas.mixture import Mixture

LAUNCHPAD_PATH = "/home/cbu/fw_config/my_launchpad.yaml"


def build_swap_mixtures(
    cation_a: str,
    cation_b: str,
    es_size: int,
    fm_size: int,
    K: int,
    seed: int,
) -> list[Mixture]:
    """Build mixtures that sweep cation_a/cation_b across Es and Fm sublattices.

    For k = 1 .. es_size-1:
        Es: cation_a^{es_size-k}  cation_b^k
        Fm: cation_a^k            cation_b^{fm_size-k}
    """
    mixtures: list[Mixture] = []
    for k in range(1, es_size):
        es_counts: dict[str, int] = {}
        fm_counts: dict[str, int] = {}

        if es_size - k > 0:
            es_counts[cation_a] = es_size - k
        es_counts[cation_b] = k

        fm_counts[cation_a] = k
        if fm_size - k > 0:
            fm_counts[cation_b] = fm_size - k

        cm = {"Es": es_counts, "Fm": fm_counts}
        mixtures.append(Mixture(composition_map=cm, K=K, seed=seed))
    return mixtures


def resolve_or_launch(
    spec: EnsureDoptCESpec,
    label: str,
    prototype: str,
    supercell_diag: tuple[int, int, int],
    *,
    launch: bool,
    min_partial_frac: float,
) -> str | None:
    """Shared lookup / launch tail used by all export functions.

    Parameters
    ----------
    spec : EnsureDoptCESpec
        The *full* (min_partial_frac=1.0) spec.
    label : str
        Human-readable label for print statements (e.g. "A=Sr Bp=Al Bpp=Ta").
    prototype : str
        Prototype string, used in the job name.
    supercell_diag : tuple[int, int, int]
        Supercell diagonal, used in the job name.
    launch : bool
        Whether to submit the workflow if no CE exists yet.
    min_partial_frac : float
        If < 1.0, also check / submit a partial-fraction variant.

    Returns
    -------
    str | None
        The CE key if found or launched, else None.
    """
    existing_ce = lookup_ce_by_key(spec.final_ce_key)
    partial_spec = dataclasses.replace(spec, min_partial_frac=min_partial_frac)
    existing_partial_ce = lookup_ce_by_key(partial_spec.final_ce_key)

    if not launch:
        if existing_ce:
            print(f"{label} ce_key={spec.final_ce_key}")
            return spec.final_ce_key
        if min_partial_frac < 1.0 and existing_partial_ce:
            print(f"{label} partial CE key={partial_spec.final_ce_key}")
            return partial_spec.final_ce_key
        return None

    if existing_ce:
        print("Final complete CE already exists, no workflow submitted.")
        print(f"{label} ce_key={spec.final_ce_key}")
        return spec.final_ce_key

    if min_partial_frac < 1.0 and existing_partial_ce:
        print("Final partial CE already exists, no workflow submitted.")
        print(f"{label} partial CE key={partial_spec.final_ce_key}")
        return partial_spec.final_ce_key

    if min_partial_frac < 1.0:
        spec = partial_spec

    j = ensure_dopt_ce(spec=spec)
    j.name = (
        f"ensure_dopt_ce::{prototype}::{supercell_diag}::{spec.calc_spec.calculator}"
    )
    j.update_metadata({"_category": spec.category})

    wf = flow_to_workflow(j)
    lp = LaunchPad.from_file(LAUNCHPAD_PATH)
    _wf_id = lp.add_wf(wf)
    print(f"{label} ce_key={spec.final_ce_key}")
    return spec.final_ce_key
