from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from jobflow import job, Flow
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase.atoms import Atoms
from atomate2.forcefields.jobs import ForceFieldRelaxMaker

from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.utils.keys import compute_set_id, fingerprint_conv_cell, rng_for_index, occ_key_for_atoms
from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.storage import store


class MaceStoredResult(TypedDict, total=False):
    set_id: str
    occ_key: str
    model: str
    relax_cell: bool
    dtype: str
    success: bool
    energy: float | None
    final_formula: str | None
    details: dict[str, Any]


def _mace_coll():
    return store.db_rw()["mace_relax"]


def _calc_key(set_id: str, occ_key: str, model: str, relax_cell: bool, dtype: str) -> dict[str, Any]:
    return {"set_id": set_id, "occ_key": occ_key, "model": model, "relax_cell": relax_cell, "dtype": dtype}


def lookup_mace_result(set_id: str, occ_key: str, *, model: str, relax_cell: bool, dtype: str) -> MaceStoredResult | None:
    return _mace_coll().find_one(_calc_key(set_id, occ_key, model, relax_cell, dtype))  # type: ignore[return-value]


def upsert_mace_result(doc: MaceStoredResult) -> None:
    key = _calc_key(doc["set_id"], doc["occ_key"], doc["model"], doc["relax_cell"], doc["dtype"])
    _mace_coll().update_one(key, {"$set": doc}, upsert=True)


@dataclass
class RandomConfigSpec:
    # inputs that define the snapshot set + which index to generate
    conv_cell: Atoms | None
    prototype: PrototypeName | None
    prototype_params: dict[str, Any] | None
    supercell_diag: tuple[int, int, int]
    replace_element: str
    composition: dict[str, float]
    seed: int
    index: int  # which snapshot index in the set to generate
    attempt: int = 0  # usually 0; bump only if collision forces a retry


@job
def make_random_config(spec: RandomConfigSpec) -> dict[str, Any]:
    """
    Deterministically generate ONE random configuration + metadata.
    """
    if (spec.conv_cell is None) == (spec.prototype is None):
        raise ValueError("Provide exactly one of conv_cell OR prototype(+params).")

    conv_cell = spec.conv_cell or make_prototype(spec.prototype, **(spec.prototype_params or {}))
    set_id = compute_set_id(
        conv_fingerprint=None if spec.prototype else fingerprint_conv_cell(conv_cell),
        prototype=(spec.prototype if spec.prototype else None),
        prototype_params=(spec.prototype_params if spec.prototype_params else None),
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        compositions=[spec.composition],
        seed=spec.seed,
        algo_version="randgen-2",
    )
    rng = rng_for_index(set_id, spec.index, spec.attempt)
    snapshot = make_one_snapshot(
        conv_cell=conv_cell,
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        composition=spec.composition,
        rng=rng,
    )
    occ_key = occ_key_for_atoms(snapshot)
    structure = AseAtomsAdaptor.get_structure(snapshot)  # pmg Structure
    return {"structure": structure, "set_id": set_id, "occ_key": occ_key}


@job
def store_mace_result(
    set_id: str,
    occ_key: str,
    model: str,
    relax_cell: bool,
    dtype: str,
    result: Any,
) -> MaceStoredResult:
    """
    Persist a minimal, queryable record from the FF relax result.
    We try a few common shapes for atomate2 FF outputs.
    """
    energy: float | None = None
    final_formula: str | None = None
    details: dict[str, Any] = {}

    # Try to extract energy + final structure robustly:
    # 1) result is a Structure with .energy
    if isinstance(result, Structure):
        energy = getattr(result, "energy", None)
        final_formula = result.composition.reduced_formula
        details["n_sites"] = len(result)
    # 2) dict-like with fields
    elif isinstance(result, dict):
        if "structure" in result and isinstance(result["structure"], Structure):
            s: Structure = result["structure"]
            energy = getattr(s, "energy", result.get("energy", None))
            final_formula = s.composition.reduced_formula
            details["n_sites"] = len(s)
        else:
            energy = result.get("energy", None)
            final_formula = result.get("final_formula", None)
            details["keys"] = list(result.keys())

    doc: MaceStoredResult = {
        "set_id": set_id,
        "occ_key": occ_key,
        "model": model,
        "relax_cell": relax_cell,
        "dtype": dtype,
        "success": True,
        "energy": energy,
        "final_formula": final_formula,
        "details": details,
    }
    upsert_mace_result(doc)
    return doc


def make_mace_relax_flow(
    *,
    # snapshot spec (deterministic)
    snapshot: RandomConfigSpec,
    # MACE/relax settings (be sure to keep in calc key!)
    model: str = "MACE-MPA-0",
    relax_cell: bool = True,
    dtype: str = "float64",
    # FireWorks routing
    category: str = "gpu",
) -> Flow:
    """
    Returns a Flow: [make_random_config] -> [ForceFieldRelax] -> [store_mace_result],
    with FW category injected on every Firework.
    """
    # Pre-submit idempotency check: skip building the FF relax if cached
    cached = lookup_mace_result(
        set_id=compute_set_id(
            conv_fingerprint=None if snapshot.prototype else fingerprint_conv_cell(
                snapshot.conv_cell or make_prototype(snapshot.prototype, **(snapshot.prototype_params or {}))
            ),
            prototype=(snapshot.prototype if snapshot.prototype else None),
            prototype_params=(snapshot.prototype_params if snapshot.prototype_params else None),
            supercell_diag=snapshot.supercell_diag,
            replace_element=snapshot.replace_element,
            compositions=[snapshot.composition],
            seed=snapshot.seed,
            algo_version="randgen-2",
        ),
        occ_key="__DEFER__",  # we don’t know occ_key until we generate; do a second lookup in the store step
        model=model, relax_cell=relax_cell, dtype=dtype,
    )
    # The above quick lookup can’t finish without occ_key; we’ll do the real skip in submit-time code (see script).
    # We still build the flow; the submitter will decide to submit or skip entirely.

    j_gen = make_random_config(snapshot)

    maker = ForceFieldRelaxMaker(
        force_field_name=model,
        relax_cell=relax_cell,
        # If you tried calculator_kwargs in your env to force float64, keep it:
        calculator_kwargs={"default_dtype": dtype},
        # you can pass relax_kwargs={"steps": 300, "fmax": 0.02} if needed
    )
    j_relax = maker.make(j_gen.output["structure"])

    j_store = store_mace_result(
        j_gen.output["set_id"],
        j_gen.output["occ_key"],
        model,
        relax_cell,
        dtype,
        result=j_relax.output,
    )

    flow = Flow([j_gen, j_relax, j_store], name="Random→MACE relax")

    # Inject FW category so only GPU workers pick it up
    for fw in flow_to_workflow(flow).fws:
        fw.spec["_category"] = category

    # Return the Jobflow Flow (the caller will convert to FW Workflow when submitting)
    return flow
