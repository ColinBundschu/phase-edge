from __future__ import annotations

from typing import Any, TypedDict
import json
import hashlib

from jobflow.core.flow import Flow
from jobflow.core.job import job
from jobflow.managers.fireworks import flow_to_workflow
from pymatgen.core import Structure
from atomate2.forcefields.jobs import ForceFieldRelaxMaker

from phaseedge.storage import store
from phaseedge.utils.keys import fingerprint_conv_cell
from phaseedge.science.prototypes import make_prototype
from phaseedge.orchestration.makers.random_config import (
    RandomConfigSpec,
    make_random_config,
)

# ---- helpers (counts-based set_id) -------------------------------------------------

def _hash_dict_stable(d: dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def compute_set_id_counts(
    *,
    conv_fingerprint: str | None,
    prototype: str | None,
    prototype_params: dict[str, Any] | None,
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    counts: dict[str, int],
    seed: int,
    algo_version: str = "randgen-2-counts-1",
) -> str:
    payload = {
        "algo": algo_version,
        "conv_fingerprint": conv_fingerprint,
        "prototype": prototype,
        "prototype_params": prototype_params,
        "supercell_diag": list(supercell_diag),
        "replace_element": replace_element,
        "counts": counts,
        "seed": seed,
    }
    return _hash_dict_stable(payload)

# ---- storage types -----------------------------------------------------------------

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

# ---- jobs --------------------------------------------------------------------------

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
    Robust to various Atomate2 return shapes.
    """
    energy: float | None = None
    final_formula: str | None = None
    details: dict[str, Any] = {}

    def _from_structure(s: Structure):
        e = getattr(s, "energy", None)
        return e, s.composition.reduced_formula, {"n_sites": len(s)}

    # 1) Direct Structure
    if isinstance(result, Structure):
        energy, final_formula, info = _from_structure(result)
        details.update(info)

    # 2) Mapping/dict (may contain 'structure' OR 'output.final_structure')
    elif isinstance(result, dict):
        out = result.get("output", {}) or {}
        s = out.get("final_structure") or result.get("final_structure") or result.get("structure")
        if isinstance(s, dict):
            try:
                s = Structure.from_dict(s)
            except Exception:
                s = None
        if isinstance(s, Structure):
            energy = out.get("final_energy") or out.get("energy") or getattr(s, "energy", None)
            final_formula = s.composition.reduced_formula
            details["n_sites"] = len(s)
        else:
            energy = out.get("final_energy") or out.get("energy") or result.get("energy")
            final_formula = result.get("final_formula")
            details["keys"] = list(result.keys())

    # 3) TaskDocument-like object (e.g., ForceFieldTaskDocument)
    else:
        out = getattr(result, "output", None)
        s_top = getattr(result, "structure", None)
        s_out = getattr(out, "final_structure", None) if out is not None else None
        s = s_out or s_top
        if isinstance(s, Structure):
            energy = (getattr(out, "final_energy", None) if out is not None else None) \
                     or (getattr(out, "energy", None) if out is not None else None) \
                     or getattr(s, "energy", None)
            final_formula = s.composition.reduced_formula
            details["n_sites"] = len(s)
        if (energy is None or final_formula is None) and hasattr(result, "as_dict"):
            try:
                rd = result.as_dict()
                outd = rd.get("output", {}) or {}
                sd = outd.get("final_structure") or rd.get("structure") or rd.get("final_structure")
                if isinstance(sd, dict):
                    try:
                        s = Structure.from_dict(sd)
                    except Exception:
                        s = None
                if isinstance(s, Structure):
                    energy = outd.get("final_energy") or outd.get("energy") or getattr(s, "energy", energy)
                    final_formula = s.composition.reduced_formula
                    details["n_sites"] = len(s)
                else:
                    energy = outd.get("final_energy") or outd.get("energy") or energy
                details.setdefault("result_type", rd.get("@class"))
            except Exception:
                pass
        details.setdefault("result_type", details.get("result_type", str(type(result))))

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

# ---- builder -----------------------------------------------------------------------

def make_mace_relax_workflow(
    *,
    snapshot: RandomConfigSpec,            # uses counts internally
    model: str = "MACE-MPA-0",
    relax_cell: bool = True,
    dtype: str = "float64",
    category: str = "gpu",
):
    """
    Build Flow [make_random_config] -> [ForceFieldRelax] -> [store_mace_result],
    convert once to FireWorks Workflow, and inject _category on every Firework.
    """
    # Advisory quick lookup (occ_key unknown here)
    _ = compute_set_id_counts(
        conv_fingerprint=None if snapshot.prototype else fingerprint_conv_cell(
            snapshot.conv_cell or make_prototype(snapshot.prototype, **(snapshot.prototype_params or {}))
        ),
        prototype=(snapshot.prototype if snapshot.prototype else None),
        prototype_params=(snapshot.prototype_params if snapshot.prototype_params else None),
        supercell_diag=snapshot.supercell_diag,
        replace_element=snapshot.replace_element,
        counts=snapshot.counts,
        seed=snapshot.seed,
    )

    j_gen = make_random_config(snapshot)
    j_gen.name = "generate_random_config"

    maker = ForceFieldRelaxMaker(
        force_field_name=model,
        relax_cell=relax_cell,
        calculator_kwargs={"default_dtype": dtype},
    )
    j_relax = maker.make(j_gen.output["structure"])
    j_relax.name = f"mace_relax[{model}]"

    j_store = store_mace_result(
        j_gen.output["set_id"],
        j_gen.output["occ_key"],
        model,
        relax_cell,
        dtype,
        result=j_relax.output,
    )
    j_store.name = "store_mace_result"

    flow = Flow([j_gen, j_relax, j_store], name="Random→MACE relax")
    wf = flow_to_workflow(flow)
    for fw in wf.fws:
        spec = dict(fw.spec or {})
        spec["_category"] = category
        fw.spec = spec
    return wf
