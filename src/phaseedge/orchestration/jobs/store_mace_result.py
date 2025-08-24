from typing import Any, TypedDict

from jobflow.core.job import job
from pymatgen.core import Structure

from phaseedge.storage import store

__all__ = [
    "MaceStoredResult",
    "lookup_mace_result",
    "upsert_mace_result",
    "store_mace_result",
]


# ---- storage types & helpers -------------------------------------------------------


class MaceStoredResult(TypedDict):
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
    # RW handle; ensure indexes are created in phaseedge.storage.store
    return store.db_rw()["mace_relax"]


def _calc_key(set_id: str, occ_key: str, model: str, relax_cell: bool, dtype: str) -> dict[str, Any]:
    return {
        "set_id": set_id,
        "occ_key": occ_key,
        "model": model,
        "relax_cell": relax_cell,
        "dtype": dtype,
    }


def lookup_mace_result(
    set_id: str,
    occ_key: str,
    *,
    model: str,
    relax_cell: bool,
    dtype: str,
) -> MaceStoredResult | None:
    """Query for an existing relax result with the compound key."""
    return _mace_coll().find_one(_calc_key(set_id, occ_key, model, relax_cell, dtype))  # type: ignore[return-value]


def upsert_mace_result(doc: MaceStoredResult) -> None:
    """Idempotent write: insert or update the result document."""
    key = _calc_key(doc["set_id"], doc["occ_key"], doc["model"], doc["relax_cell"], doc["dtype"])
    _mace_coll().update_one(key, {"$set": doc}, upsert=True)


# ---- job ---------------------------------------------------------------------------


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
    Persist a minimal, queryable record from the ForceField relax result.
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
