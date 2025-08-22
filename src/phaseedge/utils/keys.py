import hashlib
import json
from typing import Dict, List, Tuple, Any
import numpy as np
from numpy.random import default_rng, Generator
from ase.atoms import Atoms
from pymatgen.io.ase import AseAtomsAdaptor


def fingerprint_conv_cell(conv_cell: Atoms) -> str:
    pmg = AseAtomsAdaptor.get_structure(conv_cell)  # type: ignore[arg-type]
    payload = {
        "lattice": np.asarray(pmg.lattice.matrix).round(10).tolist(),
        "frac": np.asarray(pmg.frac_coords).round(10).tolist(),
        "species": [str(sp) for sp in pmg.species],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def compute_set_id(
    *,
    # identity always includes either conv_cell fingerprint or prototype+params
    conv_fingerprint: str | None,
    prototype: str | None,
    prototype_params: Dict[str, Any] | None,
    supercell_diag: Tuple[int, int, int],
    replace_element: str,
    compositions: List[Dict[str, float]],
    seed: int,
    algo_version: str = "randgen-2",
) -> str:
    """
    Deterministic identity for a logical (expandable) snapshot sequence.

    Exactly one of (conv_fingerprint) or (prototype, prototype_params) must be provided.
    """
    if (conv_fingerprint is None) == (prototype is None):
        raise ValueError("Provide exactly one of conv_fingerprint or prototype(+params).")

    payload = {
        "conv": conv_fingerprint,
        "prototype": prototype,
        "proto_params": prototype_params or {},
        "diag": supercell_diag,
        "replace": replace_element,
        "comps": compositions,
        "seed": seed,
        "algo": algo_version,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def seed_for(set_id: str, index: int, attempt: int = 0) -> int:
    h = hashlib.sha256(f"{set_id}:{index}:{attempt}".encode()).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def rng_for_index(set_id: str, index: int, attempt: int = 0) -> Generator:
    return default_rng(seed_for(set_id, index, attempt))


def occ_key_for_atoms(snapshot: Atoms) -> str:
    pmg = AseAtomsAdaptor.get_structure(snapshot)  # type: ignore[arg-type]
    payload = {
        "lattice": np.asarray(pmg.lattice.matrix).round(10).tolist(),
        "frac": np.asarray(pmg.frac_coords).round(10).tolist(),
        "species": [str(sp) for sp in pmg.species],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
