import re
from enum import Enum
from typing import Iterable
from ase.atoms import Atoms
from ase.spacegroup import crystal

class PrototypeStructure(str, Enum):
    ROCKSALT = "rocksalt"
    SPINEL = "spinel"
    DOUBLE_PEROVSKITE = "doubleperovskite"

_SEGMENT_RE = re.compile(r"^(?P<prefix>[A-Z])(?P<index>\d+)(?P<element>[A-Z][a-z]{0,2})$")

def parse_prototype(
    prototype: str,
    *,
    allowed_prefixes: Iterable[str] = ("J", "Q"),
) -> tuple[PrototypeStructure, dict[str, str]]:
    """
    Parse prototypes like 'doubleperovskite_J0Sr_Q0O' into:
      (PrototypeStructure.DOUBLE_PEROVSKITE, {'J0': 'Sr', 'Q0': 'O'})
    """
    if not prototype:
        raise ValueError("Invalid prototype: empty string.")

    tokens = prototype.split("_")
    structure_token = tokens[0]
    structure_map: dict[str, PrototypeStructure] = {e.value: e for e in PrototypeStructure}
    try:
        structure = structure_map[structure_token]
    except KeyError as exc:
        valid = ", ".join(sorted(structure_map))
        raise ValueError(
            f"Unknown prototype structure '{structure_token}'. Valid values: {valid}."
        ) from exc

    prefixes = set(allowed_prefixes)
    result: dict[str, str] = {}
    seen_keys: set[str] = set()

    for seg in tokens[1:]:
        m = _SEGMENT_RE.match(seg)
        if not m:
            raise ValueError(
                f"Bad segment '{seg}'. Expected <PREFIX><INDEX><ELEMENT>, e.g. J0Sr or Q12O."
            )
        prefix = m.group("prefix")
        if prefix not in prefixes:
            allowed_str = ", ".join(sorted(prefixes))
            raise ValueError(f"Disallowed prefix '{prefix}' in segment '{seg}'. Allowed: {allowed_str}.")
        index = m.group("index")
        element = m.group("element")
        key = f"{prefix}{index}"
        if key in seen_keys:
            raise ValueError(f"Duplicate key '{key}' encountered.")
        result[key] = element
        seen_keys.add(key)

    return structure, result

def _require_exact_keys(spec: dict[str, str], required: set[str]) -> None:
    """Ensure spec has exactly required keys (no more, no less)."""
    present = set(spec.keys())
    missing = required - present
    extra = present - required
    if missing:
        raise ValueError(f"Missing required tag(s): {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected tag(s): {sorted(extra)}; expected exactly {sorted(required)}")

def make_prototype(
    prototype: str,
    *,
    a: float,
) -> Atoms:
    """
    Build a primitive prototype cell from a prototype like:
      - 'rocksalt_Q0O'
      - 'spinel_Q0O'
      - 'doubleperovskite_J0Sr_Q0O'
    """
    if a <= 0:
        raise ValueError(f"Lattice parameter 'a' must be positive, got {a}.")

    structure, spec = parse_prototype(prototype)

    if structure is PrototypeStructure.ROCKSALT:
        _require_exact_keys(spec, {"Q0"})
        anion = spec["Q0"]
        return crystal(
            symbols=["Es", anion],
            basis=[(0, 0, 0), (0, 0, 1/2)],
            spacegroup=225, # Fm-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )

    if structure is PrototypeStructure.SPINEL:
        _require_exact_keys(spec, {"Q0"})
        anion = spec["Q0"]
        u = 0.36
        return crystal(
            symbols=["Es", "Fm", anion],
            basis=[(1/4, 3/4, 3/4), (5/8, 3/8, 3/8), (1/2 + u, u, u)],
            spacegroup=227, # Fd-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )

    if structure is PrototypeStructure.DOUBLE_PEROVSKITE:
        _require_exact_keys(spec, {"J0", "Q0"})
        inactive_cation = spec["J0"]
        anion = spec["Q0"]
        u = 0.26
        return crystal(
            symbols=["Es", "Fm", inactive_cation, anion],
            basis=[(0, 0, 0), (0, 0, 1/2), (1/4, 3/4, 3/4), (u, 0, 0)],
            spacegroup=225, # Fm-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )

    raise ValueError(f"Unknown prototype: {structure}")
