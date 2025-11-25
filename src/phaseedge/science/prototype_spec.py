import dataclasses
from dataclasses import dataclass
import re
from enum import Enum
from typing import Iterable, Mapping, Any

from ase.atoms import Atoms
from ase.spacegroup import crystal
from monty.json import MSONable


class PrototypeStructure(str, Enum):
    ROCKSALT = "rocksalt"
    SPINEL = "spinel"
    SPINEL16c = "spinel16c"
    DOUBLE_PEROVSKITE = "doubleperovskite"
    PYROCHLORE = "pyrochlore"


# Matches tags like J0Sr, Q0O, etc.
_SEGMENT_RE = re.compile(
    r"^(?P<prefix>[A-Z])(?P<index>\d+)(?P<element>[A-Z][a-z]{0,2})$"
)


def parse_prototype(
    prototype: str,
    *,
    allowed_prefixes: Iterable[str] = ("J", "Q"),
) -> tuple[PrototypeStructure, dict[str, str]]:
    """
    Parse prototypes like 'doubleperovskite_J0Sr_Q0O' into:
        (PrototypeStructure.DOUBLE_PEROVSKITE, {'J0': 'Sr', 'Q0': 'O'})

    Rules:
    - First token is the structure type (must match PrototypeStructure.value).
    - Remaining tokens encode site assignments, e.g. 'J0Sr' -> {'J0': 'Sr'}.
    - Prefixes must be in allowed_prefixes.

    Raises:
        ValueError if malformed.
    """
    if not prototype:
        raise ValueError("Invalid prototype: empty string.")

    tokens = prototype.split("_")
    structure_token = tokens[0]

    structure_map: dict[str, PrototypeStructure] = {
        e.value: e for e in PrototypeStructure
    }
    try:
        structure = structure_map[structure_token]
    except KeyError as exc:
        valid = ", ".join(sorted(structure_map))
        raise ValueError(
            f"Unknown prototype structure '{structure_token}'. "
            f"Valid values: {valid}."
        ) from exc

    prefixes = set(allowed_prefixes)
    spec: dict[str, str] = {}
    seen_keys: set[str] = set()

    for seg in tokens[1:]:
        m = _SEGMENT_RE.match(seg)
        if not m:
            raise ValueError(
                f"Bad segment '{seg}'. Expected <PREFIX><INDEX><ELEMENT>, "
                "e.g. J0Sr or Q12O."
            )

        prefix = m.group("prefix")
        if prefix not in prefixes:
            allowed_str = ", ".join(sorted(prefixes))
            raise ValueError(
                f"Disallowed prefix '{prefix}' in segment '{seg}'. "
                f"Allowed: {allowed_str}."
            )

        index = m.group("index")
        element = m.group("element")
        key = f"{prefix}{index}"

        if key in seen_keys:
            raise ValueError(f"Duplicate key '{key}' encountered.")

        spec[key] = element
        seen_keys.add(key)

    return structure, spec


def _require_exact_keys(spec: dict[str, str], required: set[str]) -> None:
    """
    Ensure spec has exactly the required keys (no more, no less).
    """
    present = set(spec.keys())
    missing = required - present
    extra = present - required

    if missing:
        raise ValueError(f"Missing required tag(s): {sorted(missing)}")

    if extra:
        raise ValueError(
            f"Unexpected tag(s): {sorted(extra)}; "
            f"expected exactly {sorted(required)}"
        )


ParamValue = float | int | str


def _pop_float(local_params: dict[str, ParamValue], key: str) -> float:
    """
    Pop a required numeric param from local_params and return it as float.
    Raises ValueError if missing or not castable.
    """
    try:
        raw_val = local_params.pop(key)
    except KeyError as exc:
        raise ValueError(
            f"Missing required numeric param '{key}'."
        ) from exc

    try:
        val = float(raw_val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Param '{key}' must be castable to float, got {raw_val!r}."
        ) from exc

    return val


def _build_primitive_cell(
    structure: PrototypeStructure,
    spec: dict[str, str],
    params: dict[str, ParamValue],
) -> tuple[Atoms, set[str]]:
    """
    Internal helper used by PrototypeSpec to:
    - construct the primitive ASE cell
    - figure out which cation sublattices are "active"

    Contract:
    - This function is ALSO the validator for 'params'. Each branch:
        * pops what it needs out of a local copy of params
        * raises if required keys are missing or invalid
        * raises if unexpected keys remain afterward
    - That means PrototypeSpec.__post_init__ can just call us;
      if we're happy, the spec is valid.

    Returns:
        (primitive_cell, active_sublattices)
    """

    # Work on a copy so we don't mutate the original dict stored on the dataclass
    local_params: dict[str, ParamValue] = dict(params)

    if structure is PrototypeStructure.ROCKSALT:
        # Needs:
        #   a : lattice parameter
        _require_exact_keys(spec, {"Q0"})
        anion = spec["Q0"]

        a = _pop_float(local_params, "a")
        if a <= 0:
            raise ValueError(f"Param 'a' must be > 0, got {a}.")

        primitive_cell = crystal(
            symbols=["Es", anion],
            basis=[(0, 0, 0), (0, 0, 1 / 2)],
            spacegroup=225,  # Fm-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=False,
        )
        active_sublattices = {"Es"}

    elif structure is PrototypeStructure.SPINEL:
        # Needs:
        #   a : cubic lattice parameter
        # Optional:
        #   u : oxygen parameter (~0.36 default)
        _require_exact_keys(spec, {"Q0"})
        anion = spec["Q0"]

        a = _pop_float(local_params, "a")
        if a <= 0:
            raise ValueError(f"Param 'a' must be > 0, got {a}.")

        u = float(local_params.pop("u", 0.36))

        primitive_cell = crystal(
            symbols=["Es", "Fm", anion],
            basis=[
                (1 / 4, 3 / 4, 3 / 4),      # Es: tetra A (conventional 8a)
                (5 / 8, 3 / 8, 3 / 8),      # Fm: octa B (conventional 16d)
                (1 / 2 + u, u, u),          # anion (oxygen / halide)
            ],
            spacegroup=227,  # Fd-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )
        active_sublattices = {"Es", "Fm"}

    elif structure is PrototypeStructure.SPINEL16c:
        # Needs:
        #   a : cubic lattice parameter
        # Optional:
        #   u : oxygen parameter (~0.36 default)
        _require_exact_keys(spec, {"Q0"})
        anion = spec["Q0"]

        a = _pop_float(local_params, "a")
        if a <= 0:
            raise ValueError(f"Param 'a' must be > 0, got {a}.")

        u = float(local_params.pop("u", 0.36))

        primitive_cell = crystal(
            symbols=["Es", "Fm", "Md", anion],
            basis=[
                (1 / 4, 3 / 4, 3 / 4),      # Es: tetra A (8a)
                (5 / 8, 3 / 8, 3 / 8),      # Fm: octa B normal (16d)
                (1 / 8, 7 / 8, 7 / 8),      # Md: octa 16c sublattice
                (1 / 2 + u, u, u),          # anion
            ],
            spacegroup=227,  # Fd-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )
        active_sublattices = {"Es", "Fm", "Md"}

    elif structure is PrototypeStructure.DOUBLE_PEROVSKITE:
        # https://next-gen.materialsproject.org/materials/mp-1205594
        # Needs:
        #   a : cubic lattice parameter
        # Optional:
        #   u : anion displacement (~0.26 default)
        # Semantic:
        #   Es / Fm are active B-site sublattices
        #   inactive_cation (J0) is the fixed A-site
        _require_exact_keys(spec, {"J0", "Q0"})
        inactive_cation = spec["J0"]
        anion = spec["Q0"]

        a = _pop_float(local_params, "a")
        if a <= 0:
            raise ValueError(f"Param 'a' must be > 0, got {a}.")

        u = float(local_params.pop("u", 0.26))

        primitive_cell = crystal(
            symbols=["Es", "Fm", inactive_cation, anion],
            basis=[
                (0, 0, 0),              # Es (active)
                (0, 0, 1 / 2),          # Fm (active)
                (1 / 4, 3 / 4, 3 / 4),  # inactive A-site cation
                (u, 0, 0),              # anion
            ],
            spacegroup=225,  # Fm-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )
        active_sublattices = {"Es", "Fm"}

    elif structure is PrototypeStructure.PYROCHLORE:
        # https://next-gen.materialsproject.org/materials/mp-757233
        # In pyrochlores the 8a site is vacant, creating a 1/8 oxygen deficiency.
        # Needs:
        #   a : cubic lattice parameter
        # Optional:
        #   x : oxygen parameter (~0.713 default)
        _require_exact_keys(spec, {"Q0"})
        anion = spec["Q0"]

        a = _pop_float(local_params, "a")
        if a <= 0:
            raise ValueError(f"Param 'a' must be > 0, got {a}.")

        x = float(local_params.pop("x", 0.713))

        primitive_cell = crystal(
            symbols=["Es", "Fm", anion, anion],
            basis=[
                (1/8, 5/8, 1/8), # Es 16d
                (3/8, 7/8, 5/8), # Fm 16c
                (1/4, 3/4, 1/4), # anion 8b
                (1/2, 0, x),   # anion 48f
            ],
            spacegroup=227,  # Fd-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )
        active_sublattices = {"Es", "Fm"}

    else:
        raise ValueError(f"Unknown prototype structure: {structure}")

    # Anything left in local_params is an unexpected kwarg.
    if local_params:
        raise ValueError(
            "Unexpected extra params for prototype "
            f"{structure.value}: {sorted(local_params.keys())}"
        )

    return primitive_cell, active_sublattices


@dataclass(frozen=True, slots=True)
class PrototypeSpec(MSONable):
    """
    Immutable, serializable description of a prototype structure.

    Fields:
        prototype: e.g. "spinel_Q0O" or "doubleperovskite_J0Sr_Q0O"
        params:    dict of geometry parameters, e.g. {"a": 8.2, "u": 0.36}

    Derived (not stored, recomputed on demand):
        primitive_cell            -> ASE Atoms primitive cell
        active_sublattices        -> which placeholder species are 'active'
        active_sublattice_counts  -> site counts per active sublattice
    """
    _: dataclasses.KW_ONLY
    prototype: str
    params: dict[str, ParamValue]

    def __post_init__(self) -> None:
        # Validation is delegated entirely to _build_primitive_cell.
        # Just to make sure it works, we call it here.
        structure, spec = parse_prototype(self.prototype)
        _build_primitive_cell(structure, spec, self.params)

    def as_dict(self) -> dict[str, Any]:
        """
        monty.json-style dict. We do NOT embed ASE Atoms.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "prototype": self.prototype,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PrototypeSpec":
        """
        Inverse of as_dict. Matches CalcSpec's style.
        """
        return cls(
            prototype=str(d["prototype"]),
            params=dict(d["params"]),
        )

    @property
    def primitive_cell(self) -> Atoms:
        """
        Build and return the primitive cell (ASE Atoms).
        """
        structure, spec = parse_prototype(self.prototype)
        primitive_cell, _ = _build_primitive_cell(structure, spec, self.params)
        return primitive_cell

    @property
    def active_sublattices(self) -> set[str]:
        """
        Return which placeholder species ('Es', 'Fm', 'Md', ...) are
        considered "active" cation sublattices.
        """
        structure, spec = parse_prototype(self.prototype)
        _, active = _build_primitive_cell(structure, spec, self.params)
        return active

    @property
    def active_sublattice_counts(self) -> dict[str, int]:
        """
        Count how many sites exist for each active sublattice in this
        primitive cell.
        """
        chem_syms: list[str] = list(self.primitive_cell.get_chemical_symbols())
        return {
            sub: chem_syms.count(sub)
            for sub in self.active_sublattices
        }
