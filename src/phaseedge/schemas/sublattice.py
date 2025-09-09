from dataclasses import dataclass
from typing import Any, Mapping

from monty.json import MSONable


@dataclass(frozen=True, slots=True)
class SublatticeSpec(MSONable):
    """
    Canonical sublattice specification.

    - replace_element: the placeholder/species in the prototype to be replaced
      on this sublattice (e.g., "Mg" for spinel A or B sites depending on prototype).
    - counts: mapping of chemical symbols to integer counts within the *supercell*
      on this sublattice (e.g., {"Fe": 23, "Mg": 233}).
    """

    replace_element: str
    counts: dict[str, int]

    def __post_init__(self) -> None:
        # Basic validation w/o raising mysterious errors downstream.
        if not isinstance(self.replace_element, str) or not self.replace_element:
            raise ValueError("SublatticeSpec.replace_element must be a non-empty string.")
        if not isinstance(self.counts, dict) or not self.counts:
            raise ValueError("SublatticeSpec.counts must be a non-empty dict[str,int].")
        for k, v in self.counts.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"SublatticeSpec.counts has invalid key: {k!r}")
            try:
                iv = int(v)
            except Exception as exc:
                raise ValueError(f"SublatticeSpec.counts['{k}'] is not an int: {v!r}") from exc
            if iv < 0:
                raise ValueError(f"SublatticeSpec.counts['{k}'] is negative: {iv}")

    # ----- Monty/MSON API -----

    def as_dict(self) -> dict[str, Any]:  # type: ignore[override]
        # Always emit the canonical schema.
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "replace_element": self.replace_element,
            "counts": dict(self.counts),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SublatticeSpec":  # type: ignore[override]
        """
        Strict reader for the canonical schema.
        """
        # Monty wrappers can be present; ignore them.
        replace = d.get("replace_element")
        if replace is None:
            raise KeyError("replace_element")

        counts_raw = d.get("counts")
        if not isinstance(counts_raw, Mapping):
            raise KeyError("counts")

        # Normalize counts to dict[str,int]
        counts: dict[str, int] = {str(k): int(v) for k, v in counts_raw.items()}
        return cls(replace_element=str(replace), counts=counts)
