from dataclasses import dataclass
from typing import Mapping, Any
from monty.json import MSONable


@dataclass(frozen=True, slots=True)
class SublatticeSpec(MSONable):
    """
    Per-sublattice replacement specification.

    replace_element: placeholder symbol in the prototype (e.g., "Mg" for Td in spinel)
    counts: mapping of element symbol -> integer count on this sublattice
            (must sum to the number of replaceable sites at build time)
    """
    replace_element: str
    counts: Mapping[str, int]

    # ---- MSONable ----
    def as_dict(self) -> dict[str, Any]:
        # Canonicalize: string keys, int values, sorted by key for determinism
        counts_dict = {str(k): int(v) for k, v in (self.counts or {}).items()}
        # (Optional) sanity: negative counts are never valid
        for elem, n in counts_dict.items():
            if n < 0:
                raise ValueError(f"Negative count for {elem}: {n}")

        counts_sorted = {k: counts_dict[k] for k in sorted(counts_dict)}
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "replace_element": self.replace_element,
            "counts": counts_sorted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SublatticeSpec":
        counts_in = d.get("counts") or {}
        counts = {str(k): int(v) for k, v in counts_in.items()}
        for elem, n in counts.items():
            if n < 0:
                raise ValueError(f"Negative count for {elem}: {n}")
        return cls(
            replace_element=str(d["replace_element"]),
            counts=counts,
        )
