from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from monty.json import MSONable


def canonical_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    """CE-style canonicalization: sort keys and cast values to int (no zero/neg checks)."""
    return {str(k): int(v) for k, v in sorted(counts.items())}


def counts_sig(counts: Mapping[str, int]) -> str:
    """Stable 'El:cnt,El2:cnt2' signature with canonical ordering."""
    cc = canonical_counts(counts)
    return ",".join(f"{k}:{int(v)}" for k, v in cc.items())


def composition_counts_from_map(
    composition_map: Mapping[str, Mapping[str, int]]
) -> dict[str, int]:
    """
    Sum per-sublattice element counts into flat totals.

    Args:
        composition_map: e.g. {"Es": {"Fe": 10, "Mg": 98}, "B": {"Fe": 5, "Mg": 7}}

    Returns:
        {"Fe": 15, "Mg": 105} (keys sorted for determinism)

    Raises:
        TypeError: if any element key is not str.
        ValueError: if any count is negative.
    """
    totals: dict[str, int] = {}
    for counts in composition_map.values():
        for elem, n in counts.items():
            if not isinstance(elem, str):
                raise TypeError("Element keys in composition_map must be str.")
            ni = int(n)
            if ni < 0:
                raise ValueError(f"Negative count for '{elem}': {ni}")
            totals[elem] = totals.get(elem, 0) + ni

    # Canonicalize order
    return canonical_counts(totals)


def composition_counts_sig(
    composition_map: Mapping[str, Mapping[str, int]],
    *,
    outer_sep: str = ",",
    kv_sep: str = ":",
    wrap_inner: tuple[str, str] = ("{", "}"),
) -> str:
    """
    Stable signature for a nested composition_map like:
      {"Es": {"Fe": 108}, "B": {"Mg": 108}}  ->  "B:{Mg:108},Es:{Fe:108}"

    - Outer keys (sublattice labels/placeholders) are sorted.
    - Inner dict is serialized via `counts_sig` (which sorts elements).
    - No flattening; structure is preserved as "sublattice:inner_sig".
    """
    left, right = wrap_inner
    parts: list[str] = []
    for sublat in sorted(composition_map):
        inner = composition_map[sublat]
        inner_sig = counts_sig(inner)  # uses your canonical_counts + "El:cnt,..." ordering
        parts.append(f"{sublat}{kv_sep}{left}{inner_sig}{right}")
    return outer_sep.join(parts)


def sorted_composition_maps(
    composition_maps: Sequence[Mapping[str, Mapping[str, int]]],
) -> tuple[dict[str, dict[str, int]], ...]:
    """
    Given a sequence of composition_map-like dicts, return a tuple of
    normalized (canonicalized) dicts with sorted keys and int counts.
    """
    normalized_composition_maps = [
        {str(ok): canonical_counts(inner) for ok, inner in sorted(cm.items())}
        for cm in composition_maps
    ]
    return tuple(sorted(normalized_composition_maps, key=composition_counts_sig))

@dataclass(frozen=True)
class Mixture(MSONable):
    """
    Canonical, reusable mixture spec for 'composition' sources.

    Fields:
      - composition_map: dict[str, dict[str, int]]
      - K: int
      - seed: int (default 0)

    Invariants:
      - Outer keys sorted.
      - Inner maps canonicalized via canonical_counts (keys sorted, ints).
    """
    composition_map: dict[str, dict[str, int]]
    K: int
    seed: int = 0

    def __post_init__(self) -> None:
        comp_norm: dict[str, dict[str, int]] = {
            str(ok): canonical_counts(inner)
            for ok, inner in sorted(self.composition_map.items())
        }
        object.__setattr__(self, "composition_map", comp_norm)
        object.__setattr__(self, "K", int(self.K))
        object.__setattr__(self, "seed", int(self.seed))

    def sort_key(self) -> tuple[
        tuple[tuple[str, tuple[tuple[str, int], ...]], ...], int, int
    ]:
        comp_key = tuple(
            (ok, tuple(sorted(inner.items())))
            for ok, inner in self.composition_map.items()  # already outer-sorted
        )
        return (comp_key, self.seed, self.K)

    def __hash__(self) -> int:
        # Hash on the structural key so we remain hashable despite dict fields.
        return hash(self.sort_key())

    # ---- MSONable ----
    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "composition_map": self.composition_map,
            "K": self.K,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Mixture":
        return cls(
            composition_map={
                str(k): canonical_counts(v) for k, v in d["composition_map"].items()
            },
            K=int(d["K"]),
            seed=int(d.get("seed", 0)),
        )

def sublattices_from_mixtures(mixtures: Sequence[Mixture]) -> dict[str, tuple[str, ...]]:
    """
    Extract the sublattice dictionary from a list of Mixture objects.
    """
    sublattices: dict[str, set[str]] = {}
    for mix in mixtures:
        for site, comp in mix.composition_map.items():
            if site not in sublattices:
                sublattices[site] = set()
            sublattices[site].update(comp.keys())
    return {site: tuple(sorted(elements)) for site, elements in sublattices.items()}

