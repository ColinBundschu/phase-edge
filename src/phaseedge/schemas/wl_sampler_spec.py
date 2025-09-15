from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping
from monty.json import MSONable


@dataclass(frozen=True, slots=True)
class WLSamplerSpec(MSONable):
    """
    Canonical, counts-only WL spec.

    - No semi-grand; no chemical potentials; no fractional composition.
    - `composition_counts`: exact integer counts across the *entire* WL supercell
      for the replaceable sublattice species (REQUIRED).
    - `sublattice_labels`: the placeholder symbols/labels (e.g., ["Es"] or ["A"])
      identifying which sublattice(s) are being replaced. These must exist in the
      prototype when constructing snapshots.

    Notes:
      - `steps` is the number of WL steps to run for THIS chunk (runtime policy,
        not part of wl_key).
      - `samples_per_bin` is a runtime capture policy (0 disables capture).
    """

    __version__: ClassVar[str] = "2"

    wl_key: str
    ce_key: str
    bin_width: float
    steps: int

    # NEW: which sublattice labels are active for WL sampling (immutable, canonical)
    sublattice_labels: tuple[str, ...]

    # Counts for the *entire* WL supercell (immutable mapping)
    composition_counts: Mapping[str, int]

    step_type: str = "swap"
    check_period: int = 5_000
    update_period: int = 1
    seed: int = 0
    samples_per_bin: int = 0

    # ----- validation & canonicalization -----
    def __post_init__(self) -> None:
        if self.bin_width <= 0:
            raise ValueError("bin_width must be > 0.")
        if self.steps <= 0:
            raise ValueError("steps must be a positive integer.")
        if self.check_period <= 0:
            raise ValueError("check_period must be a positive integer.")
        if self.update_period <= 0:
            raise ValueError("update_period must be a positive integer.")
        if self.samples_per_bin < 0:
            raise ValueError("samples_per_bin must be >= 0.")

        # --- sublattice_labels: coerce to canonical, unique, non-empty strings
        raw_labels: list[str] = [str(x) for x in self.sublattice_labels]
        if any(len(x) == 0 for x in raw_labels):
            raise ValueError("sublattice_labels may not contain empty strings.")
        # canonical: sorted unique tuple[str, ...]
        canon_labels = tuple(sorted(dict.fromkeys(raw_labels)))
        object.__setattr__(self, "sublattice_labels", canon_labels)

        # --- composition_counts: defensive copy; ints; no negatives; canonical sort
        raw_cc = dict(self.composition_counts)
        canon_cc: dict[str, int] = {}
        for k, v in raw_cc.items():
            if not isinstance(k, str):
                raise TypeError("composition_counts keys must be str.")
            iv = int(v)
            if iv < 0:
                raise ValueError(f"composition_counts['{k}'] must be >= 0.")
            canon_cc[k] = iv

        # sort keys for deterministic representation and freeze
        canon_cc = {k: canon_cc[k] for k in sorted(canon_cc)}
        object.__setattr__(self, "composition_counts", MappingProxyType(canon_cc))

    # ----- readable repr that shows sublattices and full counts as a plain dict -----
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return (
            f"{cls}("
            f"wl_key={self.wl_key!r}, ce_key={self.ce_key!r}, "
            f"bin_width={self.bin_width}, steps={self.steps}, "
            f"sublattice_labels={self.sublattice_labels!r}, "
            f"composition_counts={dict(self.composition_counts)!r}, "
            f"step_type={self.step_type!r}, check_period={self.check_period}, "
            f"update_period={self.update_period}, seed={self.seed}, "
            f"samples_per_bin={self.samples_per_bin})"
        )

    # ----- MSON API -----
    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "@version": self.__version__,
            "wl_key": self.wl_key,
            "ce_key": self.ce_key,
            "bin_width": self.bin_width,
            "steps": self.steps,
            "sublattice_labels": list(self.sublattice_labels),
            "composition_counts": dict(self.composition_counts),  # JSON-safe
            "step_type": self.step_type,
            "check_period": self.check_period,
            "update_period": self.update_period,
            "seed": self.seed,
            "samples_per_bin": self.samples_per_bin,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WLSamplerSpec":
        payload = {k: v for k, v in d.items() if not k.startswith("@")}

        # composition_counts: allow Mapping or list of [key, value] pairs
        cc_src = payload.get("composition_counts", {})
        if isinstance(cc_src, Mapping):
            cc_map: dict[str, int] = dict(cc_src)
        else:
            cc_map = dict(cc_src)

        # sublattice_labels: accept list/tuple/iterable; coerce to list[str]
        sl_src = payload.get("sublattice_labels", ())
        if isinstance(sl_src, (list, tuple)):
            sl_list = [str(x) for x in sl_src]
        else:
            # be forgiving: single string or other iterable
            sl_list = [str(x) for x in list(sl_src)] if sl_src else []

        return cls(
            wl_key=str(payload["wl_key"]),
            ce_key=str(payload["ce_key"]),
            bin_width=float(payload["bin_width"]),
            steps=int(payload["steps"]),
            sublattice_labels=tuple(sl_list),
            composition_counts=cc_map,
            step_type=str(payload.get("step_type", "swap")),
            check_period=int(payload.get("check_period", 5_000)),
            update_period=int(payload.get("update_period", 1)),
            seed=int(payload.get("seed", 0)),
            samples_per_bin=int(payload.get("samples_per_bin", 0)),
        )

    # ----- equality & hashing by value (labels + counts included) -----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WLSamplerSpec):
            return NotImplemented
        return (
            self.wl_key == other.wl_key
            and self.ce_key == other.ce_key
            and self.bin_width == other.bin_width
            and self.steps == other.steps
            and self.sublattice_labels == other.sublattice_labels
            and dict(self.composition_counts) == dict(other.composition_counts)
            and self.step_type == other.step_type
            and self.check_period == other.check_period
            and self.update_period == other.update_period
            and self.seed == other.seed
            and self.samples_per_bin == other.samples_per_bin
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.wl_key,
                self.ce_key,
                self.bin_width,
                self.steps,
                self.sublattice_labels,                 # tuple is hashable
                tuple(self.composition_counts.items()), # sorted in __post_init__
                self.step_type,
                self.check_period,
                self.update_period,
                self.seed,
                self.samples_per_bin,
            )
        )
