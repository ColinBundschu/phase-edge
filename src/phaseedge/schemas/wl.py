from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping
from monty.json import MSONable


@dataclass(frozen=True, slots=True)
class WLSamplerSpec(MSONable):
    """
    Canonical, counts-only WL spec.

    - No semi-grand; no chemical potentials; no fractional composition.
    - `composition_counts`: exact integer counts on the replaceable sublattice (REQUIRED).

    Notes:
      - `steps` is the number of WL steps to run for THIS chunk (runtime policy, not part of wl_key).
      - `samples_per_bin` is a runtime capture policy (0 disables capture).
    """

    __version__: ClassVar[str] = "1"

    wl_key: str
    ce_key: str
    bin_width: float
    steps: int
    composition_counts: Mapping[str, int]  # REQUIRED; shown in repr
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

        # Defensive copy; coerce to ints; no negatives; canonical sort
        raw = dict(self.composition_counts)
        canon: dict[str, int] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                raise TypeError("composition_counts keys must be str.")
            iv = int(v)
            if iv < 0:
                raise ValueError(f"composition_counts['{k}'] must be >= 0.")
            canon[k] = iv

        # sort keys for deterministic representation and freeze
        canon = {k: canon[k] for k in sorted(canon)}
        object.__setattr__(self, "composition_counts", MappingProxyType(canon))

    # ----- readable repr that shows the full counts as a plain dict -----
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return (
            f"{cls}("
            f"wl_key={self.wl_key!r}, ce_key={self.ce_key!r}, "
            f"bin_width={self.bin_width}, steps={self.steps}, "
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
            "composition_counts": dict(self.composition_counts),  # JSON‑safe
            "step_type": self.step_type,
            "check_period": self.check_period,
            "update_period": self.update_period,
            "seed": self.seed,
            "samples_per_bin": self.samples_per_bin,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WLSamplerSpec":
        payload = {k: v for k, v in d.items() if not k.startswith("@")}
        cc_src = payload.get("composition_counts", {})
        if isinstance(cc_src, Mapping):
            cc_map: dict[str, int] = dict(cc_src)
        else:
            # allow list of [key, value] pairs
            cc_map = dict(cc_src)

        return cls(
            wl_key=str(payload["wl_key"]),
            ce_key=str(payload["ce_key"]),
            bin_width=float(payload["bin_width"]),
            steps=int(payload["steps"]),
            composition_counts=cc_map,
            step_type=str(payload.get("step_type", "swap")),
            check_period=int(payload.get("check_period", 5_000)),
            update_period=int(payload.get("update_period", 1)),
            seed=int(payload.get("seed", 0)),
            samples_per_bin=int(payload.get("samples_per_bin", 0)),
        )

    # ----- equality & hashing by value (counts included) -----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WLSamplerSpec):
            return NotImplemented
        return (
            self.wl_key == other.wl_key
            and self.ce_key == other.ce_key
            and self.bin_width == other.bin_width
            and self.steps == other.steps
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
                tuple(self.composition_counts.items()),  # sorted in __post_init__
                self.step_type,
                self.check_period,
                self.update_period,
                self.seed,
                self.samples_per_bin,
            )
        )
