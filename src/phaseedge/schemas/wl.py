from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional, Literal
import numpy as np

EnsembleKind = Literal["canonical", "semi_grand"]

@dataclass(frozen=True)
class WLSamplerSpec:
    ce_key: str
    ensemble: EnsembleKind
    composition: Optional[Mapping[str, float]] = None      # for canonical
    chemical_potentials: Optional[Mapping[str, float]] = None  # for semi-grand
    bin_width: float = 0.01  # eV, supercell-scale
    steps: int = 1_000_000
    check_period: int = 5_000
    update_period: int = 1
    pilot_samples: int = 256       # internal; not exposed in the public API
    sigma_multiplier: float = 50.0 # “dirty hack”: ±50 σ
    seed: int = 0

@dataclass(frozen=True)
class WLResult:
    levels: np.ndarray            # [M]
    entropy: np.ndarray           # log g(E)
    histogram: np.ndarray         # counts
    visited_mask: np.ndarray      # bool mask of visited bins
    grid_anchor: float            # always 0.0
    bin_width: float
    window_used: tuple[float, float]
    meta: dict                    # ce_key, ensemble spec, seed, steps, …
