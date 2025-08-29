from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class WLSamplerSpec:
    """
    Canonical, counts-only WL spec.

    - No semi-grand; no chemical potentials; no fractional composition.
    - `composition_counts`: exact integer counts on the replaceable sublattice (REQUIRED).
    """
    ce_key: str
    bin_width: float
    steps: int
    composition_counts: Mapping[str, int]  # REQUIRED
    step_type: str = "swap"
    check_period: int = 5_000
    update_period: int = 1
    seed: int = 0


@dataclass(frozen=True)
class WLResult:
    """
    Result of a WL run on a zero-anchored uniform enthalpy grid.
    """
    levels: np.ndarray          # 1D array of visited bin levels (eV per supercell)
    entropy: np.ndarray         # log DOS on visited levels
    histogram: np.ndarray       # visit counts on visited levels
    bin_indices: np.ndarray     # integer bin indices for visited levels
    grid_anchor: float          # usually 0.0 in V1
    bin_width: float
    meta: dict[str, Any]
