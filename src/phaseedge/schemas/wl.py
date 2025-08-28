from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

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

    # Internal (not part of the public idempotency contract):
    pilot_samples: int = 256
    sigma_multiplier: float = 50.0


@dataclass(frozen=True)
class WLResult:
    """
    Result of a WL run on a zero-anchored uniform enthalpy grid.
    """
    levels: np.ndarray          # 1D array of visited bin centers/levels (eV per supercell)
    entropy: np.ndarray         # log DOS on visited levels
    histogram: np.ndarray       # visit counts on visited levels
    visited_mask: np.ndarray    # bool mask (same length as `levels`)
    grid_anchor: float          # usually 0.0 in V1
    bin_width: float
    window_used: Tuple[float, float]  # (H_min, H_max) actually used
    meta: dict[str, Any]
