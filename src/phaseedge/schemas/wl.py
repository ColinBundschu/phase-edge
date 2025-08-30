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
