from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class WLSamplerSpec:
    """
    Canonical, counts-only WL spec.

    - No semi-grand; no chemical potentials; no fractional composition.
    - `composition_counts`: exact integer counts on the replaceable sublattice (REQUIRED).

    Notes:
      - `steps` is the number of WL steps to run for THIS chunk (runtime policy, not part of wl_key).
      - `samples_per_bin` is a runtime capture policy (0 disables capture).
    """
    ce_key: str
    bin_width: float
    steps: int
    composition_counts: Mapping[str, int]  # REQUIRED
    step_type: str = "swap"
    check_period: int = 5_000
    update_period: int = 1
    seed: int = 0
    samples_per_bin: int = 0
