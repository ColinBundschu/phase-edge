"""Validation helpers for cluster expansion compatibility."""

from __future__ import annotations

from typing import Mapping


class CECompatibilityError(ValueError):
    """Raised when a composition is incompatible with a CE model."""


def validate_ce_composition(ce_doc: Mapping[str, object], composition: Mapping[str, float]) -> None:
    """Ensure the provided composition can be evaluated by the CE model.

    Parameters
    ----------
    ce_doc
        Stored CE document containing at least ``sampling.counts``.
    composition
        Mapping of species to fractional amounts. Fractions must sum to one.
    """
    try:
        counts: Mapping[str, int] = ce_doc["sampling"]["counts"]  # type: ignore[index]
    except Exception as exc:  # pragma: no cover - defensive
        raise CECompatibilityError("CE document missing sampling counts") from exc

    allowed = set(counts.keys())
    requested = set(composition.keys())
    unknown = requested - allowed
    if unknown:
        raise CECompatibilityError(f"Composition contains species not in CE: {sorted(unknown)}")

    total = sum(float(v) for v in composition.values())
    if abs(total - 1.0) > 1e-6:
        raise CECompatibilityError(f"Composition fractions must sum to 1, got {total}")

    for sp, frac in composition.items():
        if float(frac) < 0:
            raise CECompatibilityError(f"Negative fraction for species {sp}: {frac}")
