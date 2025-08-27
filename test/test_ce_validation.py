import pytest

from phaseedge.science.ce import CECompatibilityError, validate_ce_composition


def _dummy_ce() -> dict:
    return {"sampling": {"counts": {"A": 1, "B": 1}}}


def test_validate_ce_composition_success() -> None:
    ce = _dummy_ce()
    validate_ce_composition(ce, {"A": 0.5, "B": 0.5})


def test_validate_ce_composition_unknown_species() -> None:
    ce = _dummy_ce()
    with pytest.raises(CECompatibilityError):
        validate_ce_composition(ce, {"A": 0.5, "C": 0.5})


def test_validate_ce_composition_bad_sum() -> None:
    ce = _dummy_ce()
    with pytest.raises(CECompatibilityError):
        validate_ce_composition(ce, {"A": 0.2, "B": 0.2})