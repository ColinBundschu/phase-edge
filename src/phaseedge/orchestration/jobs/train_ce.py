from typing import Any, Mapping, Sequence, TypedDict, cast

import numpy as np
from jobflow.core.job import job
from pymatgen.core import Structure
from ase.atoms import Atoms
from sklearn.model_selection import KFold

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.ce_training import (
    BasisSpec,
    Regularization,
    build_disordered_primitive,
    build_subspace,
    featurize_structures,
    fit_linear_model,
    predict_from_features,
    compute_stats,
    assemble_ce,
)
from phaseedge.storage.ce_store import CEStats

__all__ = ["train_ce"]


class TrainStats(TypedDict):
    in_sample: CEStats
    five_fold_cv: CEStats
    by_composition: dict[str, Mapping[str, CEStats]]  # {"Co:75,Mn:25": {"in_sample": ..., "five_fold_cv": ...}, ...}


class _TrainOutput(TypedDict):
    payload: Mapping[str, Any]
    stats: TrainStats


def _ensure_structures(structures: Sequence[Structure | Mapping[str, Any]]) -> list[Structure]:
    """
    Accepts a sequence of pymatgen Structures or dicts (Monty-serialized) and
    returns a list of Structures. Raises with a clear error if conversion fails.
    """
    out: list[Structure] = []
    for i, s in enumerate(structures):
        if isinstance(s, Structure):
            out.append(s)
        elif isinstance(s, Mapping):
            try:
                sd = cast(dict[str, Any], dict(s))  # concrete dict for Pylance
                out.append(Structure.from_dict(sd))
            except Exception as exc:
                raise ValueError(f"structures[{i}] could not be converted from dict to Structure") from exc
        else:
            raise TypeError(f"structures[{i}] has unsupported type: {type(s)!r}")
    return out


def _n_replace_sites_from_prototype(
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
) -> int:
    """
    Count replaceable sites per supercell deterministically from the prototype.
    This is invariant to relaxation and guarantees per-site normalization is stable.
    """
    conv_cell: Atoms = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))
    n_per_prim = sum(1 for at in conv_cell if at.symbol == replace_element)  # type: ignore[attr-defined]
    if n_per_prim <= 0:
        raise ValueError(
            f"Prototype has no sites for replace_element='{replace_element}'."
        )
    nx, ny, nz = supercell_diag
    return int(n_per_prim) * int(nx) * int(ny) * int(nz)


def _infer_allowed_species(
    conv_cell: Atoms, replace_element: str, structures: Sequence[Structure]
) -> list[str]:
    """
    Allowed cation species on the replaceable sublattice inferred from training data.
    - Exclude anions from the prototype (e.g., 'O' in rocksalt).
    - Do NOT force-include `replace_element`; include only species actually observed
      on that sublattice in the training data (e.g., 'Fe','Mn').
    """
    # Heuristic anion set: anything in the prototype that is NOT the cation label
    anion_symbols = {at.symbol for at in conv_cell if at.symbol != replace_element}  # type: ignore[attr-defined]

    seen: set[str] = set()
    for s in structures:
        for site in s.sites:
            sym = getattr(getattr(site, "specie", None), "symbol", None)
            if not isinstance(sym, str):
                sym = str(getattr(site, "specie", ""))
            if sym and sym not in anion_symbols:
                seen.add(sym)

    if not seen:
        raise ValueError(
            "Could not infer allowed species from training structures; got empty set "
            "(did energies/structures come from the expected binary cation sublattice?)."
        )
    return sorted(seen)


def _composition_signature(s: Structure, allowed_species: Sequence[str]) -> str:
    """
    Stable signature like 'Co:75,Mn:25' counting ONLY the allowed cations.
    Missing species are included as zero to keep keys comparable.
    """
    counts: dict[str, int] = {el: 0 for el in allowed_species}
    for site in s.sites:
        sym = getattr(getattr(site, "specie", None), "symbol", None)
        if not isinstance(sym, str):
            sym = str(getattr(site, "specie", ""))
        if sym in counts:
            counts[sym] += 1
    # sort by species name for stability
    parts = [f"{el}:{int(counts[el])}" for el in sorted(counts)]
    return ",".join(parts)


def _stats_for_group(idxs: Sequence[int], y_true: Sequence[float], y_pred: Sequence[float]) -> CEStats:
    if not idxs:
        # shouldn't happen; return zeros with n=0 to be safe
        return cast(CEStats, {"n": 0, "mae_per_site": 0.0, "rmse_per_site": 0.0, "max_abs_per_site": 0.0})
    yt = [y_true[i] for i in idxs]
    yp = [y_pred[i] for i in idxs]
    return cast(CEStats, compute_stats(yt, yp))


@job
def train_ce(
    *,
    # training data
    structures: Sequence[Structure | Mapping[str, Any]],
    energies: Sequence[float],  # total energies (eV) for the supercell
    # prototype-only system identity (needed to build subspace)
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    # CE config
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any],
    extra_hyperparams: Mapping[str, Any],
    # CV config
    cv_seed: int | None = None,
) -> _TrainOutput:
    """
    Train a per-site Cluster Expansion model and return:
      - overall in-sample stats
      - stitched 5-fold-CV stats
      - per-composition breakdown for both
    """
    # -------- basic validation --------
    if not structures:
        raise ValueError("No structures provided.")
    if len(structures) != len(energies):
        raise ValueError(f"structures and energies length mismatch: {len(structures)} vs {len(energies)}")

    try:
        nx, ny, nz = map(int, supercell_diag)
        supercell_diag = (nx, ny, nz)
    except Exception as exc:
        raise ValueError(f"supercell_diag must be a length-3 tuple of ints; got {supercell_diag!r}") from exc

    # Convert any Monty-serialized dicts into real Structures
    structures_pm: list[Structure] = _ensure_structures(structures)

    # -------- 1) per-site targets via constant site count from prototype --------
    n_sites = _n_replace_sites_from_prototype(
        prototype=prototype,
        prototype_params=prototype_params,
        supercell_diag=supercell_diag,
        replace_element=replace_element,
    )
    if n_sites <= 0:
        raise ValueError("Computed zero replaceable sites from prototype (unexpected).")
    y_site = [float(E) / float(n_sites) for E in energies]

    # -------- 2) prototype conv cell + allowed species inference --------
    conv_cell: Atoms = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))
    allowed_species = _infer_allowed_species(conv_cell, replace_element, structures_pm)
    primitive_cfg = build_disordered_primitive(
        conv_cell=conv_cell, replace_element=replace_element, allowed_species=allowed_species
    )

    # -------- 3) subspace --------
    basis = BasisSpec(**cast(Mapping[str, Any], basis_spec))
    subspace = build_subspace(primitive_cfg=primitive_cfg, basis_spec=basis)

    # -------- 4) featurization (build once; re-used across CV folds) --------
    wrangler, X = featurize_structures(
        subspace=subspace, structures=structures_pm, supercell_diag=supercell_diag
    )
    if X.size == 0 or X.shape[1] == 0:
        raise ValueError(
            "Feature matrix is empty (no clusters generated). "
            "Try increasing cutoffs or adjusting the basis specification."
        )
    if X.shape[0] != len(y_site):
        raise RuntimeError(f"Feature/target mismatch: X has {X.shape[0]} rows, y has {len(y_site)}.")

    # -------- group membership by composition signature --------
    comp_to_indices: dict[str, list[int]] = {}
    for i, s in enumerate(structures_pm):
        sig = _composition_signature(s, allowed_species)
        comp_to_indices.setdefault(sig, []).append(i)

    # -------- 5) fit linear model on full set (in-sample) --------
    y = np.asarray(y_site, dtype=np.float64)
    reg = Regularization(**cast(Mapping[str, Any], regularization))
    coefs = fit_linear_model(X, y, reg)

    # in-sample stats (overall)
    y_pred_in = predict_from_features(X, coefs).tolist()
    stats_in = compute_stats(y_site, y_pred_in)

    # per-composition in-sample
    by_comp_in: dict[str, CEStats] = {}
    for sig, idxs in comp_to_indices.items():
        by_comp_in[sig] = _stats_for_group(idxs, y_site, y_pred_in)

    # -------- 6) 5-fold CV (stitched oof predictions) --------
    n = X.shape[0]
    k_splits = min(5, n)
    if k_splits >= 2:
        kf = KFold(n_splits=k_splits, shuffle=True, random_state=int(cv_seed) if cv_seed is not None else None)
        y_pred_oof = np.empty(n, dtype=np.float64)
        y_pred_oof[:] = np.nan
        for train_idx, test_idx in kf.split(X):
            coefs_fold = fit_linear_model(X[train_idx], y[train_idx], reg)
            y_pred_oof[test_idx] = predict_from_features(X[test_idx], coefs_fold)
        # If any row was never tested (shouldn't happen), fall back to in-sample for those rows
        if np.isnan(y_pred_oof).any():
            y_pred_oof = np.where(np.isnan(y_pred_oof), np.asarray(y_pred_in, dtype=np.float64), y_pred_oof)
        stats_cv = compute_stats(y_site, y_pred_oof.tolist())
        # per-composition CV
        by_comp_cv: dict[str, CEStats] = {}
        y_oof_list = y_pred_oof.tolist()
        for sig, idxs in comp_to_indices.items():
            by_comp_cv[sig] = _stats_for_group(idxs, y_site, y_oof_list)
    else:
        # Not enough samples for CV; mirror in-sample
        stats_cv = stats_in
        by_comp_cv = by_comp_in

    # -------- 7) assemble CE and payload --------
    ce = assemble_ce(subspace, coefs)
    if hasattr(ce, "as_dict"):
        try:
            payload = cast(Mapping[str, Any], ce.as_dict())  # type: ignore[call-arg]
        except Exception:
            payload = {"repr": repr(ce)}
    else:
        payload = {"repr": repr(ce)}

    return {
        "payload": payload,
        "stats": {
            "in_sample": cast(CEStats, stats_in),
            "five_fold_cv": cast(CEStats, stats_cv),
            "by_composition": {
                sig: {"in_sample": cast(CEStats, by_comp_in[sig]), "five_fold_cv": cast(CEStats, by_comp_cv[sig])}
                for sig in sorted(comp_to_indices)
            },
        },
    }
