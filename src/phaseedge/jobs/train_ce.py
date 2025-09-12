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
from phaseedge.science.design_metrics import compute_design_metrics, MetricOptions, DesignMetrics
from phaseedge.storage.ce_store import CEStats

__all__ = ["train_ce"]


class TrainStats(TypedDict):
    in_sample: CEStats                 # per-site metrics
    five_fold_cv: CEStats              # per-site metrics
    by_composition: dict[str, Mapping[str, CEStats]]  # {"Fe:128,Mg:128": {"in_sample": ..., "five_fold_cv": ...}, ...}


class _TrainOutput(TypedDict, total=False):
    payload: Mapping[str, Any]
    stats: TrainStats
    design_metrics: DesignMetrics


def _ensure_structures(structures: Sequence[Structure | Mapping[str, Any]]) -> list[Structure]:
    out: list[Structure] = []
    for i, s in enumerate(structures):
        if isinstance(s, Structure):
            out.append(s)
        elif isinstance(s, Mapping):
            try:
                sd = cast(dict[str, Any], dict(s))
                out.append(Structure.from_dict(sd))
            except Exception as exc:
                raise ValueError(f"structures[{i}] could not be converted from dict to Structure") from exc
        else:
            raise TypeError(f"structures[{i}] has unsupported type: {type(s)!r}")
    return out


def _n_replace_sites_from_prototype(
    *,
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    sublattices: dict[str, tuple[str, ...]],
) -> int:
    """
    Count the number of active (replace_element) sites in the **supercell** built
    by replicating the prototype conventional cell by supercell_diag.
    """
    conv_cell: Atoms = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))
    n_per_prim = sum(1 for at in conv_cell if at.symbol in sublattices)
    if n_per_prim <= 0:
        raise ValueError(f"Prototype has no sites for sublattices='{sublattices}'.")
    nx, ny, nz = supercell_diag
    return int(n_per_prim) * nx * ny * nz


def _composition_signature(s: Structure) -> str:
    counts: dict[str, int] = {}
    for site in s.sites:
        sym = getattr(getattr(site, "specie", None), "symbol", None)
        if not isinstance(sym, str):
            sym = str(getattr(site, "specie", ""))
        if sym not in counts:
            counts[sym] = 0
        counts[sym] += 1
    parts = [f"{el}:{int(counts[el])}" for el in sorted(counts)]
    return ",".join(parts)


def _stats_for_group(
    idxs: Sequence[int],
    y_true_per_prim: Sequence[float],
    y_pred_per_prim: Sequence[float],
    *,
    sites_per_prim: int,
) -> CEStats:
    """
    Compute per-site stats for a subset `idxs`, given inputs in per-prim units.
    """
    if not idxs:
        return cast(CEStats, {"n": 0, "mae_per_site": 0.0, "rmse_per_site": 0.0, "max_abs_per_site": 0.0})
    scale = 1.0 / float(sites_per_prim)
    yt = [y_true_per_prim[i] * scale for i in idxs]
    yp = [y_pred_per_prim[i] * scale for i in idxs]
    return cast(CEStats, compute_stats(yt, yp))


def _build_sample_weights(
    comp_to_indices: Mapping[str, Sequence[int]],
    n_total: int,
    weighting: Mapping[str, Any] | None,
) -> np.ndarray:
    """
    Build per-sample weights. Scheme: inverse group count with exponent alpha,
    then normalize so mean weight = 1 across all samples.
      w_i = (1 / n_g) ** alpha ;  scale by c = N / sum(w)  → mean(w) = 1
    """
    w = np.ones(n_total, dtype=np.float64)
    if not weighting:
        return w

    scheme = str(weighting.get("scheme", "")).lower()
    alpha = float(weighting.get("alpha", 1.0))

    if scheme in ("balance_by_comp"):
        for idxs in comp_to_indices.values():
            n_g = max(1, len(idxs))
            base = (1.0 / float(n_g)) ** alpha
            for i in idxs:
                w[int(i)] = base
        s = float(w.sum())
        if s > 0:
            w *= (n_total / s)
        return w
    
    raise ValueError(f"Unknown weighting scheme: {scheme!r}")


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
    sublattices: dict[str, tuple[str, ...]],
    # CE config
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any],
    # weighting
    weighting: Mapping[str, Any] | None = None,
    # CV config
    cv_seed: int | None = None,
) -> _TrainOutput:
    """
    Train a Cluster Expansion with targets **normalized per primitive/conventional cell**:

        y_cell = E_total / (nx * ny * nz)

    Using this intensive scale aligns with SMOL’s MC normalization. All **stored stats**
    (in_sample, five_fold_cv, by_composition) are reported in **per-site** units to match
    common CE practice (meV/site), by scaling y vectors before computing metrics.
    """
    # -------- basic validation --------
    if not structures:
        raise ValueError("No structures provided.")
    if len(structures) != len(energies):
        raise ValueError(f"structures and energies length mismatch: {len(structures)} vs {len(energies)}")

    n_prims = int(np.prod(supercell_diag))  # number of primitive/conventional cells in the supercell

    # Convert any Monty-serialized dicts into real Structures
    structures_pm: list[Structure] = _ensure_structures(structures)

    # -------- 1) per-primitive/conventional-cell targets (training unit) --------
    y_cell = [float(E) / float(n_prims) for E in energies]

    # -------- 2) figure out sites_per_prim so we can report per-site metrics --------
    n_sites_const = _n_replace_sites_from_prototype(
        prototype=prototype,
        prototype_params=prototype_params,
        supercell_diag=supercell_diag,
        sublattices=sublattices,
    )
    if n_sites_const % n_prims != 0:
        raise ValueError(
            f"Active-site count ({n_sites_const}) is not divisible by number of prim cells ({n_prims})."
        )
    sites_per_prim = n_sites_const // n_prims  # e.g., 4 cation sites per conventional cell in rocksalt

    # -------- 3) prototype conv cell + allowed species inference --------
    conv_cell: Atoms = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))
    primitive_cfg = build_disordered_primitive(conv_cell=conv_cell, sublattices=sublattices)

    # -------- 4) subspace --------
    basis = BasisSpec(**cast(Mapping[str, Any], basis_spec))
    subspace = build_subspace(primitive_cfg=primitive_cfg, basis_spec=basis)

    # -------- 5) featurization (build once; re-used across CV folds) --------
    _, X = featurize_structures(subspace=subspace, structures=structures_pm, supercell_diag=supercell_diag)
    if X.size == 0 or X.shape[1] == 0:
        raise ValueError(
            "Feature matrix is empty (no clusters generated). "
            "Try increasing cutoffs or adjusting the basis specification."
        )
    if X.shape[0] != len(y_cell):
        raise RuntimeError(f"Feature/target mismatch: X has {X.shape[0]} rows, y has {len(y_cell)}.")

    # -------- 6) group membership by composition signature --------
    comp_to_indices: dict[str, list[int]] = {}
    for i, s in enumerate(structures_pm):
        sig = _composition_signature(s)
        comp_to_indices.setdefault(sig, []).append(i)

    # -------- 7) weights (per-sample) --------
    n = X.shape[0]
    y = np.asarray(y_cell, dtype=np.float64)  # per-prim
    w = _build_sample_weights(comp_to_indices, n_total=n, weighting=weighting)
    sqrt_w = np.sqrt(w, dtype=np.float64)

    # -------- 8) design diagnostics on the same matrix used for fitting --------
    design = compute_design_metrics(X=X, w=w, options=MetricOptions(standardize=True, eps=1e-12))

    # -------- 9) fit linear model on full set (in-sample, weighted) --------
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    reg = Regularization(**cast(Mapping[str, Any], regularization))
    coefs = fit_linear_model(Xw, yw, reg)

    # In-sample predictions in training unit (per-prim)
    y_pred_in = predict_from_features(X, coefs).tolist()

    # ------ Report per-site metrics (scale both true and pred by 1/sites_per_prim) ------
    scale = 1.0 / float(sites_per_prim)
    y_true_site_vec = [v * scale for v in y_cell]
    y_pred_site_vec = [v * scale for v in y_pred_in]
    stats_in = compute_stats(y_true_site_vec, y_pred_site_vec)

    # per-composition in-sample (per-site)
    by_comp_in: dict[str, CEStats] = {}
    for sig, idxs in comp_to_indices.items():
        by_comp_in[sig] = _stats_for_group(idxs, y_cell, y_pred_in, sites_per_prim=sites_per_prim)

    # -------- 10) 5-fold CV (stitched oof predictions; weighted fits) --------
    k_splits = min(5, n)
    if k_splits >= 2:
        kf = KFold(n_splits=k_splits, shuffle=True, random_state=int(cv_seed) if cv_seed is not None else None)
        y_pred_oof = np.empty(n, dtype=np.float64)
        y_pred_oof[:] = np.nan
        for train_idx, test_idx in kf.split(X):
            # weighted fit on training fold only
            sw_tr = sqrt_w[train_idx]
            X_tr = X[train_idx] * sw_tr[:, None]
            y_tr = y[train_idx] * sw_tr
            coefs_fold = fit_linear_model(X_tr, y_tr, reg)
            # predict on raw test features (unscaled)
            y_pred_oof[test_idx] = predict_from_features(X[test_idx], coefs_fold)
        if np.isnan(y_pred_oof).any():
            y_pred_oof = np.where(np.isnan(y_pred_oof), np.asarray(y_pred_in, dtype=np.float64), y_pred_oof)

        # Per-site CV stats
        stats_cv = compute_stats(
            [v * scale for v in y_cell],
            [float(v) * scale for v in y_pred_oof],
        )

        # per-composition CV (per-site)
        by_comp_cv: dict[str, CEStats] = {}
        y_oof_list = y_pred_oof.tolist()
        for sig, idxs in comp_to_indices.items():
            by_comp_cv[sig] = _stats_for_group(idxs, y_cell, y_oof_list, sites_per_prim=sites_per_prim)
    else:
        stats_cv = stats_in
        by_comp_cv = by_comp_in

    # -------- 11) assemble CE and payload --------
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
            "in_sample": cast(CEStats, stats_in),     # per-site
            "five_fold_cv": cast(CEStats, stats_cv),  # per-site
            "by_composition": {
                sig: {"in_sample": cast(CEStats, by_comp_in[sig]), "five_fold_cv": cast(CEStats, by_comp_cv[sig])}
                for sig in sorted(comp_to_indices)
            },
        },
        "design_metrics": design,
    }
