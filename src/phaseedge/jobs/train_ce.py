from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from jobflow.core.job import job
from pymatgen.core import Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from smol.cofe import ClusterExpansion, ClusterSubspace, StructureWrangler
from pymatgen.entries.computed_entries import ComputedStructureEntry

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.design_metrics import compute_design_metrics, MetricOptions, DesignMetrics

__all__ = ["train_ce"]


class CEStats(TypedDict):
    """
    Canonical fit statistics (per configurational site, i.e., per cation site).
    All fields are required; keep storage deterministic and branch-free.
    """
    n: int
    mae_per_site: float
    rmse_per_site: float
    max_abs_per_site: float


class TrainStats(TypedDict):
    in_sample: CEStats                 # per-site metrics
    five_fold_cv: CEStats              # per-site metrics
    by_composition: dict[str, Mapping[str, CEStats]]  # {"Fe:128,Mg:128": {"in_sample": ..., "five_fold_cv": ...}, ...}


class _TrainOutput(TypedDict, total=False):
    payload: Mapping[str, Any]
    stats: TrainStats
    design_metrics: DesignMetrics


@dataclass(slots=True)
class BasisSpec:
    """
    Minimal basis spec for CE.
    - cutoffs: per-body-order interaction cutoffs, e.g. {1: 100.0, 2: 10.0, 3: 8.0}
    - basis:   basis family name understood by smol (e.g. 'sinusoid')
    """
    cutoffs: Mapping[Any, Any]
    basis: str = "sinusoid"

    def __post_init__(self) -> None:
        # Normalize immediately for in-process correctness
        self.cutoffs = {int(k): float(v) for k, v in dict(self.cutoffs).items()}


@dataclass(slots=True)
class Regularization:
    """
    Regularization options for the linear solve.
    type: 'ols' | 'ridge' | 'lasso' | 'elasticnet'
    alpha: strength (ignored for 'ols')
    l1_ratio: only for elasticnet
    """
    type: str = "ols"
    alpha: float = 1e-6
    l1_ratio: float = 0.5


def build_disordered_primitive(
    *,
    conv_cell: Atoms,
    sublattices: dict[str, tuple[str, ...]],
) -> Structure:
    """
    Create the CE parent primitive on the prototype lattice where the replaceable
    sublattice is a disordered site with the allowed cations.
    """
    if not sublattices:
        raise ValueError("sublattices must be non-empty.")

    prim_cfg = AseAtomsAdaptor.get_structure(conv_cell)  # type: ignore[arg-type]

    for replace_element, allowed_species in sublattices.items():
        # Uniform prior over the allowed cations. The actual fractions do not encode
        # training composition; they just declare the site space.
        frac = 1.0 / float(len(allowed_species))
        disordered: dict[Element, float] = {Element(el): frac for el in allowed_species}

        # Replace the prototype cation with the disordered site space
        prim_cfg.replace_species({Element(replace_element): disordered}) # pyright: ignore[reportArgumentType]
    return prim_cfg


def build_subspace(
    *, primitive_cfg: Structure, basis_spec: BasisSpec
) -> ClusterSubspace:
    """
    Create a ClusterSubspace from a disordered primitive and cutoffs.
    """
    return ClusterSubspace.from_cutoffs(
        structure=primitive_cfg,
        cutoffs=dict(basis_spec.cutoffs),
        basis=basis_spec.basis,
    )


def featurize_structures(
    *,
    subspace,
    structures: Sequence[Structure],
    supercell_diag: tuple[int, int, int],
):
    """
    Build a StructureWrangler and feature matrix X for the given structures.
    Ensures we pass ComputedStructureEntry (not bare Structure) to smol.

    Returns
    -------
    wrangler : StructureWrangler
    X        : np.ndarray  (n_structures, n_features)
    """

    # supercell as an integer diagonal matrix
    nx, ny, nz = map(int, supercell_diag)
    sc_mat = np.diag((nx, ny, nz))

    wrangler = StructureWrangler(subspace)
    site_map = None  # keep first mapping to stabilize features across entries

    for s in structures:
        if not isinstance(s, Structure):
            raise TypeError(f"Expected pymatgen Structure, got {type(s)!r}")
        entry = ComputedStructureEntry(structure=s, energy=0.0)  # energy unused for X
        wrangler.add_entry(
            entry,
            supercell_matrix=sc_mat,
            site_mapping=site_map,   # reuse mapping after first add
            verbose=False,
        )
        if site_map is None and wrangler.entries:
            # Persist the site mapping discovered on the first entry
            site_map = wrangler.entries[-1].data.get("site_mapping")

    X = wrangler.feature_matrix
    if X.size == 0 or X.shape[1] == 0:
        raise ValueError(
            "Feature matrix is empty (no clusters generated). "
            "Increase cutoffs or adjust the basis specification."
        )
    return wrangler, X

def fit_linear_model(
    X: NDArray[np.float64], y: NDArray[np.float64], reg: Regularization
) -> NDArray[np.float64]:
    """
    Solve for ECIs under the requested regularization. Intercept is disabled by design.
    """
    t = reg.type.lower()
    if t == "ols":
        model = LinearRegression(fit_intercept=False)
    elif t == "ridge":
        model = Ridge(alpha=reg.alpha, fit_intercept=False)
    elif t == "lasso":
        model = Lasso(alpha=reg.alpha, fit_intercept=False, max_iter=10000)
    elif t == "elasticnet":
        model = ElasticNet(alpha=reg.alpha, l1_ratio=reg.l1_ratio, fit_intercept=False, max_iter=10000)
    else:
        raise ValueError(f"Unknown regularization type: {reg.type}")

    model.fit(X, y)
    # All sklearn variants provide .coef_
    return cast(NDArray[np.float64], model.coef_.astype(np.float64, copy=False))


def predict_from_features(X: NDArray[np.float64], coefs: NDArray[np.float64]) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], (X @ coefs).astype(np.float64, copy=False))


def compute_stats(y_true: Sequence[float], y_pred: Sequence[float]) -> CEStats:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Stats require non-empty equal-length arrays.")
    n = len(y_true)
    abs_err = [abs(a - b) for a, b in zip(y_true, y_pred)]
    mae = float(sum(abs_err) / n)
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mex = float(max(abs_err))
    return {"n": n, "mae_per_site": mae, "rmse_per_site": rmse, "max_abs_per_site": mex}


def assemble_ce(subspace: ClusterSubspace, coefs: NDArray[np.float64]) -> ClusterExpansion:
    return ClusterExpansion(subspace, coefs)



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
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    sublattices: dict[str, tuple[str, ...]],
) -> int:
    """
    Count the number of active sites in the **supercell** built
    by replicating the prototype conventional cell by supercell_diag.
    """
    conv_cell = make_prototype(prototype, **dict(prototype_params))
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
    prototype: PrototypeName,
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

    Using this intensive scale aligns with SMOL's MC normalization. All **stored stats**
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
    conv_cell: Atoms = make_prototype(prototype, **dict(prototype_params))
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
