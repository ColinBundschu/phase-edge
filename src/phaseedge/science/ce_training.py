from dataclasses import dataclass
from typing import Mapping, Sequence, TypedDict, cast, Any

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, max_error

from smol.cofe import ClusterExpansion, ClusterSubspace, StructureWrangler
from pymatgen.entries.computed_entries import ComputedStructureEntry

from ase.atoms import Atoms


class FitStats(TypedDict):
    n: int
    mae_per_site: float
    rmse_per_site: float
    max_abs_per_site: float


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
    replace_element: str,
    allowed_species: Sequence[str],
) -> Structure:
    """
    Create the CE parent primitive on the prototype lattice where the replaceable
    sublattice is a disordered site with the allowed cations.

    IMPORTANT:
      - Use a dict {Element: fraction} for the disordered site, NOT a Composition.
      - Do not include `replace_element` unless it truly appears in data.
    """
    if not allowed_species:
        raise ValueError("allowed_species must be non-empty.")

    prim_cfg = AseAtomsAdaptor.get_structure(conv_cell)  # type: ignore[arg-type]

    # Uniform prior over the allowed cations. The actual fractions do not encode
    # training composition; they just declare the site space.
    frac = 1.0 / float(len(allowed_species))
    disordered: dict[Element, float] = {Element(el): frac for el in allowed_species}

    # Replace the prototype cation with the disordered site space
    prim_cfg.replace_species({Element(replace_element): disordered})
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


def compute_stats(y_true: Sequence[float], y_pred: Sequence[float]) -> FitStats:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Stats require non-empty equal-length arrays.")
    n = len(y_true)
    abs_err = [abs(a - b) for a, b in zip(y_true, y_pred)]
    mae = float(sum(abs_err) / n)
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mex = float(max_error(y_true, y_pred))
    return {"n": n, "mae_per_site": mae, "rmse_per_site": rmse, "max_abs_per_site": mex}


def assemble_ce(subspace: ClusterSubspace, coefs: NDArray[np.float64]) -> ClusterExpansion:
    return ClusterExpansion(subspace, coefs)
