from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast, Dict

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

from smol.cofe import ClusterExpansion, ClusterSubspace, StructureWrangler
from pymatgen.entries.computed_entries import ComputedStructureEntry

from ase.atoms import Atoms

from phaseedge.storage.ce_store import CEStats


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
    replace_elements: Sequence[str],
    allowed_species: Sequence[str],
) -> Structure:
    """
    Create the CE parent primitive on the prototype lattice where the
    replaceable sublattices (all sites whose symbol is in replace_elements)
    are declared as disordered with the given allowed cations.

    IMPORTANT:
      - Use a dict {Element: fraction} for the disordered site, NOT a Composition.
      - Do not include any species here unless they truly appear in the data.
      - All replaceable sublattices receive the SAME allowed set; if you need
        different sets per sublattice, we can extend this later with a per-subl spec.
    """
    if not replace_elements:
        raise ValueError("replace_elements must be a non-empty sequence of prototype placeholders.")
    if not allowed_species:
        raise ValueError("allowed_species must be non-empty.")

    prim_cfg = AseAtomsAdaptor.get_structure(conv_cell)  # type: ignore[arg-type]

    # Uniform prior over the allowed cations. The actual fractions do not encode
    # training composition; they just declare the site space.
    frac = 1.0 / float(len(allowed_species))
    disordered: Dict[Element, float] = {Element(el): frac for el in allowed_species}

    # Replace every placeholder symbol in replace_elements with this disordered site space.
    mapping = {Element(sym): disordered for sym in replace_elements}
    prim_cfg.replace_species(mapping)  # type: ignore[arg-type]

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
    subspace: ClusterSubspace,
    structures: Sequence[Structure],
    supercell_diag: tuple[int, int, int],
) -> tuple[StructureWrangler, NDArray[np.float64]]:
    """
    Build a StructureWrangler and feature matrix X for the given structures.

    Critical details:
      - Always pass the explicit supercell matrix so SMOL maps each entry
        to the subspace parent deterministically.
      - Do NOT reuse a site_mapping from another entry; mappings are entry-specific.
    """
    nx, ny, nz = map(int, supercell_diag)
    sc_mat = np.diag((nx, ny, nz))

    wrangler = StructureWrangler(subspace)
    for i, s in enumerate(structures):
        if not isinstance(s, Structure):
            raise TypeError(f"Expected pymatgen Structure at index {i}, got {type(s)!r}")
        entry = ComputedStructureEntry(structure=s, energy=0.0)
        wrangler.add_entry(
            entry,
            supercell_matrix=sc_mat,  # <--- explicit transform (key to fix)
            verbose=False,
        )

    X = wrangler.feature_matrix
    if X.size == 0 or X.shape[1] == 0:
        raise ValueError(
            "Feature matrix is empty (no clusters generated). "
            "Increase cutoffs or adjust the basis specification."
        )
    if X.shape[0] != len(structures):
        raise RuntimeError(f"Feature/target mismatch: X has {X.shape[0]} rows, structures={len(structures)}.")
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
