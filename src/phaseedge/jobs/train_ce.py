from typing import Any, Mapping

import numpy as np
from jobflow.core.job import job

from phaseedge.sampling.train_ce_driver import TrainOutput, run_train_ce
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.storage.cetrainref_dataset import Dataset

@job
def train_ce(
    *,
    # training data
    dataset_key: str,
    # prototype-only system identity (needed to build subspace)
    prototype_spec: PrototypeSpec,
    supercell_diag: tuple[int, int, int],
    sublattices: dict[str, tuple[str, ...]],
    # CE config
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any],
    # weighting
    weighting: Mapping[str, Any] | None = None,
    # CV config
    cv_seed: int | None = None,
) -> TrainOutput:
    # -------- basic validation --------
    dataset = Dataset.from_key(dataset_key)
    structures_pm = [ref.structure for ref in dataset.train_refs]
    n_prims = int(np.prod(supercell_diag))  # number of primitive/conventional cells in the supercell
    y_cell = [ref.lookup_energy() / float(n_prims) for ref in dataset.train_refs]
    return run_train_ce(
        structures_pm=structures_pm,
        y_cell=y_cell,
        prototype_spec=prototype_spec,
        supercell_diag=supercell_diag,
        sublattices=sublattices,
        basis_spec=basis_spec,
        regularization=regularization,
        weighting=weighting,
        cv_seed=cv_seed,
    )
