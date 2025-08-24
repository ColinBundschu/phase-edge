from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from ase.atoms import Atoms

def make_one_snapshot(
    *,
    conv_cell: Atoms,
    supercell_diag: Tuple[int, int, int],
    replace_element: str,
    counts: Dict[str, int],
    rng: np.random.Generator,
) -> Atoms:
    """
    Build a supercell, then replace exactly N_i sites on the target sublattice
    according to integer counts. Deterministic under the provided RNG.

    - conv_cell: primitive/conv cell ASE Atoms
    - supercell_diag: (nx, ny, nz)
    - replace_element: species in conv_cell whose sites are replaceable (e.g., "Mg")
    - counts: e.g. {"Co": 76, "Fe": 32} (MUST sum to number of replaceable sites in the supercell)
    """
    sc = conv_cell.repeat(supercell_diag)

    symbols = np.array(sc.get_chemical_symbols())
    # indices on the replaceable sublattice
    target_idx = np.where(symbols == replace_element)[0]
    n_sites = int(target_idx.size)

    total = sum(int(v) for v in counts.values())
    if total != n_sites:
        raise ValueError(
            f"Counts must sum to replacement sublattice size: got {total}, expected {n_sites}"
        )

    # deterministic assignment: permute indices, then slice by exact counts
    perm = rng.permutation(n_sites)
    start = 0
    for elem, n in sorted(counts.items()):  # sort for stable order
        n = int(n)
        if n < 0:
            raise ValueError(f"Negative count for {elem}: {n}")
        idx_slice = target_idx[perm[start:start + n]]
        symbols[idx_slice] = elem
        start += n

    sc.set_chemical_symbols(symbols.tolist())
    return sc
