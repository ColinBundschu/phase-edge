import numpy as np
from ase.atoms import Atoms
from typing import Iterable, Mapping

from phaseedge.schemas.sublattice import SublatticeSpec


def validate_counts_for_sublattice(
    *,
    conv_cell: Atoms,
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    counts: Mapping[str, int],
) -> tuple[int, np.ndarray]:
    """
    Validate that integer counts sum to the number of replaceable sites on the target sublattice.

    Returns:
        (n_sites, target_idx) where target_idx are indices of the replaceable sites in the supercell.

    Raises:
        ValueError if counts don't sum to n_sites or any count is negative.
    """
    sc = conv_cell.repeat(supercell_diag)
    symbols = np.array(sc.get_chemical_symbols())
    target_idx = np.where(symbols == replace_element)[0]
    n_sites = int(target_idx.size)

    total = 0
    for elem, n in counts.items():
        n_int = int(n)
        if n_int < 0:
            raise ValueError(f"Negative count for {elem}: {n_int}")
        total += n_int

    if total != n_sites:
        raise ValueError(
            f"Counts must sum to replacement sublattice size: got {total}, expected {n_sites}"
        )

    return n_sites, target_idx


def make_one_snapshot(
    *,
    conv_cell: Atoms,
    supercell_diag: tuple[int, int, int],
    sublattices: Iterable[SublatticeSpec],
    rng: np.random.Generator,
) -> Atoms:
    """
    Build a supercell, then for each sublattice spec, replace exactly its target sites per integer counts.
    Deterministic given the provided RNG and a fixed order of `sublattices`.

    - conv_cell: primitive or conventional prototype Atoms
    - supercell_diag: (nx, ny, nz)
    - sublattices: iterable of SublatticeSpec (one per sublattice you want to modify)
    - rng: NumPy Generator whose state determines the deterministic assignment

    Returns:
        A new ASE Atoms with the requested species assigned on each sublattice.

    Raises:
        ValueError if any sublattice counts don’t sum to its size.
    """
    sc = conv_cell.repeat(supercell_diag)
    symbols = np.array(sc.get_chemical_symbols())

    # Work on a sorted, concrete list for deterministic traversal order
    plans = sorted(sublattices, key=lambda s: s.replace_element)

    for spec in plans:
        # locate sublattice
        target_mask = (symbols == spec.replace_element)
        target_idx = np.flatnonzero(target_mask)
        n_sites = int(target_idx.size)

        # validate counts
        total = 0
        for elem, n in spec.counts.items():
            n_int = int(n)
            if n_int < 0:
                raise ValueError(f"Negative count for {elem}: {n_int}")
            total += n_int
        if total != n_sites:
            raise ValueError(
                f"[{spec.replace_element}] counts must sum to {n_sites}, got {total}"
            )

        # deterministic assignment: shuffle indices, then take exact slices per sorted element
        perm = rng.permutation(n_sites)
        start = 0
        # stable ordering so the same RNG state yields identical assignments across runs
        for elem, n in sorted(spec.counts.items()):
            n_int = int(n)
            if n_int == 0:
                continue
            sel = target_idx[perm[start:start + n_int]]
            symbols[sel] = elem
            start += n_int

    sc.set_chemical_symbols(symbols.tolist())
    return sc
