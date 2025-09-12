from typing import Mapping

import numpy as np
from ase.atoms import Atoms


def validate_counts_for_sublattices(
    *,
    conv_cell: Atoms,
    supercell_diag: tuple[int, int, int],
    composition_map: Mapping[str, Mapping[str, int]],
) -> None:
    """
    Validate that, for each replace_element key in `composition_map`, the integer
    counts sum to the number of replaceable sites on that sublattice in the
    supercell (conv_cell repeated by supercell_diag).

    Raises
    ------
    ValueError
        If any count is negative or if the counts do not sum to the number
        of sites on that sublattice.
    """
    sc = conv_cell.repeat(supercell_diag)
    symbols = np.array(sc.get_chemical_symbols())

    for replace_element, counts in composition_map.items():
        target_idx = np.where(symbols == replace_element)[0]
        n_sites = int(target_idx.size)

        if any(int(v) < 0 for v in counts.values()):
            raise ValueError(f"Negative count in counts for {replace_element}: {counts}")

        total = sum(int(v) for v in counts.values())
        if total != n_sites:
            raise ValueError(
                f"[{replace_element}] counts must sum to sublattice size: "
                f"got {total}, expected {n_sites}"
            )


def make_one_snapshot(
    *,
    conv_cell: Atoms,
    supercell_diag: tuple[int, int, int],
    composition_map: Mapping[str, Mapping[str, int]],
    rng: np.random.Generator,
) -> Atoms:
    """
    Build a supercell, then for each sublattice (identified by its placeholder
    `replace_element` symbol), assign exact integer counts by permuting that
    sublattice's indices once and filling contiguous slices of the permutation.

    Determinism
    -----------
    - Deterministic given (conv_cell, supercell_diag, composition_map, rng state).
    - We iterate both sublattices and element labels in sorted order.
    - Each call to `rng.permutation(...)` advances the RNG state, so two
      sublattices with the same size still receive different permutations.

    Notes
    -----
    This selector uses **element symbol equality** to define a sublattice:
        target_idx = np.where(symbols == replace_element)
    If multiple distinct sublattices share the *same* placeholder symbol in the
    prototype, they will be merged. For true multi-sublattice separation you’ll
    need distinct placeholders in the prototype or an index-mask approach.
    """
    validate_counts_for_sublattices(
        conv_cell=conv_cell,
        supercell_diag=supercell_diag,
        composition_map=composition_map,
    )

    sc = conv_cell.repeat(supercell_diag)
    symbols = np.array(sc.get_chemical_symbols())

    # Deterministic order across dicts
    for replace_element, counts in sorted(composition_map.items()):
        target_idx = np.where(symbols == replace_element)[0]
        n_sites = int(target_idx.size)

        # One permutation per sublattice; RNG state advances per call.
        perm = rng.permutation(n_sites)

        start = 0
        for elem, n in sorted(counts.items()):  # stable element order
            n_int = int(n)
            if n_int <= 0:
                raise ValueError(f"Zero or negative count for {elem}: {n_int}")

            idx_slice = target_idx[perm[start:start + n_int]]
            symbols[idx_slice] = elem
            start += n_int

        # Safety: ensure we consumed exactly n_sites entries on this sublattice
        if start != n_sites:
            raise RuntimeError(
                f"[{replace_element}] count slices consumed {start} out of {n_sites} sites."
            )

    sc.set_chemical_symbols(symbols.tolist())
    return sc
