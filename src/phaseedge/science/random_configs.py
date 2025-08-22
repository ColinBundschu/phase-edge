import math
from numpy.random import Generator
from ase.atoms import Atoms


def make_one_snapshot(
    conv_cell: Atoms,
    supercell_diag: tuple[int, int, int],
    *,
    replace_element: str,
    composition: dict[str, float],
    rng: Generator,
) -> Atoms:
    """
    Generate one random snapshot by assigning the `replace_element` sites
    in the replicated rock-salt prototype to species according to `composition`.

    Deterministic for a given RNG state.
    """
    proto = conv_cell * supercell_diag
    repl_idx = [i for i, at in enumerate(proto) if at.symbol == replace_element]
    n_replace = len(repl_idx)

    if not math.isclose(sum(composition.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"Fractions must sum to 1; got {composition}")

    # target integer counts (rounded), then adjust to hit exact total
    counts = {el: round(frac * n_replace) for el, frac in composition.items()}
    delta = n_replace - sum(counts.values())
    if delta:
        keys = list(counts)
        for k in range(abs(delta)):
            counts[keys[k % len(keys)]] += int(math.copysign(1, delta))

    # random assignment via permutation
    rng.shuffle(repl_idx)
    snapshot = proto.copy()
    start = 0
    for el, n_el in counts.items():
        end = start + n_el
        idx_slice = repl_idx[start:end]
        snapshot.symbols[idx_slice] = el
        start = end

    return snapshot
