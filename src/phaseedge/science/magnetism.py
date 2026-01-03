from typing import Tuple, Dict
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.schemas.calc_spec import SpinType
from phaseedge.science.prototype_spec import PrototypeSpec


def assign_up_down_spins_to_prototype(
    *,
    prototype_spec: PrototypeSpec,
    supercell_diag: Tuple[int, int, int],
    spin_type: SpinType,
) -> Structure:
    """
    Seed collinear spin *signs* (+/-) onto a supercell built from a PrototypeSpec.

    SpinType behaviors (site property: "magmom")
    --------------------------------------------
    In all cases, non-active sublattices receive magmom = 0.0
    - FERROMAGNETIC:
        all active sites -> +1
    - SUBLATTICES_ANTIALIGNED:
        Requirement: The prototype must have EXACTLY TWO active sublattices (e.g. {"Es","Fm"}).
        active_sorted[0] -> +1, active_sorted[1] -> -1
        (deterministic ordering via sorted(active_sublattices))
    - AFM_CHECKERBOARD_WITHIN_SUBLATTICE:
        Requirement: Each active sublattice placeholder symbol must correspond to EXACTLY ONE basis
        site in the primitive cell, i.e. active_sublattice_counts[sym] == 1.
        Within each active sublattice, assign sign by parity of (i+j+k) where
        (i,j,k) are the supercell replication indices inferred from fractional coords.
    """

    # Build ASE supercell then convert to pymatgen
    primitive = prototype_spec.primitive_cell
    sc = primitive.repeat(supercell_diag)
    structure = AseAtomsAdaptor.get_structure(sc)

    symbols = np.array(sc.get_chemical_symbols(), dtype=object)
    mag = np.zeros(len(symbols), dtype=float)

    active = sorted(prototype_spec.active_sublattices)
    if spin_type is SpinType.FERROMAGNETIC:
        mag[np.isin(symbols, active)] = 1.0

    elif spin_type is SpinType.SUBLATTICES_ANTIALIGNED:
        if len(active) != 2:
            raise ValueError(
                f"magnetize_prototype requires exactly 2 active sublattices; got {active} "
                f"from prototype {prototype_spec.prototype!r}"
            )
        up_sym, dn_sym = active
        mag[symbols == up_sym] = 1.0
        mag[symbols == dn_sym] = -1.0

    elif spin_type is SpinType.AFM_CHECKERBOARD_WITHIN_SUBLATTICE:
        # Fail loudly if a sublattice symbol maps to multiple basis sites
        counts = prototype_spec.active_sublattice_counts
        bad = {sym: counts.get(sym, 0) for sym in active if counts.get(sym, 0) != 1}
        if bad:
            raise ValueError(
                "AFM_CHECKERBOARD_WITHIN_SUBLATTICE requires each active sublattice "
                "placeholder symbol to appear exactly once in the primitive cell, but got "
                f"{bad}. This is ambiguous because multiple basis sites share the same "
                "placeholder symbol.\n"
                "Fix options:\n"
                "  - Use distinct placeholder symbols for distinct basis sites/sublattices, or\n"
                "  - Implement a mask-/metadata-based assignment (e.g., using per-site labels) "
                "instead of symbol equality."
            )

        # Determine the (unique) basis fractional coordinate for each active symbol
        prim_syms = np.array(primitive.get_chemical_symbols(), dtype=object)
        prim_frac = np.array(primitive.get_scaled_positions(), dtype=float)

        basis_frac_by_sym: Dict[str, np.ndarray] = {}
        for sym in active:
            idx = np.where(prim_syms == sym)[0]
            # idx.size must be 1 by check above
            basis_frac_by_sym[sym] = prim_frac[int(idx[0])].copy()

        sc_frac = np.array(sc.get_scaled_positions(), dtype=float)
        diag = np.array(supercell_diag, dtype=float)

        for sym in active:
            idx = np.where(symbols == sym)[0]
            if idx.size == 0:
                continue

            basis = basis_frac_by_sym[sym]

            # infer integer supercell translation indices (i,j,k)
            # formula: frac_super * diag = basis + (i,j,k)
            t_float = sc_frac[idx] * diag - basis
            t_int = np.rint(t_float).astype(int)

            # loud sanity check
            if not np.allclose(t_float, t_int, atol=1e-6):
                max_dev = float(np.max(np.abs(t_float - t_int)))
                raise RuntimeError(
                    f"Failed to infer integer replication indices for sublattice {sym}. "
                    f"Max deviation from integer: {max_dev}. "
                    f"supercell_diag={supercell_diag}, prototype={prototype_spec.prototype!r}"
                )

            parity = (t_int[:, 0] + t_int[:, 1] + t_int[:, 2]) & 1
            mag[idx] = np.where(parity == 0, 1.0, -1.0)

    else:
        raise ValueError(f"Unsupported spin_type={spin_type}")

    structure.add_site_property("magmom", mag.tolist())
    return structure


def _frac_key(frac: np.ndarray, *, decimals: int) -> tuple[float, float, float]:
    """
    Canonical key for matching sites between structures when positions are identical.
    We mod into [0,1) then round.
    """
    f = np.mod(frac, 1.0)
    f = np.round(f, decimals=decimals)
    return (float(f[0]), float(f[1]), float(f[2]))


def _greedy_unique_nn_map_pbc(
    *,
    lattice,
    a_frac: np.ndarray,  # shape (Na,3) sites to be mapped (e.g. out)
    b_frac: np.ndarray,  # shape (Nb,3) reference sites (e.g. template)
    tol_A: float,
) -> np.ndarray:
    """
    Return mapping idx_b_of_a with length Na, assigning each row in a to a unique
    nearest neighbor in b using PBC distances. Fails loudly if any a cannot be
    matched within tol_A.
    """
    Na = a_frac.shape[0]
    Nb = b_frac.shape[0]
    if Na != Nb:
        raise ValueError(f"NN map requires equal sizes; got {Na} vs {Nb}")

    # PBC-aware distance matrix (Å)
    dmat = lattice.get_all_distances(a_frac, b_frac)  # (Na, Nb)

    # Greedy global assignment by smallest distances
    pairs = [(float(dmat[i, j]), i, j) for i in range(Na) for j in range(Nb)]
    pairs.sort(key=lambda x: x[0])

    a2b = np.full(Na, -1, dtype=int)
    used_b = np.zeros(Nb, dtype=bool)

    for d, i, j in pairs:
        if a2b[i] != -1 or used_b[j]:
            continue
        if d > tol_A:
            break
        a2b[i] = j
        used_b[j] = True

    bad = np.where(a2b == -1)[0]
    if bad.size:
        # diagnostic: nearest distance for each failed site
        nearest = np.min(dmat[bad], axis=1)
        worst = float(np.max(nearest))
        example = bad[:10].tolist()
        raise RuntimeError(
            f"Nearest-neighbor PBC mapping failed for {bad.size}/{Na} sites within tol={tol_A} Å. "
            f"Example indices={example}. Worst nearest distance among failures: {worst} Å."
        )

    return a2b


def magnetize_structure(
    *,
    structure: Structure,
    prototype_spec: PrototypeSpec,
    supercell_diag: Tuple[int, int, int],
    spin_type: SpinType,
) -> Structure:
    """
    Assign per-site collinear magnetic moments ("magmom" site property) onto `structure`
    by:
      1) creating a prototype-derived template of +/- signs (and zeros) and
      2) transferring the sign to the actual element-occupied structure, using
         a hard-coded element->|moment| table (first pass).

    Matching strategy:
      - First try exact (rounded) fractional-coordinate key match (fast path).
      - If any sites fail to map, fall back to a unique nearest-neighbor match
        with PBC distances in Cartesian space (robust path).
    """
    # 1) Build sign template on the prototype supercell
    template = assign_up_down_spins_to_prototype(
        prototype_spec=prototype_spec,
        supercell_diag=supercell_diag,
        spin_type=spin_type,
    )

    if len(template) != len(structure):
        raise ValueError(
            f"Template and structure have different numbers of sites: "
            f"{len(template)} vs {len(structure)}. "
            f"prototype={prototype_spec.prototype!r}, supercell_diag={supercell_diag}"
        )

    # Lattice sanity check
    if not np.allclose(template.lattice.matrix, structure.lattice.matrix, atol=1e-6, rtol=0):
        raise ValueError(
            "Template and structure lattices differ; cannot safely map magmoms.\n"
            f"prototype={prototype_spec.prototype!r}, supercell_diag={supercell_diag}"
        )

    if "magmom" not in template.site_properties:
        raise RuntimeError("Template structure missing required site property 'magmom'.")

    tmpl_mag = np.array(template.site_properties["magmom"], dtype=float)

    # 2) Build mapping from fractional-coordinate key -> template index
    # (store index rather than magmom to support diagnostics / uniqueness)
    tmpl_map: Dict[tuple[float, float, float], int] = {}
    for j, site in enumerate(template.sites):
        k = _frac_key(np.array(site.frac_coords, dtype=float), decimals=8)
        if k in tmpl_map:
            raise RuntimeError(
                "Duplicate fractional-coordinate key in template; rounding collision.\n"
                f"Key={k}. Consider increasing decimals or using NN matching."
            )
        tmpl_map[k] = j

    # 3) First-pass element -> moment magnitude table
    moment_mag: Dict[str, float] = {
        "Cr": 3.0,
        "Mn": 5.0,
        "Fe": 5.0,
        "Co": 3.0,
        "Ni": 2.0,
        "V": 3.0,
        "Cu": 1.0,
        "Ti": 1.0,
        "Ce": 1.0,
    }
    magnetic_elements = set(moment_mag.keys())

    # 4) Try fast-path exact frac-key mapping
    out_mag = np.zeros(len(structure), dtype=float)
    missing: list[int] = []
    mapped_template_idx = np.full(len(structure), -1, dtype=int)

    tol_zero = 1e-12

    for i, site in enumerate(structure.sites):
        k = _frac_key(np.array(site.frac_coords, dtype=float), decimals=8)
        j = tmpl_map.get(k)
        if j is None:
            missing.append(i)
            continue
        mapped_template_idx[i] = j

    # 5) If any missing, fall back to robust unique NN mapping in Å
    if missing:
        # Use frac coords + PBC distances. Because lattices match, either lattice is fine.
        out_frac = np.array([s.frac_coords for s in structure.sites], dtype=float)
        tmpl_frac = np.array([s.frac_coords for s in template.sites], dtype=float)

        # Tight tolerance: if structures are "the same", nearest distances should be ~0
        a2b = _greedy_unique_nn_map_pbc(
            lattice=structure.lattice,
            a_frac=out_frac,
            b_frac=tmpl_frac,
            tol_A=1e-3,  # adjust to 1e-2 if you suspect slightly perturbed coords
        )
        mapped_template_idx = a2b

    # 6) Transfer with your rules
    for i, site in enumerate(structure.sites):
        j = int(mapped_template_idx[i])
        if j < 0:
            # Should not happen; NN fallback should have filled all.
            raise RuntimeError(f"Internal error: unmapped site {i}")

        m_t = float(tmpl_mag[j])
        elem = site.specie.symbol

        if abs(m_t) <= tol_zero:
            if elem in magnetic_elements:
                raise ValueError(
                    "Template indicates a non-magnetic site (magmom=0), but the target structure "
                    f"has a magnetic element {elem} at that position.\n"
                    f"Site index={i}, frac={site.frac_coords}, spin_type={spin_type}.\n"
                    "This usually means your prototype active-sublattice definition does not match "
                    "which sites you intend to allow magnetism on."
                )
            out_mag[i] = 0.0
        else:
            if elem in magnetic_elements:
                sign = 1.0 if m_t > 0 else -1.0
                out_mag[i] = sign * moment_mag[elem]
            else:
                out_mag[i] = 0.0

    out = structure.copy()
    out.add_site_property("magmom", out_mag.tolist())
    return out
