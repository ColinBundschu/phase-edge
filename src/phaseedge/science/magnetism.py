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


def _frac_key(frac: np.ndarray, *, decimals: int = 6) -> tuple[float, float, float]:
    """
    Canonical key for matching sites between structures when positions are identical.
    We mod into [0,1) then round.
    """
    f = np.mod(frac, 1.0)
    f = np.round(f, decimals=decimals)
    return (float(f[0]), float(f[1]), float(f[2]))


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

    Matching:
      - We do NOT assume site ordering matches.
      - We DO assume site positions (fractional coords) match exactly between
        `structure` and the template (within rounding tolerance).

    Rules:
      - If template site magmom == 0:
          require output site element to be non-magnetic, else raise.
      - If template site magmom != 0:
          if output element is magnetic, assign sign(template)*|moment(element)|
          else assign 0 (non-magnetic elements remain 0 even if template says +/-).
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

    # (Optional) quick lattice sanity check — same shape & orientation.
    # We allow tiny numerical differences; if you expect exact identity, tighten tolerances.
    if not np.allclose(template.lattice.matrix, structure.lattice.matrix, atol=1e-6, rtol=0):
        raise ValueError(
            "Template and structure lattices differ; cannot safely map magmoms by fractional coords.\n"
            f"prototype={prototype_spec.prototype!r}, supercell_diag={supercell_diag}"
        )

    if "magmom" not in template.site_properties:
        raise RuntimeError("Template structure missing required site property 'magmom'.")

    tmpl_mag = np.array(template.site_properties["magmom"], dtype=float)

    # 2) Build mapping from fractional-coordinate key -> template magmom
    tmpl_map: Dict[tuple[float, float, float], float] = {}
    for site, m in zip(template.sites, tmpl_mag):
        k = _frac_key(np.array(site.frac_coords, dtype=float), decimals=8)
        if k in tmpl_map:
            raise RuntimeError(
                "Duplicate fractional-coordinate key in template; rounding collision.\n"
                f"Key={k}. Consider increasing decimals or using a KD-tree match."
            )
        tmpl_map[k] = float(m)

    # 3) First-pass element -> moment magnitude table (tune as you like)
    # Values are “reasonable” high-spin-ish seeds for oxides; exact oxidation state
    # may differ, but these are just initial guesses.
    moment_mag: Dict[str, float] = {
        "Cr": 3.0,
        "Mn": 5.0,
        "Fe": 5.0,
        "Co": 3.0,
        "Ni": 2.0,
        # Add more here as you expand:
        # "V": 3.0,
        # "Cu": 1.0,
        # "Ti": 1.0,
        # "Ce": 1.0,
    }
    magnetic_elements = set(moment_mag.keys())

    # 4) Transfer to output structure
    out = structure.copy()
    out_mag = np.zeros(len(out), dtype=float)

    missing: list[int] = []
    tol_zero = 1e-12

    for i, site in enumerate(out.sites):
        k = _frac_key(np.array(site.frac_coords, dtype=float), decimals=8)
        if k not in tmpl_map:
            missing.append(i)
            continue

        m_t = tmpl_map[k]
        elem = site.specie.symbol

        if abs(m_t) <= tol_zero:
            # Template says "this site should be non-magnetic"
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
            # Template provides sign; apply only if element is magnetic
            if elem in magnetic_elements:
                sign = 1.0 if m_t > 0 else -1.0
                out_mag[i] = sign * moment_mag[elem]
            else:
                # Non-magnetic element: keep 0 even if template wants +/- here.
                out_mag[i] = 0.0

    if missing:
        # Loud failure with useful diagnostics
        example = missing[:5]
        raise RuntimeError(
            f"Failed to map {len(missing)}/{len(out)} sites by fractional coordinate. "
            f"Example indices: {example}. "
            "If the positions are not exactly identical (or rounding is too coarse), "
            "switch to a nearest-neighbor match in Cartesian space with a tolerance."
        )

    out.add_site_property("magmom", out_mag.tolist())
    return out
