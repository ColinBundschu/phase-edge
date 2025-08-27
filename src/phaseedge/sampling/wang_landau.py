from typing import Any, Mapping
import numpy as np

from smol.cofe import ClusterExpansion
from smol.moca.kernel import WangLandau as SmolWangLandau

from phaseedge.schemas.wl import WLSamplerSpec, WLResult
from phaseedge.utils.grid import snap_floor, snap_ceil, ANCHOR
from phaseedge.storage.ce_store import lookup_ce_by_key
from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import make_one_snapshot, validate_counts_for_sublattice
from phaseedge.science.ce_training import featurize_structures, predict_from_features
from pymatgen.io.ase import AseAtomsAdaptor
from typing import cast, Sequence, Tuple, Dict

def _rehydrate_ce(ce_key: str) -> dict:
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")
    return doc

def _pilot_enthalpies(doc: Mapping[str, Any], *, n: int) -> np.ndarray:
    # mirror your existing rehydrate logic but *sample* random configs
    system = cast(Mapping[str, Any], doc["system"])
    payload = cast(Mapping[str, Any], doc["payload"])
    ce = ClusterExpansion.from_dict(dict(payload))
    subspace = ce.cluster_subspace

    prototype = cast(str, system["prototype"])
    prototype_params = cast(Mapping[str, Any], system["prototype_params"])
    supercell_diag = tuple(system["supercell_diag"])
    replace_element = cast(str, system["replace_element"])

    conv = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))
    # Naive pilot: draw n random compositions from CE’s stored mixture, or uniform counts if absent.
    mixture = list(cast(Sequence[Mapping[str, Any]], doc.get("sampling", {}).get("mixture", []))) or [{}]

    structures = []
    for i in range(n):
        counts = {str(k): int(v) for k, v in dict(mixture[i % len(mixture)].get("counts", {})).items()}
        if counts:
            validate_counts_for_sublattice(conv, tuple(supercell_diag), replace_element, counts)
        snap = make_one_snapshot(conv_cell=conv, supercell_diag=tuple(supercell_diag),
                                 replace_element=replace_element, counts=counts, rng=None)
        structures.append(AseAtomsAdaptor.get_structure(snap))  # type: ignore[arg-type]

    _, X = featurize_structures(subspace=subspace, structures=structures, supercell_diag=tuple(supercell_diag))
    coefs = np.asarray(getattr(ce, "parameters", getattr(ce, "coefs")))
    H = predict_from_features(X, coefs)  # eV per supercell (canonical)
    return np.asarray(H, dtype=float)

def run_wl(spec: WLSamplerSpec) -> WLResult:
    doc = _rehydrate_ce(spec.ce_key)
    H = _pilot_enthalpies(doc, n=spec.pilot_samples)
    mu = float(np.median(H))
    sigma = float(np.std(H)) if np.std(H) > 0 else (spec.bin_width or 1e-3)

    # ± 50 σ window, snapped to zero-anchored grid
    raw_min = mu - spec.sigma_multiplier * sigma
    raw_max = mu + spec.sigma_multiplier * sigma
    H_min = snap_floor(raw_min, spec.bin_width, ANCHOR)
    H_max = snap_ceil(raw_max, spec.bin_width, ANCHOR)

    # --- Build the smol Ensemble and kernel ---
    # NOTE: ensemble building depends on your smol version and CE/ensemble wiring.
    # Pseudocode: ensemble = Ensemble.from_cluster_expansion(ce, composition=..., mu=...)
    # For V1 we assume canonical and use E as “enthalpy”.
    # If semi-grand, you’ll need to set natural_parameters accordingly.

    from smol.moca.ensemble import Ensemble  # adjust import if needed
    payload = dict(doc["payload"])
    ce = ClusterExpansion.from_dict(payload)
    ensemble = Ensemble.from_cluster_expansion(ce)

    kernel = SmolWangLandau(
        ensemble=ensemble,
        step_type="swap",              # or whichever usher you’re using
        min_enthalpy=H_min,
        max_enthalpy=H_max,
        bin_size=spec.bin_width,
        flatness=0.8,
        mod_factor=1.0,
        check_period=spec.check_period,
        update_period=spec.update_period,
        seed=spec.seed,
    )

    # Advance for spec.steps steps
    occ = ensemble.occupancy
    for _ in range(spec.steps):
        step = kernel.mcusher.propose_move(occ)
        occ = kernel._do_accept_step(occ, step)    # using smol’s API
        kernel._do_post_step()

    # Collect result (visited bins only)
    mask = kernel.entropy > 0  # type: ignore[attr-defined]
    levels = kernel.levels     # already masked in smol
    entropy = kernel.entropy
    hist = kernel.histogram

    return WLResult(
        levels=np.asarray(levels),
        entropy=np.asarray(entropy),
        histogram=np.asarray(hist),
        visited_mask=np.ones_like(levels, dtype=bool),
        grid_anchor=ANCHOR,
        bin_width=spec.bin_width,
        window_used=(H_min, H_max),
        meta=dict(
            ce_key=spec.ce_key, steps=spec.steps, seed=spec.seed,
            ensemble=spec.ensemble, composition=spec.composition,
            check_period=spec.check_period, update_period=spec.update_period,
            pilot_samples=spec.pilot_samples, sigma_multiplier=spec.sigma_multiplier,
        ),
    )
