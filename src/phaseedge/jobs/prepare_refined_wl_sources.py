from typing import Any, Mapping, Sequence, cast

from jobflow.core.job import job

from phaseedge.science.prototypes import PrototypeName
from phaseedge.utils.keys import canonical_counts, compute_ce_key


@job
def prepare_refined_wl_sources(
    *,
    # CE system identity
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,

    # CE hyperparameters (same basis/regs used for final CE key)
    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any] | None,
    extra_hyperparams: Mapping[str, Any] | None,
    weighting: Mapping[str, Any] | None,

    # Final training engine identity (goes into final ce_key)
    train_model: str,
    train_relax_cell: bool,
    train_dtype: str,

    # INTENT inputs (submit-time determinism)
    base_ce_key: str,                           # random/composition CE key
    endpoints: Sequence[Mapping[str, int]],     # canonicalized inside

    wl_policy: Mapping[str, Any],               # {bin_width, step_type, check_period, update_period, seed}
    ensure: Mapping[str, Any],                  # {steps_to_run, samples_per_bin}
    refine: Mapping[str, Any],                  # {mode, n_total|null, per_bin_cap|null, strategy}
    dopt: Mapping[str, Any],                    # {budget, ridge, tie_breaker}

    # Optional provenance (realization details; ignored for hashing)
    # from select_d_optimal_basis: list of {"source","wl_key","checkpoint_hash","bin","occ","occ_hash", ...}
    chosen: Sequence[Mapping[str, Any]] = (),
    # list of refine job outputs: each has {"wl_key","checkpoint_hash","policy",...}
    refine_results: Sequence[Mapping[str, Any]] = (),

    # Algo tag for refined intent hashing
    algo_version: str = "refined-wl-dopt-v2",
) -> Mapping[str, Any]:
    """
    Build the refined-WL *intent* source (which fully determines the CE key)
    and compute the final_ce_key. Optionally return a provenance blob
    recording which checkpoints and occ_hashes were realized, but that has
    no effect on the key (and is not included in sources used for hashing).
    """
    # --- INTENT block (used for hashing) ---
    endpoints_canon = [canonical_counts(e) for e in endpoints]
    src_intent = {
        "type": "wl_refined_intent",
        "base_ce_key": str(base_ce_key),
        "endpoints": endpoints_canon,
        "wl_policy": {
            "bin_width": float(wl_policy["bin_width"]),
            "step_type": str(wl_policy["step_type"]),
            "check_period": int(wl_policy["check_period"]),
            "update_period": int(wl_policy["update_period"]),
            "seed": int(wl_policy["seed"]),
        },
        "ensure": {
            "steps_to_run": int(ensure["steps_to_run"]),
            "samples_per_bin": int(ensure["samples_per_bin"]),
        },
        "refine": {
            "mode": str(refine["mode"]),
            "n_total": (None if refine.get("n_total") is None else int(refine["n_total"])),
            "per_bin_cap": (None if refine.get("per_bin_cap") is None else int(refine["per_bin_cap"])),
            "strategy": str(refine["strategy"]),
        },
        "dopt": {
            "budget": int(dopt["budget"]),
            "ridge": float(dopt["ridge"]),
            "tie_breaker": str(dopt.get("tie_breaker", "bin_then_hash")),
        },
        "versions": {
            "refine": "refine-wl-v1",
            "dopt": "dopt-greedy-sm-v1",
            "sampler": "wl-grid-v1",
        },
    }
    sources_intent = [src_intent]

    # --- PROVENANCE (optional, not used in hashing) ---
    # Map (wl_key, hash) -> policy (from refine job), and list selected occ_hashes deterministically
    policy_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in refine_results:
        wl_key = str(r["wl_key"])
        ck_hash = str(r["checkpoint_hash"])
        policy = cast(Mapping[str, Any], r.get("policy", {}))
        policy_map[(wl_key, ck_hash)] = policy

    chosen_by_chain: dict[tuple[str, str], list[str]] = {}
    for c in chosen:
        if str(c.get("source", "")) != "wl":
            continue
        wl_key = str(c["wl_key"])
        ck_hash = str(c["checkpoint_hash"])
        occ_hash = str(c["occ_hash"])
        chosen_by_chain.setdefault((wl_key, ck_hash), []).append(occ_hash)
    for v in chosen_by_chain.values():
        v.sort()

    provenance = {
        "type": "wl_refined_provenance",
        "chains": [
            {
                "wl_key": wl_key,
                "checkpoint_hash": ck_hash,
                "refine": policy_map.get((wl_key, ck_hash)),
                "occ_hashes": occ_hashes,
            }
            for (wl_key, ck_hash), occ_hashes in sorted(chosen_by_chain.items())
        ],
        "endpoints": endpoints_canon,
        "budget": int(dopt.get("budget", 0)),
    }

    # --- Compute final key from INTENT only ---
    final_ce_key: str = compute_ce_key(
        prototype=prototype,
        prototype_params=dict(prototype_params),
        supercell_diag=supercell_diag,
        replace_element=replace_element,
        sources=sources_intent,
        model=train_model,
        relax_cell=bool(train_relax_cell),
        dtype=train_dtype,
        basis_spec=dict(basis_spec),
        regularization=dict(regularization or {}),
        extra_hyperparams=dict(extra_hyperparams or {}),
        algo_version=str(algo_version),
        weighting=dict(weighting or {}),
    )

    return {
        "final_ce_key": final_ce_key,
        "sources_intent": sources_intent,
        "provenance": provenance,
    }
