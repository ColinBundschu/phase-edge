from dataclasses import dataclass
from typing import Any, Literal, Mapping

from monty.json import MSONable

from phaseedge.schemas.ensure_ce_from_mixtures_spec import EnsureCEFromMixturesSpec
from phaseedge.schemas.mixture import canonical_comp_map, sorted_composition_maps
from phaseedge.schemas.wl_sampler_spec import WLSamplerSpec
from phaseedge.science.refine_wl import RefineStrategy
from phaseedge.utils.keys import compute_ce_key


def composition_maps_equal(
    a: Mapping[str, Mapping[str, int]],
    b: Mapping[str, Mapping[str, int]],
) -> bool:
    """
    Compare two composition_map objects for equality.

    A composition_map looks like:
        {"Es": {"Mg": 49, "Al": 5}, "Fm": {"Mg": 5, "Al": 103}}

    Rules:
    - Raise ValueError if any element count is zero.
    - Equality means same sublattices, and within each sublattice
      the same element counts.
    """
    def validate_no_zeros(comp: Mapping[str, Mapping[str, int]]) -> None:
        for sublattice, counts in comp.items():
            for elem, count in counts.items():
                if count == 0:
                    raise ValueError(
                        f"Zero count not allowed: {sublattice}[{elem}] = 0"
                    )

    validate_no_zeros(a)
    validate_no_zeros(b)

    if a.keys() != b.keys():
        return False

    for sublattice, a_counts in a.items():
        b_counts = b.get(sublattice, {})
        if dict(a_counts) != dict(b_counts):
            return False

    return True


@dataclass(frozen=True, slots=True)
class EnsureCEFromRefinedWLSpec(MSONable):
    ce_spec: EnsureCEFromMixturesSpec
    endpoints: tuple[dict[str, dict[str, int]], ...]

    wl_bin_width: float
    wl_steps_to_run: int
    wl_samples_per_bin: int
    sl_comp_map: dict[str, dict[str, int]]
    reject_cross_sublattice_swaps: bool

    wl_step_type: str = "swap"
    wl_check_period: int = 5_000
    wl_update_period: int = 1
    wl_seed: int = 0

    refine_n_total: int = 25
    refine_per_bin_cap: int = 5
    refine_strategy: RefineStrategy = RefineStrategy.ENERGY_SPREAD
    train_model: str = "MACE-MPA-0"
    train_relax_cell: bool = False
    budget: int = 64

    # Single category for *everything* (wrapper, CE subflow, WL jobs)
    category: str = "gpu"

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "ce_spec": self.ce_spec.as_dict(),
            "endpoints": list(self.endpoints),
            "sl_comp_map": self.sl_comp_map,
            "reject_cross_sublattice_swaps": self.reject_cross_sublattice_swaps,
            "wl_bin_width": self.wl_bin_width,
            "wl_steps_to_run": self.wl_steps_to_run,
            "wl_samples_per_bin": self.wl_samples_per_bin,
            "wl_step_type": self.wl_step_type,
            "wl_check_period": self.wl_check_period,
            "wl_update_period": self.wl_update_period,
            "wl_seed": self.wl_seed,
            "refine_n_total": self.refine_n_total,
            "refine_per_bin_cap": self.refine_per_bin_cap,
            "refine_strategy": self.refine_strategy,
            "train_model": self.train_model,
            "train_relax_cell": self.train_relax_cell,
            "budget": self.budget,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EnsureCEFromRefinedWLSpec":
        ce_spec = d["ce_spec"]
        if not isinstance(ce_spec, EnsureCEFromMixturesSpec):
            ce_spec = EnsureCEFromMixturesSpec.from_dict(ce_spec)
        return cls(
            ce_spec=ce_spec,
            endpoints=sorted_composition_maps(d["endpoints"]),
            sl_comp_map=canonical_comp_map(d["sl_comp_map"]),
            reject_cross_sublattice_swaps=bool(d["reject_cross_sublattice_swaps"]),
            wl_bin_width=float(d["wl_bin_width"]),
            wl_steps_to_run=int(d["wl_steps_to_run"]),
            wl_samples_per_bin=int(d["wl_samples_per_bin"]),
            wl_step_type=str(d["wl_step_type"]),
            wl_check_period=int(d["wl_check_period"]),
            wl_update_period=int(d["wl_update_period"]),
            wl_seed=int(d["wl_seed"]),
            refine_n_total=int(d["refine_n_total"]),
            refine_per_bin_cap=int(d["refine_per_bin_cap"]),
            refine_strategy=RefineStrategy(d["refine_strategy"]),
            train_model=str(d["train_model"]),
            train_relax_cell=bool(d["train_relax_cell"]),
            budget=int(d["budget"]),
            category=str(d.get("category", "gpu")),
        )


    @property
    def refine_mode(self) -> Literal["all", "refine"]:
        return "all" if self.refine_n_total == 0 else "refine"
    

    @property
    def wl_sampler_specs(self):
        return [WLSamplerSpec(
            ce_key=self.ce_spec.ce_key,
            bin_width=self.wl_bin_width,
            steps=self.wl_steps_to_run,
            sl_comp_map=self.sl_comp_map,
            initial_comp_map=mix.composition_map,
            step_type=self.wl_step_type,
            check_period=self.wl_check_period,
            update_period=self.wl_update_period,
            seed=self.wl_seed,
            samples_per_bin=self.wl_samples_per_bin,
            reject_cross_sublattice_swaps=self.reject_cross_sublattice_swaps,
        ) for mix in self.ce_spec.mixtures if mix.composition_map not in self.endpoints]
    
    @property
    def final_ce_key(self) -> str:
        return compute_ce_key(
            prototype=self.ce_spec.prototype,
            prototype_params=dict(self.ce_spec.prototype_params),
            supercell_diag=self.ce_spec.supercell_diag,
            sources=[self.source],
            model=self.train_model,
            relax_cell=self.train_relax_cell,
            basis_spec=self.ce_spec.basis_spec,
            regularization=self.ce_spec.regularization,
            algo_version="refined-wl-dopt-v2",
            weighting=self.ce_spec.weighting,
        )

    @property
    def source(self):
        wl_policy_for_key = {
            "bin_width": self.wl_bin_width,
            "step_type": self.wl_step_type,
            "check_period": self.wl_check_period,
            "update_period": self.wl_update_period,
            "seed": self.wl_seed,
        }
        ensure_policy_for_key = {
            "steps_to_run": self.wl_steps_to_run,
            "samples_per_bin": self.wl_samples_per_bin,
        }
        refine_options_for_key = {
            "mode": self.refine_mode,
            "n_total": self.refine_n_total,
            "per_bin_cap": self.refine_per_bin_cap,
            "strategy": self.refine_strategy,
        }
        dopt_options_for_key = {
            "budget": self.budget,
            "ridge": float(1e-10),
            "tie_breaker": "bin_then_hash",
        }
        source = {
            "type": "wl_refined_intent",
            "base_ce_key": self.ce_spec.ce_key,
            "endpoints": self.endpoints,
            "wl_policy": wl_policy_for_key,
            "ensure": ensure_policy_for_key,
            "refine": refine_options_for_key,
            "dopt": dopt_options_for_key,
            "versions": {
                "refine": "refine-wl-v1",
                "dopt": "dopt-rr-sm-v1",
                "sampler": "wl-grid-v1",
            },
        }
        
        return source
