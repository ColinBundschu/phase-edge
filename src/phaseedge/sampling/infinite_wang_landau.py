"""
An "infinite-window" Wang-Landau kernel for multicanonical estimation of the DOS.

This is a minimally edited copy of smol's Wang-Landau kernel that:
- Removes the fixed enthalpy window (min/max).
- Stores per-bin state sparsely in dicts, creating bins lazily as they are visited.
- Snaps bins to a global anchor at 0.0 using bin_id = floor(E / bin_size).
- Exposes the same public properties (levels/entropy/histogram/dos) but only for visited bins.
- Keeps Wang-Landau logic (flatness check, mod_factor schedule) intact.
- Can capture up to K UNIQUE occupancy samples per visited bin (K = samples_per_bin; runtime policy).
- Can optionally collect per-bin cation-count histograms per sublattice.
- Can run in "production_mode" where WL updates are frozen but statistics are collected.

Based on: https://link.aps.org/doi/10.1103/PhysRevLett.86.2050
"""

from functools import partial
from math import log
from typing import Callable, Mapping, Any

import hashlib
import numpy as np

from smol.moca.kernel.base import ALL_MCUSHERS, MCKernel

ANCHOR: float = 0.0  # global grid anchor (baked into the logic of the class)


def _divide(x: float, m: float) -> float:
    """Use to allow Wang-Landau to be pickled (same helper shape as smol)."""
    return x / m


class InfiniteWangLandau(MCKernel):
    """Wang-Landau sampling kernel with an effectively unbounded enthalpy domain.

    Bins are created lazily when visited and stored sparsely.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = None  # Wang-Landau does not need bias.

    def __init__(
        self,
        ensemble,
        step_type: str,
        bin_size: float,
        *args,
        flatness: float = 0.8,
        mod_factor: float = 1.0,
        check_period: int = 1000,
        update_period: int = 1,
        mod_update: float | Callable[[float], float] | None = None,
        seed: int | None = None,
        samples_per_bin: int = 0,
        # -------- NEW options for statistics --------
        collect_cation_stats: bool = False,
        production_mode: bool = False,
        # Provide optional precomputed mappings (recommended):
        # - sublattice_indices: label -> indices in occupancy vector
        # - species_code_to_label: occupancy code -> chemical symbol
        sublattice_indices: Mapping[str, list[int] | np.ndarray] | None = None,
        species_code_to_label: Mapping[int, str] | None = None,
        **kwargs,
    ):
        """Initialize an infinite-window Wang-Landau Kernel.

        Args:
            ensemble (Ensemble): Ensemble object used to generate samples.
            step_type (str): An MC step type corresponding to an MCUsher.
            bin_size (float): Enthalpy bin size (eV per supercell).
            flatness (float): Flatness factor for histogram checks (0-1).
            mod_factor (float): Initial log modification factor for entropy updates (>0).
            check_period (int): Steps between flatness checks.
            update_period (int): Steps between histogram/entropy updates.
            mod_update (float|Callable): If number, divide mod_factor by this each flatness reset;
                                         if callable, arbitrary decreasing schedule.
            seed (int): RNG seed for the kernel.
            samples_per_bin (int): Runtime policy: capture at most this many UNIQUE occupancy samples per bin.
            collect_cation_stats (bool): If True, collect per-bin cation-count histograms per sublattice.
            production_mode (bool): If True, freeze WL updates (entropy/histogram/mod-factor) and only collect stats.
            sublattice_indices: Optional mapping label -> site indices for each replaceable sublattice.
            species_code_to_label: Optional mapping from integer occupancy code -> chemical symbol.
            *args, **kwargs: forwarded to MCUsher constructor.
        """
        if mod_factor <= 0:
            raise ValueError("mod_factor must be greater than 0.")
        if bin_size <= 0:
            raise ValueError("bin_size must be greater than 0.")

        self.flatness = float(flatness)
        self.check_period = int(check_period)
        self.update_period = int(update_period)
        self._m = float(mod_factor)
        self._bin_size = float(bin_size)
        self._mod_updates_buf: list[tuple[int, float]] = []
        self._mod_updates_buf.append((0, self._m))

        if callable(mod_update):
            self._mod_update: Callable[[float], float] = mod_update  # type: ignore[assignment]
        elif mod_update is not None:
            self._mod_update = partial(_divide, m=float(mod_update))
        else:
            self._mod_update = partial(_divide, m=2.0)

        # Sparse per-bin state (int bin_id -> values). A bin becomes "visited"
        # once its entropy > 0, matching smol's mask logic on fixed arrays.
        self._entropy_d: dict[int, float] = {}
        self._histogram_d: dict[int, int] = {}
        self._occurrences_d: dict[int, int] = {}
        self._mean_features_d: dict[int, np.ndarray] = {}

        # Optional sparse capture of occupancy samples per bin (UNIQUE)
        self._samples_per_bin: int = max(0, int(samples_per_bin))
        self._bin_samples_d: dict[int, list[list[int]]] = {}
        self._bin_sample_hashes_d: dict[int, set[str]] = {}  # transient; not serialized

        # Keep the last accepted occupancy so we can record it in _do_post_step
        self._last_accepted_occupancy: np.ndarray | None = None

        # The correct initialization will be handled in set_aux_state.
        self._current_enthalpy: float = np.inf
        self._current_features = np.zeros(len(ensemble.natural_parameters))
        self._nfeat = len(ensemble.natural_parameters)
        self._steps_counter = 0  # number of valid states elapsed

        # ------- NEW: sublattice/species configuration -------
        self._collect_cation_stats = collect_cation_stats
        self._production_mode = production_mode

        # label -> np.ndarray[int] (site indices)
        self._sl_index_map: dict[str, np.ndarray] = {}
        if sublattice_indices:
            for k, v in sublattice_indices.items():
                self._sl_index_map[str(k)] = np.asarray(v, dtype=np.int32)

        # code -> label mapping
        # NOTE: When collecting cation stats we require a non-empty, one-to-one mapping.
        raw_map = species_code_to_label or {}
        self._code_to_label: dict[int, str] = {int(k): str(v).strip() for k, v in raw_map.items()}

        if self._collect_cation_stats:
            if not self._code_to_label:
                raise ValueError(
                    "collect_cation_stats=True requires a non-empty species_code_to_label (code -> label) mapping."
                )
            labels = list(self._code_to_label.values())
            dup_labels = {lbl for lbl in set(labels) if labels.count(lbl) > 1}
            if dup_labels:
                raise ValueError(
                    "species_code_to_label must be one-to-one (distinct codes must map to distinct labels). "
                    f"Duplicate labels detected: {self._code_to_label}"
                )
            # persist mapping for provenance/debugging
            try:
                self.spec.species_code_to_label = dict(self._code_to_label)
            except Exception:
                pass

        # Per-bin cation count histograms:
        # bin_id -> { sublattice_label -> { element_label -> { n_sites_on_sublattice: visits } } }
        self._bin_cation_counts: dict[int, dict[str, dict[str, dict[int, int]]]] = {}

        # Population of initial trace included here.
        super().__init__(ensemble=ensemble, step_type=step_type, *args, seed=seed, **kwargs)

        if self.bias is not None:
            raise ValueError("Cannot apply bias to Wang-Landau simulation!")

        # add inputs to specification (non-serialized internal dict)
        self.spec.bin_size = self._bin_size
        self.spec.flatness = self.flatness
        self.spec.check_period = self.check_period
        self.spec.update_period = self.update_period
        self.spec.samples_per_bin = self._samples_per_bin  # runtime policy recorded in spec
        # NEW runtime flags for clarity (not required by smol, just recorded)
        self.spec.collect_cation_stats = self._collect_cation_stats
        self.spec.production_mode = self._production_mode
        if self._sl_index_map:
            self.spec.sublattice_labels = tuple(sorted(self._sl_index_map.keys()))

        # Additional clean-ups after base init.
        self._entropy_d.clear()
        self._histogram_d.clear()
        self._occurrences_d.clear()
        self._mean_features_d.clear()
        self._bin_samples_d.clear()
        self._bin_sample_hashes_d.clear()
        self._steps_counter = 0

    @property
    def bin_size(self) -> float:
        """Enthalpy bin size."""
        return self._bin_size

    def _sorted_bins(self) -> list[int]:
        # "Visited" is defined as bins with entropy > 0 (matches smol masking).
        return sorted([b for b, ent in self._entropy_d.items() if ent > 0])

    def _as_array(self, m: Mapping[int, float | int]) -> np.ndarray:
        bins = self._sorted_bins()
        return np.asarray([m.get(b, 0) for b in bins])

    @property
    def levels(self) -> np.ndarray:
        """Visited enthalpy levels (bin centers/edges representative) in eV/supercell."""
        bins = self._sorted_bins()
        return np.asarray([self._get_bin_enthalpy(b) for b in bins], dtype=float)

    @property
    def entropy(self) -> np.ndarray:
        """log(dos) on visited levels."""
        return self._as_array(self._entropy_d).astype(float)

    @property
    def dos(self) -> np.ndarray:
        """Density of states on visited levels."""
        ent = self.entropy
        return np.exp(ent - ent.min()) if ent.size else np.asarray([], dtype=float)

    @property
    def histogram(self) -> np.ndarray:
        """Histogram on visited levels."""
        return self._as_array(self._histogram_d).astype(int)

    @property
    def bin_indices(self) -> np.ndarray:
        """Integer bin indices for the visited levels (aligned to anchor=0)."""
        return np.asarray(self._sorted_bins(), dtype=int)

    @property
    def mod_factor(self) -> float:
        return self._m

    # -------------------- helpers -------------------- #

    def _get_bin_id(self, e: float) -> int | float:
        """Get bin index of an enthalpy, snapped to anchor=0.0."""
        if e == np.inf:  # happens at init
            return np.inf
        # floor division wrt global anchor 0.0
        return int(np.floor(e / self._bin_size))

    def _get_bin_enthalpy(self, bin_id: int) -> float:
        """Representative enthalpy from bin index (aligned to anchor=0.0)."""
        return bin_id * self._bin_size

    @staticmethod
    def _hash_occupancy_int32(occ: np.ndarray) -> str:
        """
        Stable, collision-resistant-ish hash for an int32 occupancy vector.
        Uses SHA1 of raw bytes (fast enough for K<=O(10) per bin).
        """
        # Ensure contiguous int32 view
        buf = np.asarray(occ, dtype=np.int32, order="C")
        return hashlib.sha1(buf.tobytes()).hexdigest()

    def _update_cation_stats_for_bin(self, b: int) -> None:
        """Update per-bin cation-count histograms for current occupancy."""
        if not self._collect_cation_stats:
            return
        if self._last_accepted_occupancy is None:
            return
        if not self._sl_index_map:
            # Not configured; nothing to do.
            return

        occ = np.asarray(self._last_accepted_occupancy, dtype=np.int32)
        bin_dict = self._bin_cation_counts.get(b)
        if bin_dict is None:
            bin_dict = {}
            self._bin_cation_counts[b] = bin_dict

        for sl, idx in self._sl_index_map.items():
            # Extract codes on that sublattice and count occurrences per species code
            codes, counts = np.unique(occ[idx], return_counts=True)
            sl_dict = bin_dict.get(sl)
            if sl_dict is None:
                sl_dict = {}
                bin_dict[sl] = sl_dict

            # runtime defenses: unmapped or duplicate labels
            seen_labels_for_this_bin_sl: set[str] = set()
            for code, n_sites in zip(codes.tolist(), counts.tolist()):
                icode = int(code)
                if icode not in self._code_to_label:
                    raise KeyError(
                        f"species_code_to_label has no entry for code {icode} "
                        f"(observed on sublattice '{sl}')."
                    )
                label = self._code_to_label[icode]
                if label in seen_labels_for_this_bin_sl:
                    twin_codes = sorted([int(k) for k, v in self._code_to_label.items() if v == label])
                    raise ValueError(
                        "Duplicate label mapping at runtime: multiple codes map to label "
                        f"'{label}' on sublattice '{sl}'. Offending codes: {twin_codes}"
                    )
                seen_labels_for_this_bin_sl.add(label)

                elem_dict = sl_dict.get(label)
                if elem_dict is None:
                    elem_dict = {}
                    sl_dict[label] = elem_dict
                # Increment the histogram at key "n_sites"
                elem_dict[n_sites] = int(elem_dict.get(n_sites, 0) + 1)

    # -------------------- MC step logic -------------------- #

    def _accept_step(self, occupancy: np.ndarray, step) -> np.ndarray:
        bin_id = self._get_bin_id(self._current_enthalpy)
        new_enthalpy = self._current_enthalpy + self.trace.delta_trace.enthalpy

        new_bin_id = self._get_bin_id(new_enthalpy)
        if not np.isfinite(new_bin_id):
            self.trace.accepted = np.array(False)
            return self.trace.accepted

        # default visited/unvisited bins mimic smol: entropy=0 for unseen bins
        entropy = float(self._entropy_d.get(int(bin_id), 0.0)) if np.isfinite(bin_id) else 0.0
        new_entropy = float(self._entropy_d.get(int(new_bin_id), 0.0))

        # mcusher is initialized by the base __init__; assert for type-checkers.
        assert self.mcusher is not None, "MCUsher is not initialized"
        log_factor = self.mcusher.compute_log_priori_factor(occupancy, step)
        exponent = entropy - new_entropy + log_factor
        self.trace.accepted = np.array(True if exponent >= 0 else exponent > log(self._rng.random()))
        return self.trace.accepted

    def _do_accept_step(self, occupancy: np.ndarray, step):
        """Accept/reject a given step and populate trace and aux states accordingly."""
        occupancy = super()._do_accept_step(occupancy, step)
        self._current_features += self.trace.delta_trace.features
        self._current_enthalpy += self.trace.delta_trace.enthalpy

        # Remember last accepted occupancy for optional per-bin sampling.
        self._last_accepted_occupancy = occupancy.copy()
        return occupancy

    def _do_post_step(self) -> None:
        """Populate histogram/entropy, and update counters accordingly."""
        bin_id = self._get_bin_id(self._current_enthalpy)

        if np.isfinite(bin_id):
            b = int(bin_id)
            # compute cumulative stats
            self._steps_counter += 1
            total = int(self._occurrences_d.get(b, 0))
            prev = self._mean_features_d.get(b)
            if prev is None:
                prev = np.zeros_like(self._current_features)
            self._mean_features_d[b] = (self._current_features + total * prev) / (total + 1)

            # optional: capture up to K UNIQUE occupancy samples per bin
            if self._samples_per_bin > 0 and self._last_accepted_occupancy is not None:
                lst = self._bin_samples_d.get(b)
                if lst is None:
                    lst = []
                    self._bin_samples_d[b] = lst

                if len(lst) < self._samples_per_bin:
                    # compute hash and dedupe
                    h = self._hash_occupancy_int32(self._last_accepted_occupancy)
                    seen = self._bin_sample_hashes_d.get(b)
                    if seen is None:
                        seen = set()
                        self._bin_sample_hashes_d[b] = seen

                    if h not in seen:
                        lst.append([int(x) for x in np.asarray(self._last_accepted_occupancy, dtype=int).tolist()])
                        seen.add(h)
                        # If we've reached the quota, drop the hash set to free memory
                        if len(lst) >= self._samples_per_bin:
                            self._bin_sample_hashes_d.pop(b, None)

            # At the same cadence as histogram updates, update cation stats
            if self._steps_counter % self.update_period == 0:
                # Update cation counts regardless of production mode (so we can collect stats)
                self._update_cation_stats_for_bin(b)

                if not self._production_mode:
                    # update histogram, entropy and occurrences
                    self._entropy_d[b] = float(self._entropy_d.get(b, 0.0) + self._m)
                    self._histogram_d[b] = int(self._histogram_d.get(b, 0) + 1)
                    self._occurrences_d[b] = int(self._occurrences_d.get(b, 0) + 1)

        # fill trace with visited-only views
        self.trace.histogram = np.empty((0,), dtype=int)
        self.trace.occurrences = np.empty((0,), dtype=int)
        self.trace.entropy = np.empty((0,), dtype=float)
        self.trace.cumulative_mean_features = np.empty((0, self._nfeat), dtype=float)
        self.trace.mod_factor = np.array([self._m])

        # flatness check on visited bins only (disabled in production_mode)
        if (not self._production_mode) and (self._steps_counter % self.check_period == 0):
            histogram = self.histogram
            if len(histogram) >= 2 and (histogram > self.flatness * histogram.mean()).all():
                self._histogram_d.clear()
                self._m = self._mod_update(self._m)
                self._mod_updates_buf.append((int(self._steps_counter), float(self._m)))

    # expose and clear the buffers; not serialized in state()
    def pop_mod_updates(self) -> list[tuple[int, float]]:
        ev = self._mod_updates_buf
        self._mod_updates_buf = []
        return ev

    def pop_bin_samples(self) -> dict[int, list[list[int]]]:
        """Return and clear the per-bin UNIQUE occupancy snapshots captured since last call."""
        out = self._bin_samples_d
        self._bin_samples_d = {}
        self._bin_sample_hashes_d = {}  # drop dedupe state on drain
        return out

    def pop_bin_cation_counts(self) -> dict[int, dict[str, dict[str, dict[int, int]]]]:
        """
        Return and clear the per-bin cation-count histograms captured since last call.

        Structure:
          { bin_id: { sublattice_label: { element_label: { n_sites_on_sublattice: visits } } } }
        """
        out = self._bin_cation_counts
        self._bin_cation_counts = {}
        return out

    def compute_initial_trace(self, occupancy: np.ndarray):
        """Compute initial values for sample trace given an occupancy."""
        trace = super().compute_initial_trace(occupancy)
        trace.histogram = np.empty((0,), dtype=int)
        trace.occurrences = np.empty((0,), dtype=int)
        trace.entropy = np.empty((0,), dtype=float)
        trace.cumulative_mean_features = np.empty((0, self._nfeat), dtype=float)
        trace.mod_factor = self._m
        return trace

    def set_aux_state(self, occupancy: np.ndarray, *args, **kwargs):
        """Set the auxiliary occupancies based on an occupancy.

        This is necessary for Wang-Landau to work properly because
        it needs to store the current enthalpy and features.
        """
        features = np.array(self.ensemble.compute_feature_vector(occupancy))
        enthalpy = float(np.dot(features, self.natural_params))
        self._current_features = features
        self._current_enthalpy = enthalpy
        # As above, ensure non-None for type-checkers.
        assert self.mcusher is not None, "MCUsher is not initialized"
        self.mcusher.set_aux_state(occupancy)

    # -------------------- checkpointing API -------------------- #

    def state(self) -> dict:
        """Return a JSON-serializable snapshot of the kernel."""
        bins = self.bin_indices
        occurrences = np.asarray([self._occurrences_d.get(int(b), 0) for b in bins], dtype=int)
        mean_feats = (
            np.asarray([self._mean_features_d[int(b)] for b in bins], dtype=float)
            if bins.size else np.empty((0, self._nfeat), dtype=float)
        )
        return {
            "version": 1,
            "bin_indices": bins.tolist(),
            "entropy": self.entropy.tolist(),
            "histogram": self.histogram.tolist(),
            "occurrences": occurrences.tolist(),
            "mean_features": mean_feats.tolist(),
            "mod_factor": float(self._m),
            "steps_counter": int(self._steps_counter),
            "current_enthalpy": float(self._current_enthalpy),
            "current_features": np.asarray(self._current_features, dtype=float).tolist(),
            "rng_state": self._encode_rng_state(self._rng.bit_generator.state),
            "bin_size": float(self._bin_size),
            # NOTE: bin samples and cation counts are drained via pop_*(), not serialized here.
        }

    def load_state(self, s: dict) -> None:
        """Restore kernel from a state() snapshot."""
        self._entropy_d.clear()
        self._histogram_d.clear()
        self._occurrences_d.clear()
        self._mean_features_d.clear()
        self._bin_samples_d.clear()       # safe
        self._bin_sample_hashes_d.clear() # safe
        self._bin_cation_counts.clear()   # safe

        bins = np.asarray(s["bin_indices"], dtype=int)
        ent = np.asarray(s["entropy"], dtype=float)
        hist = np.asarray(s["histogram"], dtype=int)
        occ = np.asarray(s["occurrences"], dtype=int)
        mfeat = np.asarray(s["mean_features"], dtype=float)

        for i, b in enumerate(bins):
            bi = int(b)
            self._entropy_d[bi] = float(ent[i])
            self._histogram_d[bi] = int(hist[i])
            self._occurrences_d[bi] = int(occ[i])
            self._mean_features_d[bi] = mfeat[i].astype(float, copy=False)

        self._m = float(s["mod_factor"])
        self._steps_counter = int(s["steps_counter"])
        self._current_enthalpy = float(s["current_enthalpy"])
        self._current_features = np.asarray(s["current_features"], dtype=float)
        self._mod_updates_buf = []
        # restore RNG
        self._rng.bit_generator.state = self._decode_rng_state(s["rng_state"])
        # sanity: bin size should match
        if "bin_size" in s and float(s["bin_size"]) != float(self._bin_size):
            raise ValueError(f"State bin_size {s['bin_size']} != kernel bin_size {self._bin_size}")

    # ---------- RNG state (de)serialization helpers ---------- #
    @staticmethod
    def _encode_rng_state(st: Mapping[str, Any]) -> dict[str, Any]:
        """
        Make NumPy BitGenerator state BSON-safe by converting any unsigned 64-bit
        integers to tagged hex strings. Everything else passes through.

        Returns a dict at the top level (Pylance-friendly).
        """
        INT64_MAX = (1 << 63) - 1

        def enc(x: Any) -> Any:
            import numpy as _np
            if isinstance(x, dict):
                return {str(k): enc(v) for k, v in x.items()}
            if isinstance(x, (list, tuple, _np.ndarray)):
                return [enc(v) for v in _np.asarray(x).tolist()]
            if isinstance(x, (_np.integer, int)):
                xi = int(x)
                if xi > INT64_MAX or xi < -INT64_MAX - 1:
                    return {"__u64__": hex(xi & ((1 << 64) - 1))}
                return xi
            return x

        res: Any = enc(dict(st))
        if not isinstance(res, dict):
            raise TypeError("Encoded RNG state must be a dict at the top level.")
        return dict(res)

    @staticmethod
    def _decode_rng_state(st: Mapping[str, Any]) -> dict[str, Any]:
        def dec(x: Any) -> Any:
            if isinstance(x, dict):
                if "__u64__" in x:
                    return int(x["__u64__"], 16)
                return {k: dec(v) for k, v in x.items()}
            if isinstance(x, list):
                return [dec(v) for v in x]
            return x

        res: Any = dec(dict(st))
        if not isinstance(res, dict):
            raise TypeError("Decoded RNG state must be a dict at the top level.")
        return dict(res)
