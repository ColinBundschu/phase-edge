from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from smol.cofe import ClusterExpansion
from smol.moca.ensemble import Ensemble

import numpy as np


def rehydrate_ensemble_by_ce_key(ce_key: str) -> Ensemble:
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")
    ce = ClusterExpansion.from_dict(doc["payload"])
    sx, sy, sz = (int(x) for x in doc["supercell_diag"])
    sc_matrix = np.diag([sx, sy, sz])
    return Ensemble.from_cluster_expansion(ce, supercell_matrix=sc_matrix)
