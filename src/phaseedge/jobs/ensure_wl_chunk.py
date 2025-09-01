from dataclasses import dataclass, asdict
from typing import Any, Mapping, Dict

from jobflow.core.job import job
from monty.json import MSONable

from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.sampling.wl_chunk_driver import WLChunkSpec, run_wl_chunk


@dataclass(frozen=True)
class WLChunkEnsureSpec(MSONable):
    """Job spec to extend a WL chain by steps_to_run."""
    run_spec: WLSamplerSpec
    wl_key: str
    steps_to_run: int

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({"@module": self.__class__.__module__, "@class": self.__class__.__name__})
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WLChunkEnsureSpec":
        d = {k: v for k, v in d.items() if not k.startswith("@")}
        if not isinstance(d.get("run_spec"), WLSamplerSpec):
            d["run_spec"] = WLSamplerSpec(**d["run_spec"])
        return cls(**d)  # type: ignore[arg-type]


@job
def ensure_wl_chunk(spec: WLChunkEnsureSpec) -> Dict[str, Any]:
    """Extend the WL chain by steps_to_run, idempotently. Fails if not on tip."""
    chunk_spec = WLChunkSpec(run_spec=spec.run_spec,
                             wl_key=spec.wl_key,
                             steps_to_run=spec.steps_to_run)
    return run_wl_chunk(chunk_spec)
