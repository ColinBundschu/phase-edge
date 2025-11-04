
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from pymatgen.core import Structure

from phaseedge.schemas.calc_spec import CalcSpec
from phaseedge.storage.store import exists_unique, lookup_total_energy_eV, lookup_unique


@dataclass(frozen=True)
class CETrainRef:
    set_id: str
    occ_key: str
    calc_spec: CalcSpec
    structure: Structure

    def lookup_energy(self) -> float:
        energy = lookup_total_energy_eV(
            set_id=self.set_id,
            occ_key=self.occ_key,
            calc_spec=self.calc_spec,
        )
        if energy is None:
            raise KeyError(f"Could not find total energy for train_ref: {self}")
        return energy
    
    def as_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "occ_key": self.occ_key,
            "calc_spec": self.calc_spec.as_dict(),
            "structure": self.structure.as_dict(),
        }
    
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CETrainRef":
        return cls(
            set_id=d["set_id"],
            occ_key=d["occ_key"],
            calc_spec=CalcSpec.from_dict(d["calc_spec"]),
            structure=Structure.from_dict(d["structure"]),
        )


@dataclass(frozen=True)
class Dataset:
    train_refs: list[CETrainRef]

    @property
    def key(self) -> str:
        sorted_refs = sorted(
            self.train_refs,
            key=lambda r: (r.set_id, r.occ_key, r.calc_spec.calculator,
                           r.calc_spec.relax_type.value, r.calc_spec.frozen_sublattices),
        )
        payload = {
            "train_refs": [
                {
                    "set_id": r.set_id,
                    "occ_key": r.occ_key,
                    "calc_spec": r.calc_spec.as_dict(),
                }
                for r in sorted_refs
            ]
        }
        # Canonical JSON: compact separators, sorted keys not needed for lists, but stable anyway
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


    @property
    def exists(self) -> bool:
        criteria = {"output.kind": "CETrainRef_dataset", "output.dataset_key": self.key}
        return exists_unique(criteria=criteria)


    @property
    def jobflow_output(self) -> dict[str, Any]:
        output = {"dataset_key": self.key}
        if not self.exists:
            output = output | {"train_refs": self.train_refs, "kind": "CETrainRef_dataset"}
        return output


    @classmethod
    def from_key(cls, dataset_key: str) -> "Dataset":
        criteria = {"output.kind": "CETrainRef_dataset", "output.dataset_key": dataset_key}
        dataset = lookup_unique(criteria=criteria)
        if dataset is None:
            raise KeyError(f"No CETrainRef_dataset found for dataset_key={dataset_key!r}")
        train_refs = []
        for ref in dataset["train_refs"]:
            train_refs.append(CETrainRef.from_dict(ref))
        return cls(train_refs=train_refs)
