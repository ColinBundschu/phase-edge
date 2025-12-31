from dataclasses import dataclass
import dataclasses
from enum import Enum
from typing import Any, Mapping
from monty.json import MSONable


class RelaxType(str, Enum):
    STATIC = "static"
    FULL = "full"


class CalcType(str, Enum):
    VASP_MP_GGA = "vasp-mp-gga"
    VASP_MP_24 = "vasp-mp-24"
    MACE_MPA_0 = "MACE-MPA-0"


class SpinType(str, Enum):
    NONMAGNETIC = "nonmagnetic"
    FERROMAGNETIC = "ferromagnetic"
    SUBLATTICES_ANTIALIGNED = "sublattices-antialigned"
    AFM_CHECKERBOARD_WITHIN_SUBLATTICE = "afm-checkerboard-within-sublattice"
    ZERO_INIT_MAGNETIC = "zero-init-magnetic"

@dataclass(frozen=True, slots=True)
class CalcSpec(MSONable):
    _: dataclasses.KW_ONLY
    calculator: str
    relax_type: RelaxType
    spin_type: SpinType
    max_force_eV_per_A: float
    frozen_sublattices: str = ""

    def __post_init__(self) -> None:
        # Make sure frozen_sublattices is in alphabetical order
        sorted_fs = sorted(self.frozen_sublattices_set)
        if self.frozen_sublattices != ",".join(str(idx) for idx in sorted_fs):
            raise ValueError("frozen_sublattices must be a comma-separated string of sorted strings.")

    def as_dict(self) -> dict:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "calculator": self.calculator,
            "relax_type": self.relax_type.value,
            "spin_type": self.spin_type.value,
            "max_force_eV_per_A": self.max_force_eV_per_A,
            "frozen_sublattices": self.frozen_sublattices,
        }
    
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalcSpec":
        return cls(
            calculator=str(d["calculator"]),
            relax_type=RelaxType(d["relax_type"]),
            spin_type=SpinType(d["spin_type"]),
            max_force_eV_per_A=float(d["max_force_eV_per_A"]),
            frozen_sublattices=str(d["frozen_sublattices"]),
        )
    
    @property
    def calculator_info(self) -> dict[str, Any]:
        """
        Interpret the 'calculator' string.

        Semantics:
        - 'NAME'                       -> force_field_name='NAME'
        - 'NAME;EXTRA'                 -> force_field_name='NAME', calc_kwargs['model']='EXTRA'
        Everything before the first ';' is the force-field name. Everything after (if non-empty)
        is passed through as the 'model' kwarg to the calculator. Whitespace is stripped.
        """
        head, sep, tail = self.calculator.partition(";")
        name = head.strip()
        calc_kwargs: dict[str, Any] = {}
        if sep and tail.strip():
            calc_kwargs["model"] = tail.strip()
        return {"calc_type": CalcType(name), "calc_kwargs": calc_kwargs}
    
    @property
    def calc_type(self) -> CalcType:
        return self.calculator_info["calc_type"]


    @property
    def calc_kwargs(self) -> dict[str, Any]:
        return self.calculator_info["calc_kwargs"]


    @property
    def frozen_sublattices_set(self) -> set[str]:
        """
        Returns the set of frozen sublattice indices, which are alphabetical strings separated by commas
        """
        if not self.frozen_sublattices:
            return set()
        return set(label.strip() for label in self.frozen_sublattices.split(","))
