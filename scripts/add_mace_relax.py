import argparse
from pathlib import Path
from typing import Any, cast

from jobflow.core.job import Job
from jobflow.core.flow import Flow
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow
from pymatgen.core import Structure

from atomate2.forcefields.jobs import ForceFieldRelaxMaker

from phaseedge.science.prototypes import make_prototype, PrototypeName
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor


def load_structure(
    structure_path: str | None,
    *,
    prototype: PrototypeName | None,
    prototype_params: dict[str, Any] | None,
) -> Structure:
    if structure_path:
        path = Path(structure_path)
        if not path.exists():
            raise FileNotFoundError(f"No such structure file: {path}")
        if path.suffix.lower() in {".vasp", ".poscar"} or path.name.upper() in {"POSCAR", "CONTCAR"}:
            return Structure.from_file(str(path))
        atoms = ase_read(str(path))
        return AseAtomsAdaptor.get_structure(atoms) # pyright: ignore[reportArgumentType]

    if prototype is None:
        raise ValueError("Provide either --structure or --prototype.")
    atoms = make_prototype(prototype, **(prototype_params or {}))
    return AseAtomsAdaptor.get_structure(atoms) # pyright: ignore[reportArgumentType]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Submit a MACE force-field relaxation via atomate2 → FireWorks."
    )
    p.add_argument("--launchpad", required=True, help="Path to my_launchpad.yaml")
    p.add_argument("--structure", help="Path to structure (POSCAR/CIF/etc.).")
    p.add_argument("--prototype", choices=["rocksalt"])
    p.add_argument("--a", type=float, help="Lattice parameter for prototype (e.g., 4.3).")
    p.add_argument("--relax-cell", action="store_true", help="Allow cell relaxation.")
    p.add_argument("--max-steps", type=int, default=300, help="Optimizer steps.")
    p.add_argument("--category", default="gpu", help="FW category (default: gpu).")
    args = p.parse_args()

    proto_params = {"a": args.a} if args.prototype and args.a else None
    structure = load_structure(args.structure, prototype=args.prototype, prototype_params=proto_params)

    maker = ForceFieldRelaxMaker(
        force_field_name="MACE-MPA-0",
        relax_cell=args.relax_cell,
        steps=args.max_steps,  # add "fmax": 0.02 if desired
        calculator_kwargs={"default_dtype": "float64"}
    )
    job_or_flow = cast(Job, maker.make(structure))
    flow = job_or_flow if isinstance(job_or_flow, Flow) else Flow([job_or_flow])

    # Convert to FireWorks Workflow first…
    wf = flow_to_workflow(flow)

    # inject the category on every Firework spec.
    for fw in wf.fws:
        fw.spec["_category"] = args.category

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)
    print(f"Workflow added. id={wf_id}. Category='{args.category}'.")
    print("Use qlaunch with your GPU qadapter/fworker to run it.")


if __name__ == "__main__":
    main()
