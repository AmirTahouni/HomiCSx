"""Run the Abaqus side of the homogeneous shear-relaxation validation.

Run from ``validation/abaqus`` with::

    abaqus cae noGUI=run_abaqus_viscoelastic.py

The script is compatible with Abaqus 2022's Python runtime.
"""

from __future__ import print_function

import json
import os

from abaqus import mdb
from abaqusConstants import (
    ANALYSIS,
    CPE6H,
    FREE,
    ISOTROPIC,
    NEO_HOOKE,
    OFF,
    ON,
    PERCENTAGE,
    PRONY,
    STANDARD,
    TIME,
    TRI,
)
import mesh
from odbAccess import openOdb

from run_abaqus_canonical import (
    add_periodic_constraints,
    create_partitioned_part,
    integrated_volume,
    volume_average,
)


HERE = os.getcwd()
WORK = os.path.join(HERE, "canonical_work")


def load_case():
    with open(os.path.join(HERE, "viscoelastic_case.json"), "r") as stream:
        return json.load(stream)


def main():
    case = load_case()
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    os.chdir(WORK)
    model_name = str("M_" + case["case_id"])
    model = mdb.Model(name=model_name)
    lx = case["domain"]["lx"]
    ly = case["domain"]["ly"]
    part = create_partitioned_part(model, "RVE", lx, ly, [])

    equilibrium = case["equilibrium_material"]
    young = equilibrium["young_modulus"]
    poisson = equilibrium["poisson_ratio"]
    equilibrium_mu = young / (2.0 * (1.0 + poisson))
    equilibrium_bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    branch_moduli = case["branch_shear_moduli"]
    instantaneous_mu = equilibrium_mu
    for branch_modulus in branch_moduli:
        instantaneous_mu += branch_modulus

    solid = model.Material(name="VISCOELASTIC_SOLID")
    solid.Hyperelastic(
        materialType=ISOTROPIC,
        testData=OFF,
        type=NEO_HOOKE,
        # Abaqus 2022 writes hyperelastic constants as long-term moduli when
        # time-domain viscoelasticity is attached. The Prony ratios below then
        # recover the requested instantaneous modulus.
        table=((0.5 * equilibrium_mu, 2.0 / equilibrium_bulk),),
    )
    prony_rows = []
    for index in range(len(branch_moduli)):
        prony_rows.append((
            branch_moduli[index] / instantaneous_mu,
            0.0,
            case["relaxation_times"][index],
        ))
    solid.Viscoelastic(domain=TIME, time=PRONY, table=tuple(prony_rows))
    model.HomogeneousSolidSection(
        name="SECTION", material="VISCOELASTIC_SOLID", thickness=1.0
    )
    part.Set(name="ALL", faces=part.faces[:])
    part.SectionAssignment(region=part.sets["ALL"], sectionName="SECTION")
    part.setMeshControls(regions=part.faces[:], elemShape=TRI, technique=FREE)
    part.setElementType(
        regions=(part.faces[:],),
        elemTypes=(mesh.ElemType(elemCode=CPE6H, elemLibrary=STANDARD),),
    )
    part.seedPart(
        size=case["mesh"]["abaqus_size"], deviationFactor=0.1, minSizeFactor=0.1
    )
    part.generateMesh()
    gamma = case["gamma12"]
    assembly, jump_x, jump_y = add_periodic_constraints(
        model, part, lx, ly, 0.0, 0.0, gamma, 0.0
    )
    dt = case["time_increment"]
    count = int(round(case["end_time"] / dt))
    # Apply the shear in a negligible-duration preload step, then hold it in
    # the visco step. Applying it directly in RELAXATION would make Abaqus ramp
    # the displacement over the first 0.05-time increment.
    model.StaticStep(
        name="APPLY",
        previous="Initial",
        timePeriod=1.0e-8,
        initialInc=1.0e-8,
        minInc=1.0e-8,
        maxInc=1.0e-8,
        nlgeom=ON,
    )
    model.ViscoStep(
        name="RELAXATION",
        previous="APPLY",
        timePeriod=case["end_time"],
        cetol=1.0e-3,
        initialInc=dt,
        minInc=dt,
        maxInc=dt,
        maxNumInc=count + 1,
        nlgeom=ON,
    )
    model.DisplacementBC(
        name="MACRO_X",
        createStepName="APPLY",
        region=assembly.sets["RP_X"],
        u1=jump_x[0],
        u2=jump_x[1],
    )
    model.DisplacementBC(
        name="MACRO_Y",
        createStepName="APPLY",
        region=assembly.sets["RP_Y"],
        u1=jump_y[0],
        u2=jump_y[1],
    )
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "IVOL", "EVOL"), frequency=1
    )
    job_name = str("viscoelastic_" + case["case_id"])
    job = mdb.Job(
        name=job_name,
        model=model_name,
        type=ANALYSIS,
        memory=90,
        memoryUnits=PERCENTAGE,
        numCpus=1,
        numDomains=1,
    )
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    odb_path = os.path.join(WORK, job_name + ".odb")
    if not os.path.exists(odb_path):
        raise RuntimeError("Abaqus did not produce {}".format(odb_path))

    odb = openOdb(odb_path, readOnly=True)
    history = []
    for frame in odb.steps["RELAXATION"].frames:
        if frame.frameValue <= 0.0:
            continue
        stress, initial_volume = volume_average(frame, "S", (3,))
        current_volume = integrated_volume(frame, "EVOL")
        history.append(
            {
                "time": float(frame.frameValue),
                "macro_f12": gamma,
                "macro_p12": stress[0],
                "mean_j": current_volume / initial_volume,
            }
        )
    odb.close()
    if len(history) != count or abs(history[-1]["time"] - case["end_time"]) > 1.0e-10:
        raise RuntimeError(
            "incomplete Abaqus history: {} frames ending at {}".format(
                len(history), history[-1]["time"] if history else None
            )
        )
    output = {
        "schema_version": case["schema_version"],
        "case_id": case["case_id"],
        "abaqus_release": "2022",
        "history": history,
        "notes": {
            "macro_p12": "volume-averaged Cauchy S12; equal to P12 for isochoric simple shear",
            "prony_shear_ratios": [value / instantaneous_mu for value in branch_moduli],
            "prony_bulk_ratios": [0.0 for unused in branch_moduli],
        },
    }
    output_path = os.path.join(HERE, "viscoelastic_abaqus_results.json")
    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, separators=(",", ": "))
        stream.write("\n")
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
