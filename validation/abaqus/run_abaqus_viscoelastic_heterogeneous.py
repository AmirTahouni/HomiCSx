"""Run heterogeneous Abaqus shear-relaxation validation cases.

Run from ``validation/abaqus`` with::

    abaqus cae noGUI=run_abaqus_viscoelastic_heterogeneous.py
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
import interaction
import load
import material
import mesh
import step
from odbAccess import openOdb

from run_abaqus_canonical import (
    add_periodic_constraints,
    classify_faces,
    create_partitioned_part,
    face_array,
    integrated_volume,
    volume_average,
)


HERE = os.getcwd()
WORK = os.path.join(HERE, "canonical_work")


def load_case():
    with open(os.path.join(HERE, "viscoelastic_case.json"), "r") as stream:
        return json.load(stream)


def add_materials_and_sections(model, part, case, geometry_case):
    matrix_faces, inclusion_faces = classify_faces(part, geometry_case["circles"])
    part.Set(name="MATRIX", faces=face_array(part, matrix_faces))
    part.Set(name="INCLUSION", faces=face_array(part, inclusion_faces))

    equilibrium = case["equilibrium_material"]
    young = equilibrium["young_modulus"]
    poisson = equilibrium["poisson_ratio"]
    equilibrium_mu = young / (2.0 * (1.0 + poisson))
    equilibrium_bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    branch_moduli = case["branch_shear_moduli"]
    instantaneous_mu = equilibrium_mu
    for branch_modulus in branch_moduli:
        instantaneous_mu += branch_modulus

    matrix_material = model.Material(name="VISCOELASTIC_MATRIX")
    matrix_material.Hyperelastic(
        materialType=ISOTROPIC,
        testData=OFF,
        type=NEO_HOOKE,
        table=((0.5 * equilibrium_mu, 2.0 / equilibrium_bulk),),
    )
    prony_rows = []
    for index in range(len(branch_moduli)):
        prony_rows.append(
            (
                branch_moduli[index] / instantaneous_mu,
                0.0,
                case["relaxation_times"][index],
            )
        )
    matrix_material.Viscoelastic(domain=TIME, time=PRONY, table=tuple(prony_rows))

    inclusion = case["inclusion_material"]
    inclusion_young = inclusion["young_modulus"]
    inclusion_poisson = inclusion["poisson_ratio"]
    inclusion_mu = inclusion_young / (2.0 * (1.0 + inclusion_poisson))
    inclusion_bulk = inclusion_young / (3.0 * (1.0 - 2.0 * inclusion_poisson))
    inclusion_material = model.Material(name="HYPERELASTIC_INCLUSION")
    inclusion_material.Hyperelastic(
        materialType=ISOTROPIC,
        testData=OFF,
        type=NEO_HOOKE,
        table=((0.5 * inclusion_mu, 2.0 / inclusion_bulk),),
    )

    model.HomogeneousSolidSection(
        name="MATRIX_SECTION", material="VISCOELASTIC_MATRIX", thickness=1.0
    )
    model.HomogeneousSolidSection(
        name="INCLUSION_SECTION", material="HYPERELASTIC_INCLUSION", thickness=1.0
    )
    part.SectionAssignment(region=part.sets["MATRIX"], sectionName="MATRIX_SECTION")
    part.SectionAssignment(
        region=part.sets["INCLUSION"], sectionName="INCLUSION_SECTION"
    )


def run_one(case, geometry_case):
    model_name = str("M_" + geometry_case["case_id"])
    model = mdb.Model(name=model_name)
    lx = case["domain"]["lx"]
    ly = case["domain"]["ly"]
    part = create_partitioned_part(
        model, "RVE", lx, ly, geometry_case["circles"]
    )
    add_materials_and_sections(model, part, case, geometry_case)
    part.setMeshControls(regions=part.faces[:], elemShape=TRI, technique=FREE)
    part.setElementType(
        regions=(part.faces[:],),
        elemTypes=(mesh.ElemType(elemCode=CPE6H, elemLibrary=STANDARD),),
    )
    part.seedPart(
        size=case["mesh"]["abaqus_size"], deviationFactor=0.1, minSizeFactor=0.1
    )
    part.generateMesh()
    element_count = len(part.elements)
    gamma = case["gamma12"]
    assembly, jump_x, jump_y = add_periodic_constraints(
        model, part, lx, ly, 0.0, 0.0, gamma, 0.0
    )
    dt = case["time_increment"]
    count = int(round(case["end_time"] / dt))
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
        name="MACRO_X", createStepName="APPLY", region=assembly.sets["RP_X"],
        u1=jump_x[0], u2=jump_x[1]
    )
    model.DisplacementBC(
        name="MACRO_Y", createStepName="APPLY", region=assembly.sets["RP_Y"],
        u1=jump_y[0], u2=jump_y[1]
    )
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "IVOL", "EVOL"), frequency=1
    )
    job_name = str("viscoelastic_" + geometry_case["case_id"])
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
    if len(history) != count or abs(history[-1]["time"] - case["end_time"]) > 1.0e-6:
        raise RuntimeError("incomplete Abaqus history for {}".format(job_name))
    del mdb.models[model_name]
    print("Completed {} with {} elements".format(job_name, element_count))
    return {"case_id": geometry_case["case_id"], "history": history}


def main():
    case = load_case()
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    os.chdir(WORK)
    results = [run_one(case, geometry_case) for geometry_case in case["heterogeneous_cases"]]
    output = {"schema_version": case["schema_version"], "abaqus_release": "2022", "cases": results}
    output_path = os.path.join(HERE, "viscoelastic_abaqus_heterogeneous_results.json")
    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, separators=(",", ": "))
        stream.write("\n")
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
