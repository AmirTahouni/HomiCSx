"""Run the Abaqus side of the three-dimensional shear-relaxation check.

Run from ``validation/abaqus`` with::

    abaqus cae noGUI=run_abaqus_viscoelastic_3d.py

The script is compatible with Abaqus 2022's Python runtime.  Affine
displacements are prescribed on the complete boundary.  For this homogeneous
patch test they produce the same uniform solution as periodic constraints,
while independently exercising the 3D constitutive and tensor paths.
"""

from __future__ import print_function

import json
import os

from abaqus import mdb
from abaqusConstants import (
    ANALYSIS,
    C3D10H,
    DEFORMABLE_BODY,
    FREE,
    ISOTROPIC,
    NEO_HOOKE,
    OFF,
    ON,
    PERCENTAGE,
    PRONY,
    STANDARD,
    TET,
    THREE_D,
    TIME,
)
import interaction
import load
import material
import mesh
import step
from odbAccess import openOdb

from run_abaqus_canonical import integrated_volume, volume_average


HERE = os.getcwd()
WORK = os.path.join(HERE, "canonical_work")


def load_case():
    with open(os.path.join(HERE, "viscoelastic_case.json"), "r") as stream:
        return json.load(stream)


def main():
    manifest = load_case()
    case = manifest["three_dimensional_case"]
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    os.chdir(WORK)

    model_name = str("M_" + case["case_id"])
    model = mdb.Model(name=model_name)
    lx = case["domain"]["lx"]
    ly = case["domain"]["ly"]
    lz = case["domain"]["lz"]
    sketch = model.ConstrainedSketch(name="BOX_SKETCH", sheetSize=2.0 * max(lx, ly, lz))
    sketch.rectangle(point1=(0.0, 0.0), point2=(lx, ly))
    part = model.Part(name="RVE", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=sketch, depth=lz)

    equilibrium = manifest["equilibrium_material"]
    young = equilibrium["young_modulus"]
    poisson = equilibrium["poisson_ratio"]
    equilibrium_mu = young / (2.0 * (1.0 + poisson))
    equilibrium_bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    branch_moduli = manifest["branch_shear_moduli"]
    instantaneous_mu = equilibrium_mu + sum(branch_moduli)

    solid = model.Material(name="VISCOELASTIC_SOLID")
    solid.Hyperelastic(
        materialType=ISOTROPIC,
        testData=OFF,
        type=NEO_HOOKE,
        table=((0.5 * equilibrium_mu, 2.0 / equilibrium_bulk),),
    )
    solid.Viscoelastic(
        domain=TIME,
        time=PRONY,
        table=tuple(
            (branch_moduli[index] / instantaneous_mu, 0.0, manifest["relaxation_times"][index])
            for index in range(len(branch_moduli))
        ),
    )
    model.HomogeneousSolidSection(name="SECTION", material="VISCOELASTIC_SOLID")
    part.Set(name="ALL", cells=part.cells[:])
    part.SectionAssignment(region=part.sets["ALL"], sectionName="SECTION")
    part.setMeshControls(regions=part.cells[:], elemShape=TET, technique=FREE)
    part.setElementType(
        regions=(part.cells[:],),
        elemTypes=(mesh.ElemType(elemCode=C3D10H, elemLibrary=STANDARD),),
    )
    part.seedPart(size=case["abaqus_size"], deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    assembly = model.rootAssembly
    instance = assembly.Instance(name="RVE-1", part=part, dependent=ON)
    dt = case["time_increment"]
    count = int(round(case["end_time"] / dt))
    model.StaticStep(
        name="APPLY", previous="Initial", timePeriod=1.0e-8,
        initialInc=1.0e-8, minInc=1.0e-8, maxInc=1.0e-8, nlgeom=ON,
    )
    model.ViscoStep(
        name="RELAXATION", previous="APPLY", timePeriod=case["end_time"],
        cetol=1.0e-3, initialInc=dt, minInc=dt, maxInc=dt,
        maxNumInc=count + 1, nlgeom=ON,
    )

    gamma = case["gamma12"]
    tolerance = 1.0e-8 * max(lx, ly, lz)
    boundary_count = 0
    for node in instance.nodes:
        x, y, z = node.coordinates
        if (x < tolerance or x > lx - tolerance or y < tolerance or
                y > ly - tolerance or z < tolerance or z > lz - tolerance):
            set_name = "BOUNDARY_NODE_{:06d}".format(node.label)
            assembly.Set(
                name=set_name,
                nodes=instance.nodes.sequenceFromLabels((node.label,)),
            )
            model.DisplacementBC(
                name="AFFINE_{:06d}".format(node.label),
                createStepName="APPLY",
                region=assembly.sets[set_name],
                u1=gamma * y,
                u2=0.0,
                u3=0.0,
            )
            boundary_count += 1
    if boundary_count == 0:
        raise RuntimeError("no boundary nodes were selected")

    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "IVOL", "EVOL"), frequency=1
    )
    job_name = str("viscoelastic_" + case["case_id"])
    job = mdb.Job(
        name=job_name, model=model_name, type=ANALYSIS, memory=90,
        memoryUnits=PERCENTAGE, numCpus=1, numDomains=1,
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
        history.append({
            "time": float(frame.frameValue),
            "macro_f12": gamma,
            "macro_p12": stress[0],
            "mean_j": current_volume / initial_volume,
        })
    odb.close()
    if len(history) != count or abs(history[-1]["time"] - case["end_time"]) > 1.0e-10:
        raise RuntimeError("incomplete Abaqus history")

    output = {
        "schema_version": manifest["schema_version"],
        "case_id": case["case_id"],
        "abaqus_release": "2022",
        "history": history,
        "notes": {
            "boundary_condition": "affine displacement on all exterior nodes",
            "macro_p12": "volume-averaged Cauchy S12; equal to P12 for isochoric simple shear",
            "boundary_node_count": boundary_count,
        },
    }
    output_path = os.path.join(HERE, "viscoelastic_abaqus_3d_results.json")
    with open(output_path, "w") as stream:
        json.dump(output, stream, indent=2, separators=(",", ": "))
        stream.write("\n")
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
