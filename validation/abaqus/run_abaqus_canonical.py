"""Build, solve, and extract the Abaqus side of the canonical suite.

This file intentionally remains compatible with the Python version bundled
with Abaqus 2022. Run it from this directory with::

    abaqus cae noGUI=run_abaqus_canonical.py

The script creates its transient job files in ``canonical_work`` and writes
``canonical_abaqus_results.json`` beside this file.
"""

from __future__ import print_function

import json
import os

from abaqus import mdb
from abaqusConstants import (
    ANALYSIS,
    CPE6,
    CPE6H,
    DEFORMABLE_BODY,
    FREE,
    ISOTROPIC,
    NEO_HOOKE,
    OFF,
    ON,
    PERCENTAGE,
    STANDARD,
    TRI,
    TWO_D_PLANAR,
)
import mesh
from odbAccess import openOdb


# Abaqus/CAE executes noGUI scripts without defining ``__file__``. The
# documented command is therefore intentionally run from this directory.
HERE = os.getcwd()
WORK = os.path.join(HERE, "canonical_work")


def read_manifest():
    with open(os.path.join(HERE, "canonical_cases.json"), "r") as stream:
        return json.load(stream)


def face_array(part, face_list):
    result = part.faces[0:0]
    for face in face_list:
        result = result + part.faces[face.index : face.index + 1]
    return result


def classify_faces(part, circles):
    particle = []
    matrix = []
    for face in part.faces:
        point = face.pointOn[0]
        inside = False
        for circle in circles:
            cx, cy = circle["center"]
            radius = circle["radius"]
            if (point[0] - cx) ** 2 + (point[1] - cy) ** 2 < (radius + 1.0e-9) ** 2:
                inside = True
                break
        (particle if inside else matrix).append(face)
    if not particle or not matrix:
        raise RuntimeError("failed to classify matrix and inclusion faces")
    return matrix, particle


def create_partitioned_part(model, name, lx, ly, circles):
    sketch = model.ConstrainedSketch(name="base", sheetSize=5.0 * max(lx, ly))
    sketch.rectangle(point1=(0.0, 0.0), point2=(lx, ly))
    part = model.Part(name=name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    del model.sketches["base"]
    if circles:
        partition = model.ConstrainedSketch(
            name="partition", sheetSize=5.0 * max(lx, ly)
        )
        for circle in circles:
            cx, cy = circle["center"]
            radius = circle["radius"]
            partition.CircleByCenterPerimeter(
                center=(cx, cy), point1=(cx + radius, cy)
            )
        part.PartitionFaceBySketch(faces=part.faces[:], sketch=partition)
        del model.sketches["partition"]
    return part


def nearest_pairs(side_a, side_b, coordinate_index, tolerance):
    if len(side_a) != len(side_b):
        raise RuntimeError(
            "opposite boundary node counts differ: {} versus {}".format(
                len(side_a), len(side_b)
            )
        )
    pairs = []
    unused = list(side_b)
    for node_a in sorted(side_a, key=lambda node: node.coordinates[coordinate_index]):
        node_b = min(
            unused,
            key=lambda node: abs(
                node.coordinates[coordinate_index] - node_a.coordinates[coordinate_index]
            ),
        )
        error = abs(
            node_b.coordinates[coordinate_index] - node_a.coordinates[coordinate_index]
        )
        if error > tolerance:
            raise RuntimeError(
                "opposite boundary nodes do not match; tangential error={}".format(error)
            )
        pairs.append((node_a, node_b))
        unused.remove(node_b)
    return pairs


def singleton_node_set(assembly, instance, name, node):
    assembly.Set(name=name, nodes=instance.nodes.sequenceFromLabels((node.label,)))


def add_periodic_constraints(model, part, lx, ly, h11, h22, h12, h21):
    assembly = model.rootAssembly
    instance = assembly.Instance(name="RVE-1", part=part, dependent=ON)
    tolerance = 1.0e-7 * max(lx, ly)
    left = [node for node in instance.nodes if abs(node.coordinates[0]) <= tolerance]
    right = [
        node for node in instance.nodes if abs(node.coordinates[0] - lx) <= tolerance
    ]
    bottom = [node for node in instance.nodes if abs(node.coordinates[1]) <= tolerance]
    top = [
        node for node in instance.nodes if abs(node.coordinates[1] - ly) <= tolerance
    ]
    lr_pairs = nearest_pairs(left, right, 1, tolerance)
    # The top-right equation follows from the other corner relations.
    bottom_reduced = [
        node for node in bottom if abs(node.coordinates[0] - lx) > tolerance
    ]
    top_reduced = [node for node in top if abs(node.coordinates[0] - lx) > tolerance]
    bt_pairs = nearest_pairs(bottom_reduced, top_reduced, 0, tolerance)

    rp_x = assembly.ReferencePoint(point=(1.15 * lx, 0.0, 0.0))
    rp_y = assembly.ReferencePoint(point=(0.0, 1.15 * ly, 0.0))
    assembly.Set(name="RP_X", referencePoints=(assembly.referencePoints[rp_x.id],))
    assembly.Set(name="RP_Y", referencePoints=(assembly.referencePoints[rp_y.id],))

    for index, pair in enumerate(lr_pairs):
        lname = "L_{:05d}".format(index)
        rname = "R_{:05d}".format(index)
        singleton_node_set(assembly, instance, lname, pair[0])
        singleton_node_set(assembly, instance, rname, pair[1])
        for dof in (1, 2):
            model.Equation(
                name="PBC_X_{:05d}_{}".format(index, dof),
                terms=((1.0, rname, dof), (-1.0, lname, dof), (-1.0, "RP_X", dof)),
            )
    for index, pair in enumerate(bt_pairs):
        bname = "B_{:05d}".format(index)
        tname = "T_{:05d}".format(index)
        singleton_node_set(assembly, instance, bname, pair[0])
        singleton_node_set(assembly, instance, tname, pair[1])
        for dof in (1, 2):
            model.Equation(
                name="PBC_Y_{:05d}_{}".format(index, dof),
                terms=((1.0, tname, dof), (-1.0, bname, dof), (-1.0, "RP_Y", dof)),
            )

    origin = min(
        instance.nodes,
        key=lambda node: node.coordinates[0] ** 2 + node.coordinates[1] ** 2,
    )
    singleton_node_set(assembly, instance, "ORIGIN", origin)
    model.DisplacementBC(
        name="FIX_ORIGIN",
        createStepName="Initial",
        region=assembly.sets["ORIGIN"],
        u1=0.0,
        u2=0.0,
    )
    return assembly, (h11 * lx, h21 * lx), (h12 * ly, h22 * ly)


def integration_key(value):
    return (
        value.instance.name,
        value.elementLabel,
        getattr(value, "integrationPoint", None),
    )


def volume_average(frame, field_name, components, weight_name="IVOL"):
    field = frame.fieldOutputs[field_name]
    volumes = frame.fieldOutputs[weight_name]
    volume_by_point = dict(
        (integration_key(value), float(value.data)) for value in volumes.values
    )
    totals = [0.0 for unused in components]
    total_volume = 0.0
    for value in field.values:
        weight = volume_by_point[integration_key(value)]
        total_volume += weight
        for index, component in enumerate(components):
            datum = value.data
            totals[index] += weight * float(datum if component is None else datum[component])
    return [value / total_volume for value in totals], total_volume


def integrated_volume(frame, field_name):
    total = 0.0
    for value in frame.fieldOutputs[field_name].values:
        total += float(value.data)
    return total


def extract_linear(odb_path, applied_strain):
    odb = openOdb(odb_path, readOnly=True)
    frame = odb.steps["LOAD"].frames[-1]
    stress, unused_volume = volume_average(frame, "S", (0, 1, 3))
    energy, unused_volume = volume_average(frame, "SENER", (None,))
    odb.close()
    return {
        "macro_strain": list(applied_strain),
        "macro_stress": stress,
        "macro_energy": energy[0],
    }


def make_linear_model(case, manifest, probe_name, probe):
    model_name = "M_{}_{}".format(case["case_id"], probe_name)
    model = mdb.Model(name=model_name)
    lx = manifest["domain"]["lx"]
    ly = manifest["domain"]["ly"]
    part = create_partitioned_part(model, "RVE", lx, ly, case["circles"])
    matrix_faces, particle_faces = classify_faces(part, case["circles"])
    part.Set(name="MATRIX", faces=face_array(part, matrix_faces))
    part.Set(name="INCLUSION", faces=face_array(part, particle_faces))
    materials = manifest["linear_materials"]
    matrix_data = materials["matrix"]
    inclusion_data = materials[case["inclusion_material"]]
    matrix_material = model.Material(name="MATRIX")
    matrix_material.Elastic(
        table=((matrix_data["young_modulus"], matrix_data["poisson_ratio"]),)
    )
    inclusion_material = model.Material(name="INCLUSION")
    inclusion_material.Elastic(
        table=((inclusion_data["young_modulus"], inclusion_data["poisson_ratio"]),)
    )
    model.HomogeneousSolidSection(name="MATRIX_SECTION", material="MATRIX", thickness=1.0)
    model.HomogeneousSolidSection(
        name="INCLUSION_SECTION", material="INCLUSION", thickness=1.0
    )
    part.SectionAssignment(region=part.sets["MATRIX"], sectionName="MATRIX_SECTION")
    part.SectionAssignment(
        region=part.sets["INCLUSION"], sectionName="INCLUSION_SECTION"
    )
    part.setMeshControls(regions=part.faces[:], elemShape=TRI, technique=FREE)
    part.setElementType(
        regions=(part.faces[:],),
        elemTypes=(mesh.ElemType(elemCode=CPE6, elemLibrary=STANDARD),),
    )
    part.seedPart(
        size=manifest["mesh"]["abaqus_size"], deviationFactor=0.1, minSizeFactor=0.1
    )
    part.generateMesh()
    element_count = len(part.elements)
    eps11, eps22, gamma12 = probe
    assembly, jump_x, jump_y = add_periodic_constraints(
        model, part, lx, ly, eps11, eps22, 0.5 * gamma12, 0.5 * gamma12
    )
    model.StaticStep(name="LOAD", previous="Initial", nlgeom=OFF)
    model.DisplacementBC(
        name="MACRO_X",
        createStepName="LOAD",
        region=assembly.sets["RP_X"],
        u1=jump_x[0],
        u2=jump_x[1],
    )
    model.DisplacementBC(
        name="MACRO_Y",
        createStepName="LOAD",
        region=assembly.sets["RP_Y"],
        u1=jump_y[0],
        u2=jump_y[1],
    )
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "E", "SENER", "IVOL", "EVOL")
    )
    return model_name, element_count


def solve_job(model_name, job_name):
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
    return odb_path


def stiffness_from_probes(probes, magnitude):
    columns = []
    for name in ("eps11", "eps22", "gamma12"):
        columns.append([value / magnitude for value in probes[name]["macro_stress"]])
    return [[columns[column][row] for column in range(3)] for row in range(3)]


def run_linear_cases(manifest):
    results = []
    for case in manifest["linear_cases"]:
        probes = {}
        for probe_name in ("eps11", "eps22", "gamma12"):
            probe = manifest["macro_strain_probes"][probe_name]
            model_name, element_count = make_linear_model(
                case, manifest, probe_name, probe
            )
            job_name = "canonical_{}_{}".format(case["case_id"], probe_name)
            odb_path = solve_job(model_name, job_name)
            probes[probe_name] = extract_linear(odb_path, probe)
            del mdb.models[model_name]
            print("Completed {} ({} elements)".format(job_name, element_count))
        results.append(
            {
                "case_id": case["case_id"],
                "stiffness": stiffness_from_probes(probes, 0.01),
                "probes": probes,
            }
        )
    return results


def run_nonlinear_case(manifest):
    case = manifest["nonlinear_case"]
    model_name = str("M_" + case["case_id"])
    model = mdb.Model(name=model_name)
    lx = manifest["domain"]["lx"]
    ly = manifest["domain"]["ly"]
    part = create_partitioned_part(model, "RVE", lx, ly, [])
    material = model.Material(name="SOLID")
    young = case["young_modulus"]
    poisson = case["poisson_ratio"]
    mu = young / (2.0 * (1.0 + poisson))
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    material.Hyperelastic(
        materialType=ISOTROPIC,
        testData=OFF,
        type=NEO_HOOKE,
        table=((0.5 * mu, 2.0 / bulk),),
    )
    model.HomogeneousSolidSection(name="SECTION", material="SOLID", thickness=1.0)
    part.Set(name="ALL", faces=part.faces[:])
    part.SectionAssignment(region=part.sets["ALL"], sectionName="SECTION")
    part.setMeshControls(regions=part.faces[:], elemShape=TRI, technique=FREE)
    part.setElementType(
        regions=(part.faces[:],),
        elemTypes=(mesh.ElemType(elemCode=CPE6H, elemLibrary=STANDARD),),
    )
    part.seedPart(
        size=manifest["mesh"]["abaqus_size"], deviationFactor=0.1, minSizeFactor=0.1
    )
    part.generateMesh()
    element_count = len(part.elements)
    gamma = case["gamma12"]
    assembly, jump_x, jump_y = add_periodic_constraints(
        model, part, lx, ly, 0.0, 0.0, gamma, 0.0
    )
    model.StaticStep(name="LOAD", previous="Initial", nlgeom=ON)
    model.DisplacementBC(
        name="MACRO_X", createStepName="LOAD", region=assembly.sets["RP_X"],
        u1=jump_x[0], u2=jump_x[1]
    )
    model.DisplacementBC(
        name="MACRO_Y", createStepName="LOAD", region=assembly.sets["RP_Y"],
        u1=jump_y[0], u2=jump_y[1]
    )
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "SENER", "IVOL", "EVOL")
    )
    job_name = str("canonical_" + case["case_id"])
    odb_path = solve_job(model_name, job_name)
    odb = openOdb(odb_path, readOnly=True)
    frame = odb.steps["LOAD"].frames[-1]
    stress, initial_volume = volume_average(frame, "S", (3,))
    energy, unused_volume = volume_average(frame, "SENER", (None,))
    current_volume = integrated_volume(frame, "EVOL")
    odb.close()
    del mdb.models[model_name]
    print("Completed {} ({} elements)".format(job_name, element_count))
    return {
        "case_id": case["case_id"],
        "deformation_gradient": [[1.0, gamma], [0.0, 1.0]],
        "macro_energy": energy[0],
        "macro_p12": stress[0],
        "mean_j": current_volume / initial_volume,
        "abaqus_quantities": {"macro_p12": "volume-averaged Cauchy S12; equal to P12 for this state"},
    }


def main():
    manifest = read_manifest()
    if not os.path.isdir(WORK):
        os.makedirs(WORK)
    os.chdir(WORK)
    output = {
        "schema_version": manifest["schema_version"],
        "abaqus_release": "2022",
        "linear_cases": run_linear_cases(manifest),
        "nonlinear_case": run_nonlinear_case(manifest),
    }
    output_path = os.path.join(HERE, "canonical_abaqus_results.json")
    with open(output_path, "w") as stream:
        # Explicit separators avoid trailing spaces emitted by Abaqus 2022's
        # Python 2 JSON encoder when pretty-printing lists and dictionaries.
        json.dump(output, stream, indent=2, separators=(",", ": "))
        stream.write("\n")
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
