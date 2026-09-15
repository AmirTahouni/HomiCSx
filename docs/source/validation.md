# Verification and validation

HomiCSx uses complementary verification and validation checks:

- analytical homogeneous-material tests exercise the finite-element and homogenization pipeline;
- internal consistency checks assess stiffness symmetry and finite outputs; and
- geometry-matched Abaqus comparisons assess agreement with an independent implementation.

## Analytical linear verification

The automated test suite assigns identical isotropic linear-elastic properties to the matrix and inclusion phases of a geometrically heterogeneous plane-strain mesh. The computed homogenized stiffness must agree with the analytical isotropic stiffness within 0.5% in relative Frobenius norm. This checks the public workflow from geometry generation through meshing, material assignment, periodic constraints, solution, and homogenized result extraction.

## Abaqus finite-strain comparison

The repository contains a compact extract of nine outcome-blind, geometry-matched HomiCSx--Abaqus comparisons. The cases span particle counts 20, 50, and 110 and clearance-to-radius ratios 0.25, 0.50, and 0.75 at stretch 1.50.

The predeclared acceptance limits are 1% for macroscopic energy and `P11`, and 3% for matrix energy Q99 and maximum-principal-Biot-stress Q99. Maximum observed absolute differences are:

| Metric | Maximum difference | Limit |
|---|---:|---:|
| Macroscopic energy | 0.106% | 1% |
| Macroscopic `P11` | 0.137% | 1% |
| Matrix energy Q99 | 0.516% | 3% |
| Matrix principal-Biot-stress Q99 | 2.052% | 3% |

Recompute the comparison from the stored reference values with:

```console
python -m validation.abaqus.compare_reference
```

The detailed scope, material and mesh settings, provenance, and limitations are stored under `validation/abaqus`. Abaqus is treated as an independent comparison implementation, not ground truth. Agreement on these selected aggregate metrics does not establish pointwise field identity, general mesh independence, or validation of features outside the documented scope.
