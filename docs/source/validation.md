# Verification and validation

HomiCSx uses complementary verification and validation checks:

- analytical homogeneous-material tests exercise the finite-element and homogenization pipeline;
- internal consistency checks assess stiffness symmetry and finite outputs; and
- geometry-matched Abaqus comparisons assess agreement with an independent implementation.

## Analytical verification

The automated test suite assigns identical isotropic linear-elastic properties
to the matrix and inclusion phases of geometrically heterogeneous meshes. In
2D plane strain, the homogenized stiffness must agree with the analytical
isotropic tensor within 0.5% in relative Frobenius norm. A full six-load-case
3D patch test uses a 1.5% limit to accommodate the deterministic unstructured
test mesh. These tests exercise geometry generation, meshing, material
assignment, periodic constraints, solution, and homogenized result extraction.

The 2D check also contains a deterministic coarse-to-fine convergence gate on
one fixed seeded geometry. With Gmsh minimum/maximum sizes of 0.08/0.16, the
relative stiffness error is 2.543%; at 0.025/0.05 it is 0.211%. Refinement
therefore reduces the error by a factor of 12.0. CI requires the coarse error
to remain below 5%, the fine error below 0.5%, and the fine error to be less
than half the coarse error. This two-level gate demonstrates controlled error
reduction without claiming monotonicity across independently generated
intermediate unstructured meshes.

For finite strain, numerical central differences of the compressible
Neo-Hookean strain energy are compared component-by-component with the
implemented first Piola--Kirchhoff stress in both 2D and 3D. The undeformed
configuration is also required to have zero energy and zero stress.

An end-to-end homogeneous finite-strain patch test applies the explicit
macroscopic deformation gradient $F=\mathrm{diag}(1.1,1)$ to a two-phase mesh
whose phases share identical Neo-Hookean properties. The nonlinear periodic
solver's macroscopic energy, complete first Piola--Kirchhoff stress tensor, and
mean $J$ must each agree with the analytical constitutive response within 0.5%.
The explicit deformation gradient avoids relying on load-case naming
conventions in the verification definition.

Hook tests verify registration order, shared-state propagation, load-case and
persistent state scopes, and the documented behavior when a hook raises an
exception.

## Abaqus finite-strain comparison

The repository contains a compact extract of nine outcome-blind, geometry-matched HomiCSx--Abaqus comparisons. The cases span particle counts 20, 50, and 110 and clearance-to-radius ratios 0.25, 0.50, and 0.75 at stretch 1.50.

The package-level suite uses conventional macroscopic quantities rather than the localization statistics studied in the associated research paper. Its acceptance limit is 1% for macroscopic energy and stress. Mean `J` uses a tighter tolerance because both solvers impose a deformation gradient with determinant 1.5. Maximum observed absolute differences are:

| Metric | Maximum difference | Limit |
|---|---:|---:|
| Macroscopic energy | 0.106% | 1% |
| Macroscopic `P11` | 0.137% | 1% |
| Macroscopic `P22` | 0.0051% | 1% |
| Macroscopic `P33` | 0.111% | 1% |
| Mean `J` | 1.42e-8% | 0.000001% |

Recompute the comparison from the stored reference values with:

```console
python -m validation.abaqus.compare_reference
```

The detailed scope, material and mesh settings, provenance, and limitations are stored under `validation/abaqus`. Abaqus is treated as an independent comparison implementation, not ground truth. Agreement on these selected aggregate metrics does not establish pointwise field identity, general mesh independence, or validation of features outside the documented scope.
