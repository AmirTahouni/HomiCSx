# HomiCSx--Abaqus finite-strain comparison

This compact dataset records nine outcome-blind, geometry-matched comparisons at macroscopic deformation gradient `diag(1.5, 1.0, 1.0)`. It covers three particle counts and three clearance-to-radius levels. Abaqus is an independent comparison implementation, not ground truth.

The retained verification quantities are macroscopic energy, the normal components `P11`, `P22`, and `P33` of macroscopic first Piola--Kirchhoff stress, and mean deformation Jacobian `J`. The acceptance limit is 1% for macroscopic energy and stresses. The mean-J comparison uses a tighter 0.000001% gate because both solvers impose the same macroscopic deformation with determinant 1.5.

## Canonical, reproducible suite

The package-specific suite in this directory supersedes the research-derived
table above as the primary reproducible Abaqus verification. It deliberately
uses no localization or percentile (`Q`) quantities. Its conventional outputs
are the homogenized plane-strain stiffness, macro stress, macro strain, macro
energy, and (for finite strain) mean deformation Jacobian.

The deterministic cases are defined in `canonical_cases.json`:

1. `homogeneous_nonunit`: a 2.0 by 0.75 homogeneous patch represented with two
   identically assigned phases. This detects domain-length and periodic-jump
   errors.
2. `centered_circle_vf20`: a centered circular inclusion at exactly 20% area
   fraction and a 10:1 Young's-modulus contrast.
3. `periodic_split_circle`: one physical circle split across the left and right
   boundaries. This directly exercises matching periodic boundary topology.
4. `homogeneous_simple_shear`: a finite-strain homogeneous simple shear with
   `gamma12=0.2`.

The linear Abaqus models use `CPE6` elements, small-strain isotropic elasticity,
and equation-based periodic boundary conditions. Three independent macro-strain
probes reconstruct the complete 2D plane-strain stiffness. The nonlinear model
uses `CPE6H`, `NLGEOM=ON`, and Abaqus's compressible Neo-Hookean model. The
nonlinear comparison is intentionally restricted to macro energy, shear stress,
and `J` for homogeneous isochoric simple shear: those quantities coincide
exactly with HomiCSx's `NeoHookeanIsotropic` definition even though the two
programs use different volumetric/deviatoric splits for general deformations.

Run the two solvers and recompute the gates from the repository root:

```bash
python -m validation.abaqus.run_homicsx_canonical
cd validation/abaqus
abaqus cae noGUI=run_abaqus_canonical.py
cd ../..
python -m validation.abaqus.compare_canonical
```

Abaqus must be launched from `validation/abaqus` because Abaqus 2022 does not
define `__file__` for a `noGUI` script. Solver databases and logs are written to
the ignored `canonical_work/` directory. The two compact JSON result files are
retained for review and for comparison tests. The Abaqus script uses only the
bundled CAE/Standard Python environment and requires no user subroutine compiler.

The committed reference run used Abaqus 2022. All canonical gates are 1% for
linear stiffness, stress, and energy, 1% for nonlinear energy and shear stress,
and 0.1% for mean `J`.

Reference-run results:

| Case | Stiffness error | Maximum probe-stress error | Maximum probe-energy error |
|---|---:|---:|---:|
| Homogeneous non-unit patch | 0.224% | 0.273% | 0.284% |
| Centered circle, 20% area fraction | 0.396% | 0.542% | 0.539% |
| Periodic boundary-split circle | 0.115% | 0.282% | 0.137% |

For homogeneous finite simple shear, the macro-energy error is 0.000159%, the
macro-shear-stress error is 0.000157%, and the mean-`J` error is below
`1e-8`%. Abaqus is treated as an independent comparison implementation, not as
ground truth.

Run the comparison without an Abaqus installation:

```console
python -m validation.abaqus.compare_reference
```

The command recomputes every relative difference from `reference_results.csv`; stored pass flags are deliberately not used. Expected maximum absolute differences are 0.106% for macroscopic energy, 0.137% for `P11`, 0.0051% for `P22`, 0.111% for `P33`, and approximately 1.42e-8% for mean `J`.

Raw Abaqus ODB files, dense field exports, and research-specific localization statistics are intentionally excluded. `metadata.json` records the material, loading, mesh, selection, and source provenance needed to interpret this extract. This suite demonstrates macroscopic agreement for the selected cases; it does not establish pointwise field identity, general mesh independence, or validity for every HomiCSx feature.
