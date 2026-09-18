# HomiCSx--Abaqus validation

## Canonical, reproducible suite

The package-specific suite in this directory is the primary reproducible
Abaqus verification. Its conventional outputs
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
the ignored `canonical_work/` directory. The compact Abaqus JSON result and a
documented HomiCSx snapshot are retained for review. The automated gate
regenerates current HomiCSx linear results in a temporary directory before
comparison, so stale committed output cannot mask a solver regression. The Abaqus script uses only the
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

## Viscoelastic verification

`viscoelastic_case.json` defines an additional homogeneous, two-branch shear-
relaxation benchmark. A simple shear of `gamma12=0.01` is held for five time
units. The equilibrium Neo-Hookean branch has `E=10` and `nu=0.25`; Maxwell
branch shear moduli are 3 and 2, with relaxation times 0.2 and 1.0.

The validation compares the complete macro-`P12` history through three
independent paths:

1. the HomiCSx driver, including meshing, periodic constraints, material-state
   evolution, and macroscopic averaging;
2. direct evaluation of HomiCSx's exponential state recurrence; and
3. an Abaqus/Standard time-domain Prony-series model using `CPE6H` elements.

The Abaqus model uses long-term hyperelastic moduli, as required by the Abaqus
2022 CAE material convention when Prony viscoelasticity is attached. It applies
the shear in a negligible-duration preload step before the relaxation hold so
the first 0.05-time increment is not contaminated by a displacement ramp.

Run and compare it with:

```bash
python -m validation.abaqus.run_homicsx_viscoelastic
cd validation/abaqus
abaqus cae noGUI=run_abaqus_viscoelastic.py
cd ../..
python -m validation.abaqus.compare_viscoelastic
```

For the committed reference run, the HomiCSx end-to-end curve agrees with its
direct material recurrence to within `6e-12`% peak-normalized maximum error.
The Abaqus curve agrees with HomiCSx to within 0.000621% by the same measure;
the endpoint stress error is 0.0000028%, and maximum absolute mean-`J` error is
`1.12e-8`.

Two heterogeneous cases extend the same 100-increment relaxation protocol:

- a centered hyperelastic circle at 20% area fraction in the viscoelastic
  matrix; and
- a hyperelastic circle split across the periodic left/right boundary.

Run them with `run_homicsx_viscoelastic_heterogeneous.py` and
`run_abaqus_viscoelastic_heterogeneous.py`; `compare_viscoelastic.py` evaluates
the homogeneous, heterogeneous, and 3D suites together. On the refined independent
meshes, the centered case has 1.22% maximum peak-normalized curve error, 0.93%
RMS error, and 1.73% endpoint error. The periodic split case has 1.35%, 1.04%,
and 1.97%, respectively. Maximum mean-`J` disagreement is `3.08e-6`.

For generalized-Maxwell phases, HomiCSx now inserts the algorithmically updated
nonequilibrium branch stress directly into the weak equilibrium residual. The
Newton Jacobian is obtained by automatic differentiation while the previous
converged viscous metrics remain fixed; state is committed only after global
convergence. Thus the validation exercises time-dependent redistribution of
the heterogeneous fluctuation field, rather than adding viscous stress only
during post-processing.

A separate 3D homogeneous unit-cube patch holds the same simple shear for 50
increments. HomiCSx uses its periodic tetrahedral pipeline; Abaqus uses an
independently meshed `C3D10H` solid with affine exterior displacements. Run it
with `run_homicsx_viscoelastic_3d.py` and
`run_abaqus_viscoelastic_3d.py`. The peak-normalized maximum macro-`P12` curve
error is 0.000671%, the endpoint error is 0.0000028%, and maximum absolute
mean-`J` error is below `3e-10`. A heterogeneous 3D sphere-in-matrix fast/slow
rate regression additionally requires distinct converged fluctuation fields.
That regression exercises 3D heterogeneous redistribution, but is not an
external Abaqus comparison.

Raw Abaqus ODB files and dense field exports are intentionally excluded. The
JSON manifests record material, loading, mesh, and acceptance definitions.
These suites demonstrate agreement for the selected cases; they do not
establish pointwise field identity, general mesh independence, or validity for
every HomiCSx feature.
