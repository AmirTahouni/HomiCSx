# HomiCSx: An end-to-end FEniCSx framework for finite-element computational homogenization

**Article type:** Original Software Publication  
**Author:** Amir Reza Tahouni  
**Affiliation:** Independent researcher, Tehran, Iran
**Corresponding email:** tahouniamirreza@gmail.com

## Abstract

HomiCSx is an open-source Python framework for first-order finite-element
computational homogenization of heterogeneous materials. It treats geometry
generation, periodic-conforming meshing, material assignment, finite-element
formulation, linear or finite-strain homogenization, workflow customization,
and essential result extraction as one reproducible pipeline. The supported
core includes two- and three-dimensional particulate cells, triangular or
tetrahedral Gmsh meshes, multiphase linear elasticity, compressible
Neo-Hookean materials, and a finite-strain generalized-Maxwell model. Periodic
fluctuation constraints are imposed through DOLFINx multipoint constraints;
user-defined load cases, nonlinear materials, and hooks allow research-specific
extensions without rewriting the driver. Verification combines analytical
homogeneous patch tests, constitutive energy-gradient checks, mesh-refinement
gates, and deterministic regression tests. Independent Abaqus comparisons use
conventional macroscopic quantities and cover linear heterogeneous cells,
finite-strain hyperelastic shear, and homogeneous and heterogeneous
viscoelastic relaxation. Maximum discrepancies in the canonical linear suite
are below 0.55%; heterogeneous viscoelastic stress-history discrepancies are
below 1.36% by the peak-normalized maximum measure for the tested cells.
HomiCSx is intended to make customized computational-homogenization studies
easier to construct, inspect, reproduce, and extend while stating a deliberately
bounded support and maintenance scope.

**Keywords:** computational homogenization; finite element method; FEniCSx;
representative volume element; composite materials; viscoelasticity

## 1. Motivation and significance

Finite-element computational homogenization requires a chain of choices that
are often implemented in separate research scripts: constructing a
representative cell, preserving periodic geometry, producing compatible
opposite-boundary meshes, assigning phases, enforcing periodic constraints,
solving multiple macroscopic load cases, and reducing microscopic fields to
macroscopic quantities. This fragmentation makes it difficult to adapt a
workflow without also weakening its traceability.

HomiCSx provides these steps through a common set of Python data objects and
drivers built on FEniCSx/DOLFINx and Gmsh. Its main contribution is not a new
homogenization theory. It is an extensible end-to-end implementation in which
the geometry, discretization, constitutive response, macroscopic loading,
nonlinear solution, and result extraction remain explicit and replaceable.
This is useful for studies that require systematic microstructure generation
or research-specific operations during a nonlinear homogenization history.

The software supports linear and finite-strain problems in two and three
dimensions. Research-specific behavior can be introduced through custom
material implementations, callable load cases, and typed hooks invoked at
documented points in the nonlinear driver. The package therefore occupies a
middle ground between one-off finite-element scripts and closed workflows that
offer limited intervention in the solution sequence.

## 2. Software description

### 2.1. Architecture and workflow

The workflow has five cooperating layers. The geometry layer creates seeded,
periodic particulate cells or accepts explicitly constructed inclusions. The
mesh layer builds the corresponding OpenCASCADE model, assigns physical tags,
and generates triangular or tetrahedral meshes. Opposite boundaries are meshed
with translated Gmsh periodic constraints so their nodes match. The material
layer maps phase identifiers to constitutive objects. The finite-element layer
constructs linear or nonlinear periodic fluctuation problems and applies
multipoint constraints. Finally, homogenization drivers execute standard or
user-defined macroscopic load cases and return macroscopic histories and
optional fields.

The core geometry types include circular and elliptical inclusions in 2D and
spherical and spheroidal inclusions in 3D, with monodisperse or polydisperse
sampling and clearance control. Periodic images retain source identity so that
boundary-split particles are represented consistently. Geometry generation is
seedable for reproducibility. The publication-supported discretization path
uses triangles or tetrahedra; forced quad/hex workflows are outside the current
support boundary.

### 2.2. Homogenization formulation

For linear elasticity, the microscopic displacement is decomposed into a
macroscopic affine field and a periodic fluctuation. Elementary macroscopic
strain probes recover the effective stiffness from volume-averaged stress. In
finite strain, the deformation gradient is decomposed as

\[
\mathbf{F}(\mathbf{X})=\bar{\mathbf{F}}+\nabla\tilde{\mathbf{u}}(\mathbf{X}),
\]

where the fluctuation is periodic on opposite cell boundaries. The nonlinear
driver solves incremental equilibrium with adaptive stepping and records
volume-averaged first Piola--Kirchhoff stress, energy, deformation Jacobian,
and optional tangent information.

Built-in nonlinear behavior includes compressible Neo-Hookean elasticity and a
generalized-Maxwell solid. For Maxwell phases, the algorithmically updated
nonequilibrium branch stress participates directly in the weak residual and
its Newton Jacobian is obtained by automatic differentiation. Previous
converged viscous metrics remain fixed during a global solve and are committed
only after convergence, preserving step-retry semantics.

### 2.3. Customization and outputs

Users can define macroscopic deformation histories as Python callables and can
derive new nonlinear material classes through the public material interface.
Hooks provide controlled access before and after documented stages of the
homogenization loop. They have explicit ordering and state scopes and can be
used to collect fields, compute application-specific metrics, or implement
additional workflow logic. Core outputs include effective stiffness, macro
stress/strain/energy histories, Jacobian histories, and XDMF field exports for
external post-processing.

## 3. Illustrative examples

The repository includes executable demonstrations for geometry generation,
2D and 3D linear homogenization, nonlinear hyperelasticity, generalized-Maxwell
homogenization, mesh convergence, and hook-based field processing. A minimal
linear workflow creates a seeded geometry, generates a periodic-conforming
mesh, assigns phase materials, and passes these objects to the linear driver.
The same separation of concerns is retained for nonlinear analyses, allowing a
geometry or material implementation to be changed without replacing the full
pipeline.

[FIGURE 1: Component and data-flow diagram, generated from a reproducible
source rather than a screenshot.]

[FIGURE 2: Representative periodic 2D/3D cells and matching opposite-boundary
meshes.]

## 4. Verification and validation

The automated test suite covers deterministic geometry behavior, tagging,
periodic mesh construction, public API behavior, hook semantics, constitutive
parameter validation, and solver integration. Homogeneous 2D plane-strain and
six-load-case 3D linear problems are compared with analytical isotropic
stiffness. A deterministic 2D refinement gate reduces relative stiffness error
from 2.543% on the coarse mesh to 0.211% on the refined mesh. Neo-Hookean first
Piola stress is independently checked against numerical energy derivatives in
2D and 3D, and an end-to-end nonlinear patch recovers analytical macroscopic
energy, stress, and mean Jacobian.

The external validation suite uses independently generated Abaqus models and
only conventional macroscopic quantities. Three linear plane-strain cases—a
homogeneous non-unit cell, a centered 20% circular inclusion, and a periodic
boundary-split inclusion—have maximum stiffness, probe-stress, and probe-energy
differences below 0.55%. Homogeneous finite simple shear agrees to within
0.00016% for macro energy and shear stress.

Generalized-Maxwell validation compares complete macro-shear-stress relaxation
histories. The homogeneous 2D and 3D curves agree with Abaqus to within
0.00063% and 0.00068% peak-normalized maximum error, respectively. For a
viscoelastic matrix containing a hyperelastic inclusion, the centered and
periodic-split 2D cases give 1.22% and 1.35% maximum curve errors. Automated
fast/slow-rate regressions in 2D and 3D additionally require different
converged heterogeneous fluctuation fields. The external evidence does not
establish pointwise field identity, all loading paths, or external validation
of a heterogeneous 3D viscoelastic cell.

[FIGURE 3: Macro-response validation curves and acceptance limits.]

## 5. Impact and reuse

HomiCSx enables a researcher to move from a seeded particulate description to
homogenized linear, hyperelastic, or viscoelastic response in one inspectable
workflow. Its hook mechanism is particularly relevant when a study needs
quantities or interventions that are not general enough to belong in a fixed
post-processing interface. The software is already used by its author in a
study of matrix localization in random periodic composites; that application
uses custom material and field-processing logic while retaining the common
geometry, mesh, equilibrium, and homogenization pipeline.

Potential reuse includes parametric studies of particulate composites,
development and comparison of constitutive laws, generation of training data
for reduced-order or surrogate models, and teaching reproducible RVE analysis.
No claim of external adoption is made at this stage. The MIT license, public
interfaces, tests, and archived releases are intended to allow independent use
and continuation.

## 6. Limitations and maintenance

The tested environment is Linux, including Linux under WSL2, with Python 3.10,
DOLFINx 0.9.0, and dolfinx_mpc 0.9.0. Two-dimensional FEM uses plane strain;
plane stress is not implemented. The supported meshing path uses triangles and
tetrahedra. Imported meshes, overlapping-void and open-cell workflows,
advanced visualization, and stochastic convenience modules are outside the
publication-supported core. The latter two modules are explicitly experimental.

HomiCSx is maintained as archival research software. No feature schedule or
guaranteed support response is promised. Reproducibility, documentation,
correctness fixes as resources permit, versioned releases, and preservation
are prioritized. The permissive license allows community forks and continued
development if active maintenance changes.

## 7. Availability and reproducibility

Source code: https://github.com/AmirTahouni/HomiCSx  
Documentation: https://homicsx.readthedocs.io/en/latest/  
License: MIT  
Version described: [RELEASE REQUIRED]  
Archive DOI: [ZENODO DOI REQUIRED]

The repository contains pinned Conda environment specifications, installation
instructions, automated tests, benchmark manifests, independent Abaqus runner
scripts, compact reference results, and executable acceptance gates. Abaqus is
not required to recompute comparisons from the committed compact results.

## CRediT authorship contribution statement

**Amir Reza Tahouni:** Conceptualization, Methodology, Software, Validation,
Investigation, Data curation, Visualization, Writing—original draft,
Writing—review and editing.

## Funding

This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

## Declaration of competing interest

The author declares no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this
paper. [AUTHOR TO CONFIRM]

## Data availability

The software, validation manifests, scripts, and compact numerical results are
available in the public repository and will be preserved in the archived
release identified above. Large proprietary Abaqus working databases are not
required to evaluate the committed comparison results and are not distributed.

## Software metadata

| Field | Value |
|---|---|
| Current code version | [RELEASE REQUIRED] |
| Permanent link to code/repository | [ZENODO DOI REQUIRED] |
| Code repository | https://github.com/AmirTahouni/HomiCSx |
| Legal software license | MIT |
| Code versioning system | Git |
| Software code languages/tools | Python, UFL, Gmsh API |
| Compilation requirements | Conda environment; Python 3.10; DOLFINx 0.9.0; dolfinx_mpc 0.9.0 |
| Operating environment | Linux; WSL2 tested |
| Support email | tahouniamirreza@gmail.com |

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the author used an OpenAI coding assistant
to support code review, test and validation development, documentation editing,
and manuscript drafting. After using this service, the author reviewed and
edited the content as needed and takes full responsibility for the content of
the publication.

## References

1. M. W. Scroggs, J. S. Dokken, C. N. Richardson, G. N. Wells, Construction of
   arbitrary order finite element degree-of-freedom maps on polygonal and
   polyhedral cell meshes, ACM Transactions on Mathematical Software 48 (2)
   (2022). [VERIFY DOI AND WHETHER THIS IS THE BEST DOLFINx CITATION]
2. C. Geuzaine, J.-F. Remacle, Gmsh: A 3-D finite element mesh generator with
   built-in pre- and post-processing facilities, International Journal for
   Numerical Methods in Engineering 79 (11) (2009) 1309–1331.
   https://doi.org/10.1002/nme.2579.
3. P. Henyš, L. Čapek, J. Březina, Comparison of current methods for
   implementing periodic boundary conditions in multi-scale homogenisation,
   European Journal of Mechanics - A/Solids 78 (2019) 103825.
   https://doi.org/10.1016/j.euromechsol.2019.103825.
4. [ADD VERIFIED FIRST-ORDER/FINITE-STRAIN COMPUTATIONAL HOMOGENIZATION
   REFERENCES.]
5. A. R. Tahouni, HomiCSx, version [VERSION], Zenodo (2026).
   [ZENODO DOI REQUIRED].
