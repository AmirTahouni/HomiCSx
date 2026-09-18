# HomiCSx

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22811693.svg)](https://doi.org/10.5281/zenodo.22811693)

HomiCSx is an open-source Python framework for first-order finite-element
computational homogenization on FEniCSx. It connects particulate geometry,
periodic-conforming Gmsh meshing, material assignment, linear and finite-strain
periodic problems, homogenization, hooks, and essential output in one
reproducible workflow.

<p align="center">
  <img src="paper/figures/architecture_workflow.png" width="850" alt="HomiCSx architecture and workflow">
</p>

## Supported core

- Seeded periodic RSA generation for monodisperse or polydisperse circles,
  spheres, axis-aligned ellipses, and axis-aligned ellipsoids.
- Explicit construction of custom particulate cells, including rotated
  ellipses and ellipsoids and coated/interphase inclusions.
- Periodic-conforming Gmsh meshes with phase and boundary tags. Triangles,
  tetrahedra, and tested 2D all-quadrilateral meshes are supported; general 3D
  hexahedral generation is not.
- Multiphase small-strain linear elasticity and finite-strain homogenization.
- A built-in compressible isotropic Neo-Hookean model, plus user-defined
  hyperelastic materials through UFL strain-energy functions and matching
  numerical post-processing methods.
- A finite-strain generalized-Maxwell material with state management.
- Built-in and custom load histories, adaptive stepping, typed hooks, and
  macroscopic stress, energy, and Jacobian histories. Finite-difference
  tangent histories are available for history-independent materials.
- XDMF displacement and reconstructed stress/energy output for
  history-independent materials, suitable for ParaView.

<p align="center">
  <img src="paper/figures/periodic_meshes.png" width="850" alt="Periodic-conforming 2D and 3D meshes">
</p>

The higher-level `homicsx.stochastic` convenience functions and
`homicsx.visualization` helpers are experimental and excluded from the
publication-supported API.

## Geometry scope

The random generators use periodic random sequential adsorption and explicitly
store the periodic images needed for boundary-crossing particles. Random
ellipses and ellipsoids are currently axis-aligned because independent random
orientations require a validated orientation-aware collision and clearance
algorithm. For prescribed geometries, `Inclusion.orientation` supports a 2D
angle or three 3D XYZ rotation angles.

The generator reports nominal volume fraction from original particles. For
overlapping/open-cell constructions, the current correction is approximate;
that workflow is outside the publication-supported core.

## Execution model

HomiCSx 1.x supports solver execution on one MPI rank. DOLFINx and PETSc still
use MPI internally, but distributed HomiCSx homogenization is deliberately
rejected until cross-rank result consistency is established and continuously
tested. Run workflows without `mpiexec`.

## Installation

HomiCSx 1.0.1 is tested on Linux, including Linux under WSL2, with Python 3.10,
DOLFINx 0.9.0, and `dolfinx_mpc` 0.9.0. The versioned Conda environment is the
authoritative dependency specification. Linux is the verified platform;
native Windows and macOS are not currently in the test matrix.

```bash
git clone https://github.com/AmirTahouni/HomiCSx.git
cd HomiCSx
conda env create -f environment.yml
conda activate homicsx_env
python -m pip install --no-deps -e .
python -c "import homicsx; print(homicsx.__version__)"
```

Other dependency versions and native Windows/macOS installations may work but
are outside the current verification matrix.

## Tested examples

The maintained examples are exercised by the test suite. Start with the compact
2D linear workflow from a source checkout:

```bash
python examples/linear_periodic_2d.py
```

The [`examples/`](examples/) directory contains deterministic, non-interactive
geometry, 2D and 3D linear, hyperelastic, and viscoelastic
workflows. It is the single maintained entry point for runnable examples; the
documentation provides the corresponding narrative tutorials.

<p align="center">
  <img src="paper/figures/paraview_fields.png" width="850" alt="HomiCSx finite-element fields rendered in ParaView">
</p>

## Verification and documentation

The [verification documentation](docs/source/validation.md) summarizes
analytical tests, mesh-refinement gates, current-code regression tests, and independent
Abaqus comparisons based on conventional macroscopic stress, strain, energy,
Jacobian, and stiffness measures.

- [Documentation](https://homicsx.readthedocs.io/en/latest/)
- [Abaqus comparison suite](validation/abaqus/README.md)
- [Supported API](PUBLIC_API.md)
- [Support and maintenance policy](SUPPORT.md)
- [Contributing guide](CONTRIBUTING.md)
- [Citation metadata](CITATION.cff)
- [Archived v1.0.0 release](https://doi.org/10.5281/zenodo.22811694)

## Author and license

HomiCSx is developed by Amir Reza Tahouni
(tahouniamirreza@gmail.com) and distributed under the MIT License.
