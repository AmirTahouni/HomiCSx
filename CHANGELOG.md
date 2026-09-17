# Changelog

All notable changes to HomiCSx are documented here. The project follows
semantic versioning for its publication-supported public API.

## 1.0.0 - 2026-09-17

First archival research-software release prepared for the SoftwareX software
paper.

### Added

- Periodic particulate geometry generation in two and three dimensions.
- Periodic-conforming Gmsh meshes with phase and boundary tagging.
- Linear periodic homogenization for multiphase isotropic elasticity.
- Finite-strain homogenization with a built-in Neo-Hookean law, a public
  user-defined hyperelastic-energy interface, and generalized-Maxwell response.
- Custom material interfaces, load callables, and typed nonlinear hooks.
- Deterministic geometry, 2D/3D linear, hyperelastic, heterogeneous
  viscoelastic, and linear-MPI example scripts.
- Analytical, regression, mesh-refinement, and Abaqus comparison suites.
- Documentation for theory, loading conventions, validation, limitations,
  experimental modules, support, and the public API.

### Changed

- Periodic-conforming meshing is the default supported homogenization path.
- Viscoelastic branch stresses participate in nonlinear equilibrium and their
  internal metrics are committed only after converged increments.
- Package metadata, CI, environment specifications, and citation metadata are
  aligned for an archival release.
- Runnable examples, compact validation data, tests, and supporting project
  documents are included in the source distribution.

### Support boundary

- Plane strain is supported in 2D; plane stress is not implemented.
- Triangle, tetrahedron, and tested 2D all-quadrilateral meshes form the
  supported discretization path; general 3D hexahedral meshing is rejected.
- Stochastic convenience and visualization modules remain experimental.
- Maintenance is best-effort archival support without a guaranteed response
  time or feature schedule.

The archival identifiers are added to the repository after the tagged release
is deposited.
