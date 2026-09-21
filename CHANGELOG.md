# Changelog

All notable changes to HomiCSx are documented here. The project follows
semantic versioning for its publication-supported public API.

## 1.0.1 - 2026-09-21

### Changed

- Corrected the nonlinear result documentation, rejected non-finite
  macroscopic outputs, and guarded solver-owned macroscopic deformation data
  exposed to observational hooks.
- Restricted the documented solver contract to one MPI rank and made the
  homogenization drivers reject distributed communicators explicitly.
- Changed the package maturity classifier from Production/Stable to Beta.
- Made the canonical Abaqus acceptance test regenerate HomiCSx results from
  the current checkout before comparison with the archived external reference.
- Expanded the viscoelastic formulation and related-software discussion in
  the manuscript and documentation.
- Restricted generalized-Maxwell state updates to first-order simplex meshes
  and rejected equilibrium-only XDMF stress/energy reconstruction.
- Repaired central finite-difference tangents and added step-consistent
  generalized-Maxwell tangents that replay every perturbation from the same
  previous converged state without contaminating the committed solution.
- Corrected history-dependent macroscopic stress, energy, Jacobian, and
  tangent averages to use geometric cell measures on supported first-order
  nonuniform meshes.
- Validated nonlinear run controls and made post-convergence and post-stress
  hooks observational through detached solution and material-state snapshots.
- Propagated custom matrix-phase and physical-tag conventions throughout the
  nonlinear residual, averaging, state-reset, tangent, and XDMF paths, with
  sparse `MeshTags` handled by entity ID rather than array position.
- Repaired macroscopic CSV output, introduced heterogeneous-safe long-format
  state CSV output, and applied `output_prefix` to XDMF filenames.
- Documented nonlinear tangents as the full row-major `d vec(P)/d vec(F)`,
  added explicit status and fail-fast/record policies, and made post-tangent
  hooks observational through detached snapshots.
- Added unconditional custom-load validation, packaging CI, focused solver
  coverage gates, broader Ruff checks, and commit-pinned workflow actions.

### Removed

- Removed the unfinished, unsupported J2-plasticity prototype and the
  distributed-execution example.
- Removed a redundant research-derived Abaqus dataset whose provenance and
  version metadata no longer matched the publication validation suite.

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
  viscoelastic example scripts.
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

Version DOI: https://doi.org/10.5281/zenodo.22811694
Concept DOI: https://doi.org/10.5281/zenodo.22811693
