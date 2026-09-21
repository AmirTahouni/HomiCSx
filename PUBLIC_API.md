# Public API and compatibility policy

HomiCSx 1.0 defines a publication-supported research-software API. The supported public API is the
set of names exported from the top-level `homicsx` package, except for the
explicitly experimental stochastic names listed below. Public submodule paths
may be used for discovery, but compatibility guarantees apply to top-level
imports rather than private names beginning with an underscore.

The publication-supported top-level API covers geometry inputs and results,
mesh settings and generation, linear elasticity, the hyperelastic material
interface and built-in Neo-Hookean implementation, generalized-Maxwell
materials, material assignment, problem settings, linear and nonlinear
homogenization drivers and results, adaptive settings, simulation state, and
typed hook data objects.

For nonlinear results, `Ceff` is the complete row-major derivative
`d vec(P) / d vec(F)`, not a Voigt matrix. `tangent_status` records whether a
tangent was computed, skipped, not scheduled, or failed. Requested failures
raise by default; callers may explicitly select the record-and-continue mode.
Post-convergence, post-stress, and post-tangent hooks receive detached field
and constitutive-state snapshots. CSV output uses a wide macroscopic-history
table and a long material-state table that supports phase-specific variables.

HomiCSx 1.x supports solver execution on one MPI rank. MPI remains a runtime
dependency of DOLFINx and PETSc, but distributed homogenization is not part of
the supported API; the public drivers reject communicators with more than one
rank.

The hyperelastic extension interface accepts a user-defined UFL strain-energy
density through `psi_form`; numerical energy and first-Piola stress methods are
also required for macroscopic post-processing. HomiCSx validates its built-in
Neo-Hookean implementation, not every user-supplied constitutive law.

The following top-level compatibility exports remain experimental:

- `EnsembleStatSummary` and `EnsembleStudyResult`;
- `perform_ensemble_study`;
- `sweep_volume_fraction_linear`; and
- `sweep_stiffness_contrast_linear`.

The `homicsx.stochastic` and `homicsx.visualization` modules carry no 1.x
compatibility guarantee and may change or be removed in a future release.
Their presence does not place them in the publication-supported API.

For the supported API, bug fixes and compatible additions may appear in patch
releases. A breaking change should be documented in the changelog and, when
practical, preceded by a deprecation warning for at least one minor release.
Urgent correctness or security fixes may require faster changes. Because the
project follows an availability-dependent maintenance model, this policy does
not promise a release schedule or indefinite compatibility.
