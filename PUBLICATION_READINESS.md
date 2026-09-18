# HomiCSx Publication Readiness Audit

## Purpose

This document records the evidence-based audit for preparing HomiCSx as a stable, citable research-software release. The selected venue is SoftwareX, using its Original Software Publication route.

The publication release should present HomiCSx as an extensible, end-to-end framework for linear and nonlinear finite-element computational homogenization. The supported workflow spans periodic microstructure generation, meshing, finite-element formulation, homogenization, hook-based customization, and essential post-processing.

## Current baseline

- Published and archived version: 1.0.0; corrective release candidate: 1.0.1.
- License: MIT.
- Public repository history begins on 2026-05-05.
- Version 1.0.0 is tagged and published on GitHub and archived by Zenodo with
  version DOI `10.5281/zenodo.22811694` and concept DOI
  `10.5281/zenodo.22811693`.
- Documentation is published through Read the Docs.
- Supported solver environment is described by `environment.yml` and targets Python 3.10, DOLFINx 0.9.0, and dolfinx_mpc 0.9.0.
- ABAQUS 2022 is installed locally on Windows.
- WSL2 with Ubuntu and Conda is available locally.
- A clean `homicsx_dev` Conda environment can be created from `environment-dev.yml` under WSL2.
- The v1.0.0 release gate passed 114 tests under Python 3.10 and DOLFINx
  0.9.0, a strict Sphinx build, package builds, and Abaqus-reference checks.
  Post-release review identified inconsistent distributed behavior; the
  corrective release therefore defines all homogenization drivers as
  single-rank and rejects larger communicators. The revised suite passes 120
  tests with 62.49% statement coverage against a 60% CI floor; the strict
  Sphinx build and 12-page manuscript compilation also pass.
- Homogeneous plane-strain verification now includes a deterministic two-level convergence gate on one fixed seeded geometry. Relative stiffness error falls from 2.543% at minimum/maximum mesh sizes 0.08/0.16 to 0.211% at 0.025/0.05, a 12.0-fold reduction. CI requires coarse error below 5%, fine error below 0.5%, and at least a factor-two reduction; no claim of monotonic intermediate convergence is made for independently regenerated unstructured meshes.
- The repository contains a focused HomiCSx--Abaqus suite using macroscopic
  energy, stress, strain, stiffness, and relaxation histories. The canonical
  linear and homogeneous nonlinear acceptance gate regenerates HomiCSx output
  from the current checkout before comparing it with archived Abaqus
  references.

## Publication scope

### Proposed supported core

- Periodic particulate geometry generation in two and three dimensions for documented shapes.
- Gmsh geometry construction, meshing, physical tagging, and DOLFINx conversion.
- Multiphase linear-elastic material assignment.
- Linear periodic homogenization.
- Finite-strain hyperelastic periodic homogenization.
- Custom nonlinear material implementations through the documented material interfaces.
- Nonlinear workflow customization through hooks.
- Essential history, field, and XDMF outputs required to reproduce documented examples.

### Features excluded from the publication-supported scope

These features require additional evidence before they are included in the publication-supported core:

- General 3D hexahedral meshing. Tested 2D all-quadrilateral meshing is in the
  supported core.
- Overlapping-void and open-cell-foam workflows.
- Stochastic sweep and ensemble convenience functions.
- Visualization helpers.
- Imported external-mesh workflows.
- Advanced post-processing beyond essential result extraction.

The stochastic and visualization modules are explicitly experimental and
excluded from publication claims and the 1.x compatibility guarantee. They are
retained for evaluation and may change or be removed in a future release.
Experimental classification is not a judgment that every feature is defective;
it prevents stronger support claims than the evidence justifies.

## Findings by priority

### P0 Submission blockers

1. **Core analytical verification implemented.** End-to-end linear tests check finite output, stiffness symmetry, analytical homogeneous stiffness recovery in 2D plane strain and full six-load-case 3D, and deterministic coarse-to-fine error reduction. The nonlinear periodic solver recovers analytical homogeneous Neo-Hookean macroscopic energy, PK1 stress, and mean J; the constitutive PK1 is independently checked against the numerical energy gradient in 2D and 3D. Hook ordering, shared state, scope, and failure isolation are tested. Further convergence studies can broaden evidence but are no longer a submission blocker for the documented core.
2. **Packaging metadata completed for the source-release workflow.** `pyproject.toml` now records authorship, license, readme, classifiers, URLs, Python support, and optional test/docs dependencies. The Conda environment is explicitly authoritative for the compiled FEniCSx/PETSc/MPI runtime stack rather than making an unreliable PyPI dependency claim.
3. **Archival maintenance guidance completed.** `CONTRIBUTING.md`, `SUPPORT.md`, and `SECURITY.md` now document issue reporting, contribution expectations, availability-dependent maintenance, and license-enabled continuity through community forks.
4. **Archival release metadata completed.** Version 1.0.0, author affiliation,
   no-ORCID submission, no-funding declaration, no-competing-interest
   declaration, Git tag, GitHub release, and Zenodo identifiers are resolved.
   Reconfirm the live SoftwareX instructions immediately before submission.

The publication-preparation work is merged into `main`. GitHub Actions covers
pull requests, pushes to `main`, and manual runs; local release gates provide
the full solver, documentation, validation, and package checks.

### P1 High-priority quality risks

1. **Documentation and solver environments aligned.** Read the Docs and the solver environment now target Python 3.10.
2. **Documentation imports are heavily mocked.** A successful documentation build does not establish that documented public imports work against the real scientific dependencies.
3. **Dependency roles are now explicit.** Conda owns the compiled runtime stack; `pyproject.toml` describes the package and optional pure-Python tooling; `docs/requirements.txt` contains only direct documentation dependencies.
4. **The documentation dependency file has been reduced to direct, bounded requirements.** Malformed API docstrings, duplicate indexing, the missing static path, and orphan demo pages have been corrected. The Sphinx build now passes with warnings treated as errors and is enforced in CI.
5. **Public API and compatibility policy documented.** `PUBLIC_API.md` defines top-level supported imports, explicitly non-guaranteed experimental exports, and deprecation practice. Nonlinear settings, results, simulation state, and typed hook data are available from the top-level package and covered by regression tests.
6. **Unsupported branches classified and documented.** Plane stress now raises a tested, actionable `NotImplementedError`; other unsupported workflows are listed on the limitations page. Remaining `NotImplementedError` paths are private defensive shape branches or belong to explicitly experimental stochastic helpers.
7. **Runtime output relies heavily on direct `print` calls.** Library-level diagnostics should be reviewed and generally routed through logging or explicit result objects.
8. **Runnable examples consolidated.** The root-level notebook collection was
   removed; five deterministic serial workflows now form the single maintained
   `examples/` surface.
9. **Notebook repository noise removed.** Stale, experimental, and
   research-specific notebooks remain recoverable in Git history but are not
   distributed as supported release examples.
10. **Citation metadata is synchronized.** The author has elected not to add an
    ORCID. The concept DOI is recorded now; the version-specific v1.0.1 DOI
    must be synchronized after Zenodo creates the corrective-release record.

### P2 Documentation and presentation improvements

1. **README completed.** It now states the need, supported scope, minimal
   example, installation path, validation status, citation, and support limits.
2. Add conceptual documentation for conventions, periodicity, stress and strain measures, averaging, load cases, hooks, and result interpretation.
3. Maintain the limitations page so unsupported behavior remains distinct from planned or experimental behavior.
4. Add reproducible tutorials for linear homogenization, nonlinear homogenization, custom materials, and hooks.
5. Add a verification and validation section containing benchmark definitions, software versions, tolerances, reference values, and regeneration instructions.
6. Correct spelling, terminology, and inconsistent capitalization throughout the README and documentation.
7. **Publication figures completed.** Reproducible figures communicate the
   architecture, periodic meshes, conventional validation histories, and
   ParaView fields.

## Test and verification programme

### Fast tests without full finite-element solves

- Dataclass and input validation across valid and invalid parameter combinations.
- Geometry volume or area calculations.
- Determinism for seeded generation.
- Periodic-image identity and minimum-image distance behavior.
- Clearance enforcement and infeasible-packing failures.
- Material-parameter transformations and invalid constitutive parameters.
- Voigt conversion ordering and tensor symmetry.
- Load-case construction and hook ordering.
- Result-container behavior and serialization-friendly summaries.

### Solver integration tests

- Homogeneous 2D plane-strain linear elasticity recovers the analytical stiffness.
- Homogeneous 3D linear elasticity recovers the analytical stiffness.
- Periodic heterogeneous linear cases satisfy symmetry and energy consistency.
- Finite-strain homogeneous Neo-Hookean cases recover analytical energy and stress.
- Hook callbacks run in documented order and can collect fields without changing the default solution.
- A hook that requests adaptive reduction produces the documented state transition.
- Public drivers reject communicators larger than one rank with an actionable
  error; distributed execution is not advertised or supported.

### Numerical verification checks

- Hill-Mandel energy consistency.
- Volume and phase-volume consistency.
- Stiffness symmetry within a defined tolerance.
- Positive-definiteness checks where theoretically applicable.
- Mesh-refinement behavior.
- Voigt, Reuss, and Hashin-Shtrikman bounds where applicable.
- Deterministic regression values for small canonical problems, with tolerances justified by discretization and solver variation.

## ABAQUS validation extraction

The reusable validation suite will be derived from the research workflow but will not duplicate the scientific localization study.

### Repository contents

- Small geometry definitions or deterministic geometry generators.
- ABAQUS input-generation and result-extraction scripts that can be redistributed.
- HomiCSx scripts using the same geometry, material model, loading path, and output definitions.
- Compact machine-readable reference results.
- Version and provenance metadata.
- Comparison code producing explicit pass or fail metrics.
- Documentation for running with and without an ABAQUS license.

### Excluded by default

- ODB, PRT, COM, and other large or transient ABAQUS files.
- Full parameter sweeps from the research paper.
- All eleven count-clearance settings and all research realizations.
- Manuscript-specific statistical inference and publication figures.
- Absolute machine paths and local environment assumptions.

### Initial benchmark candidates

1. Homogeneous 2D finite-strain plane-strain response.
2. One periodic circular-inclusion geometry at moderate resolution.
3. One geometry exercising the hook-based matrix-field extraction used by the research study.
4. A short loading-path comparison through stretch 1.50.
5. Mesh sensitivity sufficient to distinguish implementation disagreement from discretization disagreement.

The validation report must define the constitutive parameter conversion, stress measure, energy density, reference-area weighting, element or quadrature sampling, percentile method, periodic equations, and numerical tolerances identically on both sides.

## Sustainability and maintenance posture

HomiCSx should be prepared as a stable archival research release rather than marketed as an actively developed service. The eventual maintenance statement should communicate that:

- no regular feature schedule or guaranteed support response is promised;
- reproducibility, documentation, testing, versioned releases, and archival preservation are priorities;
- major correctness defects may be considered as resources permit;
- external contributions may be reviewed without a guaranteed response time; and
- the permissive license and documented architecture allow continued community use and forking.

This posture must be checked against the target journal immediately before submission. It must not imply commitments the author does not intend to make.

## Release plan

### Milestone 0.2.0

- Establish the supported and experimental scope.
- Repair packaging and dependency metadata.
- Create the development and test environment.
- Add CI and test markers.
- Add community and maintenance documentation.
- Obtain a clean baseline and address critical correctness failures.

### Milestone 0.3.0

- Complete core solver verification.
- Add the compact ABAQUS comparison suite.
- Add validation documentation and machine-readable reference results.
- Consolidate essential workflows into deterministic scripts and tested
  documentation examples. **Completed.**

### Milestone 1.0.0

- Freeze the publication API and compatibility statement.
- Complete documentation and release notes.
- Run the full verification and validation matrix.
- Build and inspect documentation from a clean environment.
- Prepare the software paper and AI-use disclosure.
- Create the tagged release and archival deposit approved for submission.

### Milestone 1.0.1

- Align the distributed-execution claim with the serial implementation.
- Restrict history-dependent viscoelastic workflows to the verified simplex
  path, provide state-preserving step-consistent tangents, and explicitly
  reject unsupported XDMF stress/energy reconstruction.
- Replace the nonlinear canonical self-check with a current-checkout solver
  execution and harden finite-difference tangent handling.
- Add focused coverage gates, linting, package validation, and installed-wheel
  smoke testing to CI.
- Rebuild the manuscript and create a corrective GitHub/Zenodo release.

## Decision gates

### Gate A Supported scope

Every feature advertised as supported must have documentation, an executable example, and an objective verification path. Otherwise it is experimental, deferred, or removed from the release claims.

### Gate B SoftwareX compliance

Use the current Original Software Publication template, keep the main text
within the journal's word and figure limits, complete its software metadata
table, and audit the package and manuscript against the public reviewer form.

### Gate C Research-paper sequencing

Before submission of the scientific paper, produce at minimum an immutable HomiCSx release with an archive DOI. Prefer peer-reviewed software-paper acceptance first, but do not make the research submission indefinitely dependent on journal review latency.

### Gate D Publication release

Version 1.0.0 was tagged after the clean environment, core test suite,
validation suite, documentation build, citation metadata, release notes, and
archival contents were reviewed. Version 1.0.1 repeats those gates and corrects
the limitations and release-engineering issues found during post-release
review.

## Immediate next actions

1. Obtain the author's approval of the exact submission files.
2. Verify every bibliographic reference against its publisher record.
3. Complete the live submission-system identity, residence, sanctions,
   payment, and upload checks.
