# HomiCSx Publication Readiness Audit

## Purpose

This document records the initial evidence-based audit for preparing HomiCSx as a stable, citable research-software release. The intended first-choice venue is the Journal of Open Source Software, with another archival software journal retained as a fallback if HomiCSx does not satisfy JOSS scope or sustainability expectations.

The publication release should present HomiCSx as an extensible, end-to-end framework for linear and nonlinear finite-element computational homogenization. The supported workflow spans periodic microstructure generation, meshing, finite-element formulation, homogenization, hook-based customization, and essential post-processing.

## Current baseline

- Package version: 0.1.0.
- License: MIT.
- Public repository history begins on 2026-05-05.
- Public history currently contains 53 commits and no tagged release visible in the local clone.
- Documentation is published through Read the Docs.
- Supported solver environment is described by `environment.yml` and targets Python 3.10, DOLFINx 0.9.0, and dolfinx_mpc 0.9.0.
- ABAQUS 2022 is installed locally on Windows.
- WSL2 with Ubuntu and Conda is available locally.
- A clean `homicsx_dev` Conda environment can be created from `environment-dev.yml` under WSL2.
- Current result on 2026-09-16: 62 tests pass under Python 3.10 and DOLFINx 0.9.0. The suite includes analytical 2D/3D linear patch tests, a solver-level homogeneous finite-strain patch test, nonlinear constitutive consistency, hook semantics, and the compact Abaqus macroscopic-reference checks.
- Homogeneous plane-strain verification recovers the analytical stiffness within a 0.5% relative-norm tolerance on the fine test mesh. An initial four-level diagnostic reduced the error from 2.54% on the coarsest mesh to 0.21% on the finest, although independently generated unstructured meshes did not produce monotonic intermediate errors. A controlled mesh and periodic-constraint convergence study remains necessary.
- The repository now contains a compact nine-case HomiCSx--Abaqus macroscopic energy, stress, and deformation reference suite with machine-readable provenance and executable acceptance checks. Research-specific localization statistics remain in the associated study, which also retains the full calculations, scripts, meshes, solver inputs, ODB files, and extracted element data.

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

- Finite-strain viscoelasticity and state evolution.
- Quad and hex meshing across the advertised geometry range.
- Overlapping-void and open-cell-foam workflows.
- Stochastic sweep and ensemble convenience functions.
- Visualization helpers.
- Imported external-mesh workflows.
- Advanced post-processing beyond essential result extraction.

The stochastic and visualization modules are explicitly experimental and excluded from publication claims. They are initially retained for compatibility. Before 1.0, they will either receive a focused rework with adequate tests or be deprecated and removed from the public release. Experimental classification is not a judgment that every feature is defective; it prevents stronger support claims than the evidence justifies.

## Findings by priority

### P0 Submission blockers

1. **Core analytical coverage substantially expanded.** End-to-end linear tests check finite output, stiffness symmetry, and analytical homogeneous stiffness recovery in 2D plane strain and full six-load-case 3D. The nonlinear periodic solver recovers analytical homogeneous Neo-Hookean macroscopic energy, PK1 stress, and mean J; the constitutive PK1 is independently checked against the numerical energy gradient in 2D and 3D. Hook ordering, shared state, scope, and failure isolation are tested. A controlled mesh/periodic-constraint convergence study remains open.
2. **Packaging metadata completed for the source-release workflow.** `pyproject.toml` now records authorship, license, readme, classifiers, URLs, Python support, and optional test/docs dependencies. The Conda environment is explicitly authoritative for the compiled FEniCSx/PETSc/MPI runtime stack rather than making an unreliable PyPI dependency claim.
3. **Archival maintenance guidance completed.** `CONTRIBUTING.md`, `SUPPORT.md`, and `SECURITY.md` now document issue reporting, contribution expectations, availability-dependent maintenance, and license-enabled continuity through community forks.
4. **JOSS public-history timing is not yet favorable.** The first public commit is dated 2026-05-05. A submission should not be attempted before at least six months of genuine public history and a fresh venue check.

The publication-preparation branch now defines the supported scope and provides a GitHub Actions workflow for pull requests, pushes to `main`, and manual runs. The first hosted run passed all 49 tests on 2026-09-15; these changes remain subject to review and merge.

### P1 High-priority quality risks

1. **Documentation and solver environments aligned.** Read the Docs and the solver environment now target Python 3.10.
2. **Documentation imports are heavily mocked.** A successful documentation build does not establish that documented public imports work against the real scientific dependencies.
3. **Dependency roles are now explicit.** Conda owns the compiled runtime stack; `pyproject.toml` describes the package and optional pure-Python tooling; `docs/requirements.txt` contains only direct documentation dependencies.
4. **The documentation dependency file has been reduced to direct, bounded requirements.** Malformed API docstrings, duplicate indexing, the missing static path, and orphan demo pages have been corrected. The Sphinx build now passes with warnings treated as errors and is enforced in CI.
5. **The public API and compatibility policy are not stated.** Top-level exports exist, but stability expectations and deprecation rules are absent.
6. **Several advertised branches explicitly raise `NotImplementedError`.** These need clear documentation, tests for the expected failure, implementation, or removal from release claims.
7. **Runtime output relies heavily on direct `print` calls.** Library-level diagnostics should be reviewed and generally routed through logging or explicit result objects.
8. **Examples are notebook-heavy.** Important publication examples need deterministic, scriptable counterparts that can run in automated checks.
9. **The repository contains two notebooks larger than 1 MB.** Notebook outputs and embedded data should be reviewed for reproducibility, noise, and repository size.
10. **Citation metadata lacks an ORCID and release/archive identifiers.** These should be added when available and synchronized with the publication release.

### P2 Documentation and presentation improvements

1. Rewrite the README around a precise statement of need, supported scope, minimal example, installation path, documentation, validation status, citation, and support expectations.
2. Add conceptual documentation for conventions, periodicity, stress and strain measures, averaging, load cases, hooks, and result interpretation.
3. Add a limitations page that distinguishes unsupported behavior from planned or experimental behavior.
4. Add reproducible tutorials for linear homogenization, nonlinear homogenization, custom materials, and hooks.
5. Add a verification and validation section containing benchmark definitions, software versions, tolerances, reference values, and regeneration instructions.
6. Correct spelling, terminology, and inconsistent capitalization throughout the README and documentation.
7. Replace screenshots or decorative result plots with figures that communicate architecture, conventions, or verified behavior.

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
- Mesh and result behavior remain valid under MPI execution where supported.

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
- Convert essential notebooks into deterministic scripts or tested documentation examples.

### Milestone 1.0.0

- Freeze the publication API and compatibility statement.
- Complete documentation and release notes.
- Run the full verification and validation matrix.
- Build and inspect documentation from a clean environment.
- Prepare the software paper and AI-use disclosure.
- Create the tagged release and archival deposit approved for submission.

## Decision gates

### Gate A Supported scope

Every feature advertised as supported must have documentation, an executable example, and an objective verification path. Otherwise it is experimental, deferred, or removed from the release claims.

### Gate B JOSS viability

Reassess after six months of public history, completed core tests, at least one versioned prerelease, and documented research use. If scope or sustainability remains doubtful, prepare a pre-submission inquiry or redirect to a suitable archival software journal.

### Gate C Research-paper sequencing

Before submission of the scientific paper, produce at minimum an immutable HomiCSx release with an archive DOI. Prefer peer-reviewed software-paper acceptance first, but do not make the research submission indefinitely dependent on journal review latency.

### Gate D Publication release

Do not tag 1.0.0 until the clean environment, core test suite, validation suite, documentation build, citation metadata, release notes, and archival contents have all been reviewed.

## Immediate next actions

1. Add continuous integration for the clean environment or a documented equivalent if hosted CI cannot support the solver stack reliably.
2. Extend the initial 2D linear analytical check with mesh convergence, 3D linear verification, and finite-strain analytical verification.
3. Inspect the nonlinear driver, hook lifecycle, constitutive implementation, and averaging definitions against the research workflow.
4. Extract one minimal ABAQUS benchmark and document exact cross-solver conventions.
5. Draft the supported-versus-experimental feature matrix for author approval.
