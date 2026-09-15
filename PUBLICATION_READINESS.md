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
- Baseline result on 2026-09-15: 48 tests passed under Python 3.10 and DOLFINx 0.9.0, with 40% total statement coverage.
- The associated research project contains geometry-matched HomiCSx and ABAQUS calculations, scripts, meshes, solver inputs, ODB files, extracted element data, and summary outputs.

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

### Candidate experimental features

These features require additional evidence before they are included in the publication-supported core:

- Finite-strain viscoelasticity and state evolution.
- Quad and hex meshing across the advertised geometry range.
- Overlapping-void and open-cell-foam workflows.
- Stochastic sweep convenience functions.
- Ensemble summary and plotting utilities.
- Imported external-mesh workflows.
- Advanced post-processing beyond essential result extraction.

Experimental classification is not a judgment that a feature is defective. It prevents the publication release from making stronger support claims than its tests and validation justify.

## Findings by priority

### P0 Submission blockers

1. **The homogenization solvers lack adequate regression and verification coverage.** The current suite emphasizes geometry and Gmsh construction. The new end-to-end linear smoke test checks finite output and approximate stiffness symmetry, but not yet an analytical homogenized result.
2. **No continuous-integration workflow is present.** Reviewers cannot see automated evidence that supported behavior remains intact.
3. **There is no compact, repository-owned verification and validation suite.** The research project contains strong ABAQUS evidence, but it is paper-specific, path-dependent, large, and not organized as reusable HomiCSx validation.
4. **Packaging metadata is incomplete.** `pyproject.toml` lacks authors, license metadata, readme, classifiers, URLs, dependencies or dependency strategy, optional development dependencies, and other release metadata expected of a reusable Python package.
5. **The supported public scope is not defined.** The README presents a broad feature list without differentiating stable, experimental, and known-unsupported behavior.
6. **The project lacks contribution, support, issue-reporting, and maintenance guidance.** This is especially important for an archival maintenance posture.
7. **JOSS public-history timing is not yet favorable.** The first public commit is dated 2026-05-05. A submission should not be attempted before at least six months of genuine public history and a fresh venue check.

### P1 High-priority quality risks

1. **Documentation and solver environments diverge.** Read the Docs targets Python 3.13 while the solver environment targets Python 3.10.
2. **Documentation imports are heavily mocked.** A successful documentation build does not establish that documented public imports work against the real scientific dependencies.
3. **Dependency specifications are split and inconsistent.** `environment.yml`, `docs/requirements.txt`, and `pyproject.toml` do not express one coherent support policy.
4. **The documentation dependency file appears generated and over-pinned.** It includes many unrelated or transitive packages and duplicates incompatible Sphinx and MyST constraints.
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
2. Add analytical homogeneous-material verification for linear and finite-strain solvers.
3. Inspect the nonlinear driver, hook lifecycle, constitutive implementation, and averaging definitions against the research workflow.
4. Extract one minimal ABAQUS benchmark and document exact cross-solver conventions.
5. Draft the supported-versus-experimental feature matrix for author approval.
