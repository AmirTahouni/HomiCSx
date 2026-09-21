# Pre-submission reviewer audit

This audit applies the questions most likely to determine editorial screening
and technical review. It is evidence-based and should be repeated against the
exact archived release.

## Scope and significance

**Assessment: ready.** The manuscript identifies the contribution as an
end-to-end, extensible implementation rather than a new homogenization theory.
It states the supported dimensionality, discretizations, material models,
outputs, and extension mechanisms. The impact argument is credible for
customized composite-cell studies and does not claim an external user base.

## Correctness and validation

**Assessment: ready.** Analytical patch tests,
constitutive energy-gradient checks, mesh refinement, regression tests, and
separately constructed Abaqus cases cover the advertised linear,
hyperelastic, and viscoelastic core. The comparison reports conventional
macroscopic quantities, not the application paper's localization statistics.
History-dependent averages on supported first-order meshes use geometric cell
measures and normalized quadrature weights; a deliberately nonuniform-cell
regression verifies this path. Custom phase/tag conventions and sparse tags
are tested across multiple load cases. Step-consistent tangents are checked
across every 2D column, a 3D shear
column, nontrivial prior history, perturbation sizes, and a real-solver
one-sided fallback, while committed fields and branch states are preserved.
CSV/XDMF output, heterogeneous state layouts, invalid custom loads, explicit
tangent failures, and detached post-tangent hook data are also covered.
The manuscript explicitly excludes pointwise equivalence and heterogeneous 3D
viscoelastic Abaqus validation from what has been established.

## Reproducibility

**Assessment: ready.** The repository contains a
version-constrained Conda environment, deterministic examples, tests, figure generators,
validation metadata, and compact reference results. The final release must be
tested in a fresh environment and archived before submission. The v1.0.1
corrective release candidate passes 157 tests, 66.68% statement coverage, a
strict documentation build, distribution checks, and manuscript compilation;
the new archive DOI remains to be inserted after Zenodo deposits the release.

## Usability and documentation

**Assessment: ready.** Installation, loading conventions, theory, examples,
validation, limitations, support expectations, and generated API documentation
are present. Experimental stochastic and visualization conveniences are
clearly separated from the publication-supported core.

## Sustainability

**Assessment: candid and acceptable for archival research software.** The MIT
license, public repository, versioned releases, tests, documentation, and
Zenodo archive support reuse without promising an indefinite feature
schedule. The paper accurately states that fixes and support are best-effort.

## Claims and presentation

**Assessment: ready after final author review.** Quantitative claims are tied
to named validation cases and bounded metrics. Four figures have distinct
roles: architecture, periodic meshing, illustrative field output, and external
validation. The manuscript avoids unsupported adoption, performance, and
generality claims.

## Remaining blockers

1. Archive v1.0.1, insert its version DOI, and obtain author approval of the exact final manuscript.
2. Complete the live submission-system identity, residence, sanctions, and
   payment declarations accurately.
