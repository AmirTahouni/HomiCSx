# SoftwareX submission checklist

## Blocking author inputs

- [x] Confirm the final author spelling: **Amir Reza Tahouni**.
- [x] Provide the affiliation to print, or explicitly choose “Independent
  researcher, Tehran, Iran.”
- [x] Confirm the corresponding-author email: tahouniamirreza@gmail.com.
- [x] Provide an ORCID, or decide to submit without one: submit without one.
- [x] Confirm whether any funding supported HomiCSx development: no funding.
- [x] Decide whether to rely on the associated application paper as a preprint:
  no; keep the software paper independently supported unless this changes.

## Software release

- [x] Add deterministic, tested scripts for the essential linear and
  heterogeneous viscoelastic workflows.
- [x] Choose the publication version: `1.0.0`.
- [x] Finish the release notes and public API freeze.
- [x] Run the full clean-environment test, strict documentation, validation,
  package-build, and installed-wheel matrix for the 1.0.0 release candidate.
- [x] Create and push the signed or annotated Git tag.
- [x] Create the GitHub release.
- [x] Archive the release in Zenodo and obtain the version DOI and concept DOI.
- [x] Add DOI and release metadata to `CITATION.cff`, README, and manuscript.
- [x] Verify that the permanent repository/release link contains the README,
  open-source license, source code, environment specification, tests, and
  validation data.

## Manuscript and submission files

- [x] Compare against the official SoftwareX Original Software Publication
  template, Version 6 (March 2026), and retain its required section and metadata
  structure in the selected LaTeX source.
- [x] Select LaTeX and produce a compiling `elsarticle` manuscript draft.
- [x] Keep `manuscript.md` synchronized with the submission LaTeX source.
- [x] Keep the main text within the template's 4000-word limit. The current
  main body is approximately 1750 words; the abstract is separate.
- [x] Use no more than six figures. The current manuscript has four.
- [x] Complete the required C1--C8 software metadata table. C2 uses the
  mandatory GitHub repository; the Zenodo version DOI is used for the software
  citation.
- [x] Prepare a workflow/component figure from a reproducible source.
- [x] Prepare one reproducible ParaView field figure illustrating stress and
  strain-energy output without using localization/Q statistics.
- [x] Prepare a validation figure comparing the conventional macro-response
  histories; do not use research-paper localization/Q statistics.
- [x] Draft compliant highlights, an optional graphical abstract, and a
  SoftwareX-specific cover letter; reconfirm upload requirements in the live
  submission system.
- [x] Verify every reference and DOI against its publisher or authoritative
  repository record; retain the audit in `reference_audit.md`.
- [x] Add the archived HomiCSx software citation as a software reference.
- [x] Add a focused comparison with MicroStructPy, DAMASK, and MOOSE.
- [x] State the generalized-Maxwell free energy, stress contribution, update,
  initialization, and Abaqus parameter mapping explicitly.
- [x] State the one-rank execution boundary consistently in code, docs, and
  manuscript.
- [x] Retain the generative-AI disclosure because AI assisted code review,
  validation preparation, documentation, and manuscript drafting.
- [ ] Complete the submission-system declarations and sanctions/payment checks
  using the author's own accurate identity and residence information.

## Final quality gate

- [x] Reviewer-form audit: scientific scope is explicit.
- [x] Reviewer-form audit: novelty and research impact are supported without
  claiming adoption that has not occurred.
- [x] Reviewer-form audit: architecture and experimental setting are clear.
- [x] Reviewer-form audit: empirical evidence supports every advertised core
  feature, or the limitation is stated.
- [x] Reviewer-form audit: release-dependent metadata tables are complete and
  consistent.
- [x] Render and visually inspect the final PDF.
- [x] Compile and visually inspect every page of the pre-release LaTeX PDF.
- [ ] Obtain author approval of the exact files before submission.
- [x] Confirm the declaration of no competing interests.
- [ ] Publish a corrective release containing the post-review changes and
  replace the v1.0.0 version DOI in submission files with its immutable DOI.
