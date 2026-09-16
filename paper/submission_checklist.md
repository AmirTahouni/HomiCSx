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
- [ ] Choose the publication version (proposed: `1.0.0`).
- [ ] Finish the release notes and public API freeze.
- [ ] Run the full clean-environment test and documentation matrix.
- [ ] Create and push the signed or annotated Git tag.
- [ ] Create the GitHub release.
- [ ] Archive the release in Zenodo and obtain the version DOI and concept DOI.
- [ ] Add DOI and release metadata to `CITATION.cff`, README, and manuscript.
- [ ] Verify that the permanent repository/release link contains the README,
  open-source license, source code, environment specification, tests, and
  validation data.

## Manuscript and submission files

- [ ] Download the current official SoftwareX Original Software Publication
  template and choose Word or LaTeX.
- [ ] Transfer `manuscript.md` without altering the template formatting.
- [ ] Keep main text, abstract, captions, and footnotes within 3000 words.
- [ ] Use no more than six figures.
- [ ] Complete the required software metadata table.
- [x] Prepare a workflow/component figure from a reproducible source.
- [x] Prepare one reproducible ParaView field figure illustrating stress and
  strain-energy output without using localization/Q statistics.
- [x] Prepare a validation figure comparing the conventional macro-response
  histories; do not use research-paper localization/Q statistics.
- [x] Draft compliant highlights, an optional graphical abstract, and a
  SoftwareX-specific cover letter; reconfirm upload requirements in the live
  submission system.
- [ ] Verify every reference and DOI against its publisher record.
- [ ] Add the archived HomiCSx software citation as a software reference.
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
- [ ] Reviewer-form audit: release-dependent metadata tables are complete and
  consistent.
- [ ] Render and visually inspect the final PDF.
- [ ] Obtain author approval of the exact files before submission.
