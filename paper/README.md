# SoftwareX submission package

This directory contains the working source for the HomiCSx software paper.
The target article type is **Original Software Publication** in SoftwareX.

The current journal guide requires use of the official SoftwareX template,
limits the main text to 3000 words (excluding title, author information,
references, and metadata tables), permits at most six figures, and requires at
least one of the template's software metadata tables. The editable source is
maintained in `manuscript.md`. A compiled LaTeX draft is provided as
`manuscript.tex` and uses Elsevier's `elsarticle` class with the SoftwareX
Original Software Publication section and metadata structure. Before
submission, compare it once more with the exact template offered by the live
Guide for Authors and submission system.

Compile from this directory with MiKTeX or TeXstudio by running `pdflatex`
twice on `manuscript.tex`. The checked-in `manuscript.pdf` is the visually
inspected draft output; its release and archive identifiers remain pending.

Do not submit this draft until all items in `submission_checklist.md` are
resolved, a versioned release has been archived with a DOI, and the manuscript
has been transferred into the then-current official template.
