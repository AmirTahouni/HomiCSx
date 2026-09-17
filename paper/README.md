# SoftwareX submission package

This directory contains the working source for the HomiCSx software paper.
The target article type is **Original Software Publication** in SoftwareX.

The manuscript was compared against the official SoftwareX article template,
Version 6 (March 2026). That template limits the main text to 4000 words,
permits at most six figures, requires five named main sections, and prescribes
a C1--C8 software metadata table. The current draft has a 98-word abstract,
approximately 1750 main-body words, four figures, all required sections, and a
complete C1--C8 table. The editable source is maintained in `manuscript.md`;
`manuscript.tex` uses Elsevier's `elsarticle` class and preserves the required
SoftwareX structure.

Compile from this directory with MiKTeX or TeXstudio by running `xelatex`
twice on `manuscript.tex`. The checked-in `manuscript.pdf` is the visually
inspected draft output and cites the archived version DOI and concept DOI.

Do not submit this draft until all remaining items in `submission_checklist.md`
are resolved. Reconfirm the template version in the live submission system
immediately before upload.
