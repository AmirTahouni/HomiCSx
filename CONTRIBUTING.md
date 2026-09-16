# Contributing to HomiCSx

Bug fixes, documentation improvements, focused tests, and reproducibility
improvements are welcome. For a substantial feature or API change, please open
an issue before investing significant work so that its scope and validation can
be discussed.

## Development setup

HomiCSx uses compiled FEniCSx dependencies, so create the tested Conda
environment rather than installing the solver stack with `pip`:

```bash
conda env create -f environment-dev.yml
conda activate homicsx_dev
python -m pip install --no-deps -e .
python -m pytest -q
```

Please keep changes focused, add or update tests where behavior changes, and
update user documentation when the public interface changes. Numerical changes
should state the expected mechanical behavior, tolerances, and the evidence used
to select them. Reference data must include enough provenance to reproduce it.

## Project scope and review

The publication-supported scope is described in `PUBLICATION_READINESS.md`.
The `homicsx.stochastic` and `homicsx.visualization` modules are experimental;
changes to them do not imply that they have entered the supported API.

HomiCSx currently follows a sustainability-oriented maintenance model rather
than an active feature roadmap. Contributions will be considered as maintainer
availability permits, and no review or release timetable is guaranteed.

By contributing, you agree that your contribution is provided under the MIT
License included in this repository.
