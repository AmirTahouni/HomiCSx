# Maintained examples

This is the single repository-level home for runnable HomiCSx examples. The
scripts use fixed geometries and coarse meshes so their complete workflows can
serve as reproducibility checks. Narrative explanations and additional code
snippets are maintained in the documentation rather than duplicated in
notebooks.

From the repository root, run:

```bash
python -m examples.linear_periodic_2d
python -m examples.hyperelastic_periodic_2d
python -m examples.viscoelastic_periodic_2d
```

The scripts print small JSON summaries and do not open plots or write solver
fields. These three serial workflows are exercised by the automated test suite.
The linear MPI smoke test can be run with:

```bash
mpiexec -n 2 python examples/mpi_linear_smoke.py
```

Nonlinear MPI is not currently supported. The finer validation models and
independent Abaqus comparisons are under `validation/abaqus/`.

The notebooks formerly stored in the root-level `demos/` directory were
removed before version 1.0 because they duplicated these workflows and included
stale, experimental, or research-specific code. They remain available in the
Git history but are not part of the supported release surface.

