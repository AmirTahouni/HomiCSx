# Scriptable examples

These compact examples complement the narrative notebooks in `demos/`. They
use fixed geometries and coarse meshes so that the complete workflows can be
executed as reproducibility checks.

From the repository root, run:

```bash
python -m examples.linear_periodic_2d
python -m examples.viscoelastic_periodic_2d
```

The scripts print small JSON summaries and do not open plots or write solver
fields. They are exercised by the automated test suite. The finer validation
models and independent Abaqus comparisons are under `validation/abaqus/`.

