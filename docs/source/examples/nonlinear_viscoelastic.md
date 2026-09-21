# Nonlinear viscoelastic homogenization

This deterministic example places a hyperelastic inclusion in a
generalized-Maxwell matrix, applies a held finite simple shear, and records the
relaxing macroscopic first Piola shear stress. It also demonstrates a typed
post-stress hook and checks that the hook observes the same history returned by
the driver.

Run it from the repository root:

```bash
python examples/viscoelastic_periodic_2d.py
```

The test suite executes the same `run_example()` function and requires a
decreasing stress history and exact agreement between the hook-collected and
driver-returned values. See the validation page for the separate Abaqus
comparisons; this example is a usage demonstration, not the validation itself.

```{literalinclude} ../../../examples/viscoelastic_periodic_2d.py
:language: python
:linenos:
```
