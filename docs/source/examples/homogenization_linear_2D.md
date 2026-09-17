# Linear 2D homogenization

This deterministic example constructs a two-phase cell, generates a
periodic-conforming Gmsh mesh, assigns isotropic linear materials, and computes
the complete plane-strain effective stiffness tensor. Pass
`quadrilateral=True` to `run_example` to exercise the supported 2D
all-quadrilateral path.

Run it from the repository root:

```bash
python examples/linear_periodic_2d.py
```

The automated test suite requires a finite positive stiffness, the expected
3-by-3 matrix shape, and a relative symmetry error below 2%. A separate test
runs the quadrilateral option and confirms the resulting cell type.

```{literalinclude} ../../../examples/linear_periodic_2d.py
:language: python
:linenos:
```
