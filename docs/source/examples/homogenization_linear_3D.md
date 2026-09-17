# Linear 3D homogenization

This deterministic example constructs a centered spherical inclusion, creates
a periodic-conforming tetrahedral mesh, assigns two isotropic linear phases,
and computes the complete 6-by-6 effective stiffness tensor.

Run it from the repository root:

```bash
python examples/linear_periodic_3d.py
```

The automated test suite checks the matrix shape, positive trace, tetrahedral
cell type, and a relative stiffness-symmetry error below 2%. The coarse mesh is
intentional: this is a fast API workflow check, not a mesh-converged benchmark.
See {doc}`../validation` for quantitative verification evidence.

```{literalinclude} ../../../examples/linear_periodic_3d.py
:language: python
:linenos:
```
