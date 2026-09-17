# Nonlinear hyperelastic homogenization

This deterministic example solves a two-phase periodic cell in finite simple
shear. Both phases use the built-in compressible `NeoHookeanIsotropic` law, so
the example exercises the publication-supported material path rather than a
notebook-only substitute.

The built-in strain-energy density is

$$
\Psi(\mathbf F)=\frac{\mu}{2}
\left(I_1-d-2\ln J\right)+\frac{\lambda}{2}(J-1)^2,
$$

where $d$ is the modeled dimension. For 2D plane strain this is identical to
the three-dimensional expression evaluated with $F_{33}=1$. The implementation
requires $J=\det\mathbf F>0$.

Run the example from the repository root:

```bash
python examples/hyperelastic_periodic_2d.py
```

The test suite executes the same `run_example()` function and checks the step
count and finite positive macroscopic shear stress and energy.

```{literalinclude} ../../../examples/hyperelastic_periodic_2d.py
:language: python
:linenos:
```
