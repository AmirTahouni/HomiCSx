# Nonlinear hyperelastic homogenization

This deterministic example solves a two-phase periodic cell in finite simple
shear. Both phases use the built-in compressible `NeoHookeanIsotropic` law, so
the example exercises the publication-supported material path directly.

The nonlinear assembly is not limited to this formula. Subclass
`HyperelasticMaterial` to provide a UFL `psi_form(F)` and matching numerical
energy and first-Piola stress methods for macroscopic post-processing. HomiCSx
automatically differentiates the UFL energy in the equilibrium residual. The
built-in law is the externally validated reference implementation; users must
verify their own constitutive functions.

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

Run this example on one MPI rank. Nonlinear multipoint-constraint solves are
not currently supported in distributed execution; see {doc}`../limitations`.

```{literalinclude} ../../../examples/hyperelastic_periodic_2d.py
:language: python
:linenos:
```
