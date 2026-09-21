# Loading conventions

Nonlinear built-in load cases use a scalar load parameter $a$ to construct the
macroscopic deformation gradient $\overline{F}$. The sign convention is
important: positive `a` means **compression** for the named normal and biaxial
load cases, but positive stretch for `iso_stretch`.

| Load name | Non-identity components of $\overline{F}$ |
| --- | --- |
| `uniaxial_x` | $F_{11}=1-a$ |
| `uniaxial_y` | $F_{22}=1-a$ |
| `uniaxial_z` (3D) | $F_{33}=1-a$ |
| `biaxial` | $F_{11}=F_{22}=1-a$ |
| `triaxial` (3D) | $F_{11}=F_{22}=F_{33}=1-a$ |
| `shear_xy` | $F_{12}=a$ |
| `shear_xz` (3D) | $F_{13}=a$ |
| `shear_yz` (3D) | $F_{23}=a$ |
| `iso_stretch` | every normal component is $1+a$ |

Use `custom_loads` when a different convention or loading path is required.
For example, uniaxial extension in the x direction can be supplied as:

```python
def uniaxial_extension(a):
    return np.diag([1.0 + a, 1.0])

result = driver.run(
    custom_loads={"uniaxial_extension": uniaxial_extension},
    from_built_in_loads=[],
)
```

The solver reports the deformation gradient actually used at each converged
step in `history["Fbar"]`; this is the authoritative record for reproducibility.

Malformed custom-load outputs are rejected before the nonlinear solve. A load
callable must return a finite `(dim, dim)` deformation-gradient array; the same
validation is repeated after pre-step hooks modify the target.

When `csv_opt=True`, `output_prefix` prefixes the macroscopic and state-history
CSV files. State histories use one row per load, phase, cell, quadrature point,
variable, and component, so phases may expose different state variables. The
same prefix is applied to XDMF output when `xdmf_opt=True`.
