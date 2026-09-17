# Geometry generation

HomiCSx represents a particulate unit cell with `RVEGeometry`. A geometry may
come from the seeded periodic random sequential adsorption (RSA) generator or
be constructed explicitly from `Inclusion` objects. The resulting object is
independent of Gmsh and can be inspected before meshing.

## Random periodic geometries

`GeometryInput` selects dimension, dispersity, shape, target volume fraction,
clearance, domain size, and random seed. The supported random shapes are:

| Dimension | Round | Nonspherical |
|---|---|---|
| 2D | circle | axis-aligned ellipse |
| 3D | sphere | axis-aligned ellipsoid |

Monodisperse generation uses a fixed particle count and derives a common size
from the target volume fraction:

```python
from homicsx import GeometryInput, particulate_geometry_generator

settings = GeometryInput(
    dim=2,
    dispersion="mono",
    shape="circle",
    volume_fraction=0.20,
    num_particles=20,
    clearance=0.01,
    domain_size=(1.0, 1.0),
    seed=42,
)
geometry = particulate_geometry_generator(settings)
```

For polydisperse circles or spheres, provide `min_radius` and `max_radius`. For
polydisperse ellipses or ellipsoids, provide `axis_ratios`, `min_scale`, and
`max_scale` as well as the radius bounds required by the current validated
input schema:

```python
settings = GeometryInput(
    dim=3,
    dispersion="poly",
    shape="ellipsoid",
    volume_fraction=0.12,
    volume_fraction_tolerance=0.01,
    clearance=0.01,
    domain_size=(1.0, 1.0, 1.0),
    axis_ratios=(1.0, 1.5, 2.0),
    min_radius=0.03,
    max_radius=0.10,
    min_scale=0.03,
    max_scale=0.10,
    seed=7,
)
geometry = particulate_geometry_generator(settings)
```

The generator stores each original particle and any translated periodic images
required when it crosses a cell boundary. A fixed seed makes placement
reproducible for a fixed HomiCSx and NumPy version.

## Prescribed and rotated inclusions

Explicit construction is appropriate for benchmark cells, imported particle
descriptions, or orientations not supplied by the random generator. A 2D
orientation is one counter-clockwise angle in radians. A 3D orientation is a
tuple of rotations about the global X, Y, and Z axes, applied in that order.

```python
import numpy as np
from homicsx import Inclusion, RVEGeometry

geometry_2d = RVEGeometry(
    dim=2,
    domain_size=(1.0, 1.0),
    inclusions=[
        Inclusion(
            center=(0.5, 0.5),
            phase_id=1,
            shape="ellipse",
            radii=(0.22, 0.08),
            orientation=np.pi / 6,
        )
    ],
)

geometry_3d = RVEGeometry(
    dim=3,
    domain_size=(1.0, 1.0, 1.0),
    inclusions=[
        Inclusion(
            center=(0.5, 0.5, 0.5),
            phase_id=1,
            shape="ellipsoid",
            radii=(0.22, 0.12, 0.08),
            orientation=(0.2, -0.1, 0.4),
        )
    ],
)
```

When a prescribed inclusion crosses a periodic boundary, the caller must also
provide its translated image inclusions and set `periodic_source_id`. The
random generator creates these images automatically.

## Interphases and multiple phases

`interphase_thickness_ratio` adds a concentric inner/outer construction, and
`interphase_phase_id` assigns its material phase. Separate inclusions may use
different `phase_id` values; `RVEGeometry.phase_ids` should list all phases,
with the matrix phase first.

## Current limitations

- Independently oriented random ellipses and ellipsoids are not generated. The
  existing RSA clearance approximation is axis-aligned and must not be reused
  for arbitrary rotations.
- The outer RVE is an axis-aligned rectangle or cuboid beginning at the origin.
- Open-cell and overlapping-particle generation uses an approximate volume
  fraction correction and is outside the publication-supported core.
- The optional visualization helpers are experimental; geometry generation
  itself is part of the supported core.

See {doc}`../limitations` for the complete support boundary and
{doc}`homogenization_linear_2D` for an end-to-end generated-cell example.
