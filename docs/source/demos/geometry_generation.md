---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: testEnv
    language: python
    name: python3
---

# Geometry generation demo

```python
from homicsx import GeometryInput
from homicsx.geometry import patriculate_geometry_generator
from homicsx.visualization import visualize_geometry
```

Generate 3D mono-disperse unit-cell geometry with spherical inclusions

```python
geometry_input = GeometryInput(
    dim=3,
    dispersion="mono",
    volume_fraction=0.1,
    num_particles=3,
    clearance=0.015,
    domain_size=(1, 1, 1),
    shape="sphere",
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Generate 3D mono-disperse unit-cell geometry with axis-aligned ellipsoidal inclusions

```python
geometry_input = GeometryInput(
    dim=3,
    dispersion="mono",
    volume_fraction=0.1,
    num_particles=5,
    clearance=0.015,
    domain_size=(1, 1, 1),
    shape="ellipsoid",
    axis_ratios=(1, 2, 3)
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Generate 3D poly-disperse unit-cell geometry with spherical inclusions

```python
geometry_input = GeometryInput(
    dim=3,
    dispersion="poly",
    volume_fraction=0.1,
    volume_fraction_tolerance=0.01,
    clearance=0.015,
    domain_size=(1, 1, 1),
    shape="sphere",
    min_radius=0.05,
    max_radius=0.2,
    min_scale=0.1,
    max_scale=0.2
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Generate 3D poly-disperse unit-cell geometry with axis-aligned ellipsoidal inclusions

```python
geometry_input = GeometryInput(
    dim=3,
    dispersion="poly",
    volume_fraction=0.1,
    volume_fraction_tolerance=0.01,
    clearance=0.015,
    domain_size=(1, 1, 1),
    shape="ellipsoid",
    axis_ratios=(1, 2, 1),
    min_radius=0.05,
    max_radius=0.2,
    min_scale=0.1,
    max_scale=0.3
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Generate 2D mono-disperse unit-cell geometry with circular inclusions

```python
geometry_input = GeometryInput(
    dim=2,
    dispersion="mono",
    volume_fraction=0.2,
    num_particles=5,
    clearance=0.015,
    domain_size=(1, 1),
    shape="circle",
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Generate 2D mono-disperse unit-cell geometry with axis-aligned elliptical inclusions

```python
geometry_input = GeometryInput(
    dim=2,
    dispersion="mono",
    volume_fraction=0.2,
    num_particles=5,
    clearance=0.02,
    domain_size=(1, 1),
    shape="ellipse",
    axis_ratios=(2, 1), # GMSH occ limitation: major radius rx must be larger than minor radius ry.
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Generate 2D poly-disperse unit-cell geometry with circular inclusions

```python
geometry_input = GeometryInput(
    dim=2,
    dispersion="poly",
    volume_fraction=0.2,
    volume_fraction_tolerance=0.01,
    clearance=0.015,
    domain_size=(1, 1),
    shape="circle",
    min_radius=0.05,
    max_radius=0.15,
    min_scale=0.1,
    max_scale=0.2
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Generate 2D poly-disperse unit-cell geometry with axis-aligned elliptical inclusions

```python
geometry_input = GeometryInput(
    dim=2,
    dispersion="poly",
    volume_fraction=0.2,
    volume_fraction_tolerance=0.01,
    clearance=0.015,
    domain_size=(1, 1),
    shape="ellipse",
    axis_ratios=(2, 1),
    min_radius=0.02,
    max_radius=0.1,
    min_scale=0.1,
    max_scale=0.5
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Open-cell foam geometry generation

```python
geometry_input = GeometryInput(
    dim=2,
    dispersion="poly",
    volume_fraction=0.85,
    volume_fraction_tolerance=0.01,
    clearance=0.005,
    domain_size=(1, 1),
    shape="ellipse",
    axis_ratios=(1.4, 1),
    min_radius=0.02,
    max_radius=0.1,
    min_scale=0.02,
    max_scale=0.04,
    allow_overlap=True,
)

geometry = patriculate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```
