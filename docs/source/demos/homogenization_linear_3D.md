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

# Linear 3D homogenization demo

```python
from homicsx import(
    GeometryInput,
    PhysicalTags, 
    MeshSettings, 
    LinearElasticIsotropic, 
    MaterialAssignment, 
    ProblemSettings
)
from homicsx.geometry import particulate_geometry_generator
from homicsx.visualization import visualize_geometry
from homicsx.mesh import generate_mesh
from homicsx.homogenization.driver import LinearHomogenizationDriver
from homicsx.utils import (
    extract_effective_moduli_3d,
    mori_tanaka,
    hashin_shtrikman,
)

import numpy as np
```

Generate 3D mono-disperse unit-cell geometry with spherical inclusions.

```python
geometry_input = GeometryInput(
    dim=3,
    dispersion="mono",
    volume_fraction=0.1,
    num_particles=80,
    clearance=0.015,
    domain_size=(1, 1, 1),
    shape="sphere",
    seed=42,
)

geometry = particulate_geometry_generator(geometry_input)

visualize_geometry(geometry)
```

Generate physical tagging convention, mesh settings, and the mesh entities, including the dolfinx mesh, and dolfinx compatible cell tags and facet tags. The `PhysicalTags` object contains the tagging convention which will later be used for identification of mesh entities throughout the the code.

The `MeshSettings` data object is fed to the top level meshing function, with the cell tags, facet tags, and the mesh itself as the output.

```python
physical_tags = PhysicalTags()

mesh_settings = MeshSettings(
    min_size=0.03,
    max_size=0.05,
    physical_tags=physical_tags,
)

mesh, ct, ft = generate_mesh(
    geometry=geometry,
    mesh_settings=mesh_settings,
)
```

For material assignment, the corresponding materials are first defined. Then, the `MaterialAssignment` object is made. The material assignment is later paired with the `PhysicalTags` object for physical tag assignment in the FE problem definition. By default, the the matrix identified with id=0.

```python
E_mat = 1.0
nu_mat = 0.2
mat_matrix = LinearElasticIsotropic(young_modulus=E_mat, poisson_ratio=nu_mat)

E_inc = 50.0
nu_inc = 0.2
mat_particle = LinearElasticIsotropic(young_modulus=E_inc, poisson_ratio=nu_inc)

material_assignment = MaterialAssignment(
    materials_by_phase={
        0: mat_matrix,
        1: mat_particle,
    }
)
```
The FE problem is then defined via a `ProblemSettings` data object.

```python
fem_settings = ProblemSettings(
    dim = geometry_input.dim,
    kinematics='small_strain',
    element_family='Lagrange',
    element_degree=1,
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
```

Then, homogenization elements, including the mesh elements, material assignment, problem settings, and other data is used to initialize the `LinearDriver`, which is then used to acquire the homogenization solution.

```python
linear_driver = LinearHomogenizationDriver(
    mesh_obj=mesh,
    cell_tags=ct,
    facet_tags=ft,
    assignment=material_assignment,
    settings=fem_settings,
    physical_tags=physical_tags,
    domain_size=geometry_input.domain_size,
    matrix_phase_id=0,
    mode='complete',
)
result = linear_driver.run()
```

Lastly, the homogenization result can be used for validation and post-processing. HomiCSx includes functions that calculate the Hashin-Shtrikman analytical bounds and the Mori-Tanaka estimates.

```python
C_hom_mt = mori_tanaka(
    E_m=E_mat,
    nu_m=nu_mat,
    E_f=E_inc,
    nu_f=nu_inc,
    vol_f=geometry_input.volume_fraction,
    dim=3,
)

C_hom = result.C_hom
print('C_hom: ')
with np.printoptions(suppress=True, precision=3):
    print(C_hom)

print('\nC_hom_mt: ')
with np.printoptions(suppress=True, precision=3):
    print(C_hom_mt)

kappa_mt, mu_mt, E_mt, nu_mt = extract_effective_moduli_3d(C_hom_mt)
kappa_hom, mu_hom, E_hom, nu_hom = extract_effective_moduli_3d(C_hom)

hs_bounds = hashin_shtrikman(
    E_m=E_mat,
    nu_m=nu_mat,
    E_f=E_inc,
    nu_f=nu_inc,
    vol_f=geometry_input.volume_fraction,
    dim=3,
)

E_hs_lower = hs_bounds['E'][0]
E_hs_upper = hs_bounds['E'][1]

print('\nsummary:')
print('-'*20)
print('Young modulus:')
print(f'homogenization result: {E_hom}')
print(f'Mori-Tanaka estimate: {E_mt}')
print(f'Hashin-Shtrikman upper bound: {E_hs_upper}')
print(f'Hashin-Shtrikman lower bound: {E_hs_lower}')
print('-'*20)
print('Poison ratio:')
print(f'homogenization result: {nu_hom}')
print(f'Mori-Tanaka estimate: {nu_mt}')
```
