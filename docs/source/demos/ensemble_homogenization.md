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

# Ensemble homogenization demo

```python
from homicsx import(
    GeometryInput, 
    PhysicalTags, 
    MeshSettings, 
    LinearElasticIsotropic, 
    MaterialAssignment, 
    ProblemSettings
)
from homicsx.stochastic import perform_ensemble_study
```

Preparing geometry input for 2D mono-disperse unit-cell geometry with spherical inclusions

```python
geometry_input = GeometryInput(
    dim=2,
    dispersion="mono",
    volume_fraction=0.2,
    num_particles=30,
    clearance=0.015,
    domain_size=(1, 1),
    shape="circle",
)
```

Preparing physical tagging convention and mesh settings

```python
physical_tags = PhysicalTags()

mesh_settings = MeshSettings(
    min_size=0.025,
    max_size=0.035,
    physical_tags=physical_tags,
)
```

Defining the material assignment

```python
E_mat = 1.0
nu_mat = 0.3
mat_matrix = LinearElasticIsotropic(young_modulus=E_mat, poisson_ratio=nu_mat)

E_inc = 100.0
nu_inc = 0.3
mat_particle = LinearElasticIsotropic(young_modulus=E_inc, poisson_ratio=nu_inc)

material_assignment = MaterialAssignment(
    materials_by_phase={
        0: mat_matrix,
        1: mat_particle,
    }
)
```

Preparing the FEM solver settings

```python
fem_settings = ProblemSettings(
    dim = geometry_input.dim,
    kinematics='small_strain',
    two_dimensional_formulation='plane_strain',
    element_family='Lagrange',
    element_degree=1,
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
```

Performing ensemble study

```python
ensemble_solution = perform_ensemble_study (
    ensemble_size=100,
    geometry_input=geometry_input,
    mesh_settings=mesh_settings,
    Physical_tags=physical_tags,
    assignment=material_assignment,
    fem_settings=fem_settings,
    matrix_phase_id=0,
    homogenization_solver="linear",
    homogenization_mode="partial",
)
```

Printing the ensemble result summary

```python
ensemble_solution.print_summary()
```

Visualizing the result moduli as a histogram chart

```python
ensemble_solution.visualize_moduli_histogram(num_bins=10)
```
