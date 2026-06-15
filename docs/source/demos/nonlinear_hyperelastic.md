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

# Nonlinear hyperelastic homogenization

```python
from __future__ import annotations

from dataclasses import dataclass

import petsc4py.PETSc as PETSc
import numpy as np
from ufl import det, tr

from homicsx import(
    Inclusion,
    RVEGeometry, 
    PhysicalTags, 
    MeshSettings,
    MaterialAssignment,
    ProblemSettings,
)
from homicsx.core.material import HyperelasticMaterial, MaterialState
from homicsx.mesh import generate_mesh
from homicsx.core.homogenization import AdaptiveSettings
from homicsx.homogenization.driver import NonlinearHomogenizationDriver
```

Custom geometry and mesh generation

```python
dim = 2
domain_size = (1, 1)

inclusion = Inclusion(
    center=[0.5, 0.5],
    phase_id=1,
    shape='circle',
    radii=0.2,
)

geometry = RVEGeometry(
    dim=dim,
    domain_size=domain_size,
    inclusions=[inclusion],
)

physical_tags = PhysicalTags()

mesh_settings = MeshSettings(
    min_size=0.02,
    max_size=0.04,
    physical_tags=physical_tags,
    quad_hex=True,
)

domain, ct, ft = generate_mesh(
    geometry=geometry,
    mesh_settings=mesh_settings,
)
```

Custom hyperelastic material definition

```python
@dataclass
class NeoHookeanDecoupled(HyperelasticMaterial):
    def __init__(self, kappa, mu):
        self.kappa = kappa
        self.mu = mu
    
    def psi_form(self, F):
        dim = F.ufl_shape[0]
        C = F.T * F
        J = det(F)
        I1 = J**(-2/3) * (tr(C) + 1)
        return self.kappa/2 * (J - 1)**2 + self.mu/2 * (I1 - 3)
    
    def evaluate_energy(self, F: np.ndarray, dim: int) -> float:
        """Numeric evaluation matching psi_form."""
        J = np.linalg.det(F)
        C = F.T @ F
        tr_C = np.trace(C)
        if dim == 2:
            I1 = J**(-2/3) * (tr_C + 1)
        else:
            I1 = J**(-2/3) * tr_C
        return self.kappa/2 * (J - 1)**2 + self.mu/2 * (I1 - 3)
    
    def get_quadrature_point_stress(self, state: MaterialState, F: np.ndarray, quad_point_idx: int) -> np.ndarray:
        dim = F.shape[0]
        J = np.linalg.det(F)
        Finv = np.linalg.inv(F)
        FinvT = Finv.T
        
        C = F.T @ F
        tr_C = np.trace(C)
        
        P_iso = self.mu * J**(-2.0/3.0) * (F - (1.0/3.0) * tr_C * FinvT)
        P_vol = self.kappa * J * (J - 1.0) * FinvT
        
        return P_iso + P_vol

kappa_mat = 17.5
mu_mat = 8.0
mat_matrix = NeoHookeanDecoupled(kappa=kappa_mat, mu=mu_mat)

kappa_inc = 100 * kappa_mat
mu_inc = 100 * mu_mat
mat_inclusion = NeoHookeanDecoupled(kappa=kappa_inc, mu=mu_inc)

material_assignment = MaterialAssignment(
    materials_by_phase={
        0: mat_matrix,
        1: mat_inclusion,
    }
)
```

Solution with custom loading

```python
fem_settings = ProblemSettings(
    dim = dim,
    kinematics='finite_strain',
    two_dimensional_formulation='plane_strain',
    element_family='P',
    element_degree=1,
    petsc_options={
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 10,
    }
)

def uni_tension_func(a):
    F = np.eye(2, dtype=PETSc.ScalarType)
    F[0, 0] = 1.0 + a
    return F

driver = NonlinearHomogenizationDriver(
    mesh_obj=domain,
    cell_tags=ct,
    facet_tags=ft,
    assignment=material_assignment,
    settings=fem_settings,
    physical_tags=physical_tags,
    domain_size=domain_size,
    matrix_phase_id=0,
    quad_degree=4,
    enable_hooks=False,
)

result = driver.run(
    tangent_every=1,
    output_prefix='rve',
    max_strain=0.4,
    custom_loads={
        'tension': uni_tension_func,
    },
    from_built_in_loads=[],
    adaptive_settings=AdaptiveSettings(
        initial_step_ratio=0.08,
        min_step= 1e-6,
        target_iters_min=4,
        target_iters_max=8,
        cutback_factor=0.8,
        growth_factor=1.3,
    ),
    xdmf_opt=False,
    csv_opt=False,
    plot_summary=True,
    plot_individual=True,
)

print(result.summary)
```
