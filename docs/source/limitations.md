# Scope and limitations

## Supported publication scope

HomiCSx 0.1.0 supports the documented end-to-end workflow for periodic
particulate geometry generation, Gmsh meshing and tagging, multiphase
linear-elastic homogenization, finite-strain Neo-Hookean homogenization,
custom nonlinear materials, hook-based workflow customization, and essential
result extraction.

The tested environment is Linux, or Linux under WSL2, with the exact solver
versions in `environment.yml`. Other dependency versions and platforms may
work but are not currently part of the verification matrix.

## Unsupported or deferred behavior

- Two-dimensional FEM is plane strain only. Plane stress raises a deliberate
  `NotImplementedError` with guidance to use plane strain or a 3D model.
- Quad and hex meshing are not supported across the advertised geometry range.
- Overlapping-void and open-cell-foam workflows lack publication-level tests.
- Imported external meshes are outside the publication-supported workflow.
- Finite-strain viscoelasticity and state evolution remain experimental.
- Advanced post-processing beyond essential result and XDMF extraction is not
  part of the supported core.

Private defensive branches in geometry helpers may raise `NotImplementedError`
for shapes that cannot pass the validated public inputs. These are not
advertised user workflows.

## Experimental modules

The stochastic ensemble/sweep conveniences and visualization helpers are
retained for compatibility but excluded from publication claims. They may
change or be removed before 1.0. See {doc}`experimental` and the repository's
`PUBLIC_API.md` for details.
