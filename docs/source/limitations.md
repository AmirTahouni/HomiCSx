# Scope and limitations

## Supported publication scope

HomiCSx 1.0.1 supports the documented end-to-end workflow for periodic
particulate geometry generation, Gmsh meshing and tagging, multiphase
linear-elastic homogenization, finite-strain hyperelastic homogenization
through a public energy-based material interface and built-in Neo-Hookean law,
finite-strain generalized-Maxwell homogenization, custom nonlinear materials,
hook-based workflow customization, and essential result extraction.

Generated meshes are periodic-conforming by default: Gmsh copies each master
boundary mesh to its translated opposite boundary. This requires geometrically
periodic opposite-side topology. HomiCSx raises an error instead of silently
creating a nonmatching mesh when the boundary entities cannot be paired.
`MeshSettings.periodic_mesh=False` is available for deliberate legacy or
nonperiodic workflows, but that mode is outside the publication-supported
homogenization path.

The tested environment is Linux, or Linux under WSL2, with the exact solver
versions in `environment.yml`. Other dependency versions and platforms may
work but are not currently part of the verification matrix.

## Unsupported or deferred behavior

- Two-dimensional FEM is plane strain only. Plane stress raises a deliberate
  `NotImplementedError` with guidance to use plane strain or a 3D model.
- Two-dimensional all-quadrilateral meshing is supported for documented
  linear and history-independent hyperelastic workflows. Generalized-Maxwell
  state updates currently require triangle or tetrahedron cells. General
  three-dimensional hexahedral meshing is not supported.
- Prescribed ellipses and ellipsoids may be rotated. The random RSA generators
  currently generate axis-aligned nonspherical particles; independently
  oriented random packing is deferred because it requires a validated exact
  collision-and-clearance algorithm.
- Overlapping-void and open-cell-foam workflows lack publication-level tests.
- Imported external meshes are outside the publication-supported workflow.
  Such meshes are not automatically made periodic-conforming.
- Generalized-Maxwell external validation covers two-dimensional plane-strain
  simple-shear relaxation for homogeneous, centered-inclusion, and periodic
  boundary-split cells, plus a homogeneous three-dimensional simple-shear
  patch. A heterogeneous 3D fast/slow-rate regression exercises time-dependent
  redistribution, but no heterogeneous 3D Abaqus comparison is currently
  provided. Other loading paths and non-Maxwell history-dependent laws have not
  received equivalent external validation. Internal viscous metrics are
  represented cellwise using DG0 coefficients on first-order simplex cells.
- XDMF displacement and reconstructed stress/energy fields are supported for
  history-independent materials. Viscoelastic XDMF stress/energy
  reconstruction is rejected because a state snapshot per output step is not
  yet stored; use a post-stress hook for state-aware extraction.
- Finite-difference homogenized tangents are supported for
  history-independent materials. History-dependent algorithmic tangents are
  not yet implemented because perturbations must not commit an additional
  relaxation step.
- HomiCSx 1.x supports solver execution on one MPI rank. The public drivers
  reject distributed communicators until cross-rank consistency is verified in
  continuous integration.
- External hyperelastic validation exercises the built-in Neo-Hookean law.
  User-defined energies use the same assembly interface, but their constitutive
  correctness remains the user's responsibility.

Private defensive branches in geometry helpers may raise `NotImplementedError`
for shapes that cannot pass the validated public inputs. These are not
advertised user workflows.

## Experimental modules

The stochastic ensemble/sweep conveniences and visualization helpers are
retained for evaluation but excluded from publication claims and the 1.x
compatibility guarantee. They may change or be removed in a future release.
See {doc}`experimental` and the repository's `PUBLIC_API.md` for details.
