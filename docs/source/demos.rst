Demos
=====

Start with the deterministic examples below. Their ``run_example`` functions
are executed by the automated test suite, so they stay aligned with the public
API and supported solver behavior.

.. list-table:: Tested examples
   :header-rows: 1
   :widths: 30 70

   * - Demo
     - Description
   * - :doc:`demos/geometry_generation`
     - Seeded particulate geometry generation
   * - :doc:`demos/homogenization_linear_2D`
     - Linear periodic homogenization in 2D
   * - :doc:`demos/homogenization_linear_3D`
     - Linear periodic homogenization in 3D
   * - :doc:`demos/nonlinear_hyperelastic`
     - Built-in Neo-Hookean finite-strain homogenization
   * - :doc:`demos/nonlinear_viscoelastic`
     - Heterogeneous generalized-Maxwell relaxation and typed hooks

Extended and experimental notebooks
-----------------------------------

These notebooks illustrate research-oriented or experimental workflows. They
are retained as useful examples but are not all automated release gates.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Demo
     - Description
   * - :doc:`demos/damage_with_hook`
     - A custom damage-hook pattern, not a built-in damage constitutive model
   * - :doc:`demos/ensemble_homogenization`
     - Experimental stochastic ensemble convenience API
   * - :doc:`demos/sweep_studies`
     - Experimental parameter-sweep convenience API
   * - :doc:`demos/convergence_mesh_size`
     - Exploratory mesh-size convergence workflow
   * - :doc:`demos/convergence_num_particles`
     - Exploratory particle-count convergence workflow

.. toctree::
   :hidden:

   demos/geometry_generation
   demos/homogenization_linear_2D
   demos/homogenization_linear_3D
   demos/nonlinear_hyperelastic
   demos/nonlinear_viscoelastic
   demos/damage_with_hook
   demos/ensemble_homogenization
   demos/sweep_studies
   demos/convergence_mesh_size
   demos/convergence_num_particles
