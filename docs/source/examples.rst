Examples
========

The repository-level ``examples/`` directory is the single maintained home for
runnable workflows. Each serial script exposes a ``run_example`` function that
is executed by the automated test suite. The pages below explain those scripts
and the supported geometry workflow.

.. list-table:: Maintained examples
   :header-rows: 1
   :widths: 32 68

   * - Example
     - Description
   * - :doc:`examples/geometry_generation`
     - Seeded generation and explicitly prescribed rotated inclusions
   * - :doc:`examples/homogenization_linear_2D`
     - Linear periodic homogenization with triangle and quadrilateral coverage
   * - :doc:`examples/homogenization_linear_3D`
     - Linear periodic homogenization on a tetrahedral 3D cell
   * - :doc:`examples/nonlinear_hyperelastic`
     - Finite-strain hyperelastic homogenization and custom-energy interface
   * - :doc:`examples/nonlinear_viscoelastic`
     - Heterogeneous generalized-Maxwell relaxation and a typed hook

Experimental stochastic and visualization conveniences are documented only as
experimental interfaces; they are not represented as supported tutorials. See
:doc:`experimental` and :doc:`limitations`.

.. toctree::
   :hidden:

   examples/geometry_generation
   examples/homogenization_linear_2D
   examples/homogenization_linear_3D
   examples/nonlinear_hyperelastic
   examples/nonlinear_viscoelastic
