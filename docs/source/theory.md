# Theory

## First-Order Computational Homogenization

HomiCSx implements first-order computational homogenization for heterogeneous materials based on Representative Volume Elements (RVEs). Given scale separation $l_\mu \ll l_M$, the method replaces an explicit heterogeneous microstructure with an equivalent homogeneous continuum whose effective properties are determined from microscale finite element solutions.

The RVE domain $\Omega \subset \mathbb{R}^d$ (with $d=2,3$) must be statistically representative. Periodic boundary conditions are adopted throughout, providing faster convergence with RVE size compared to linear displacement or uniform traction conditions.

## Kinematic Decomposition

The microscopic displacement field decomposes into a linear macroscopic part and a periodic fluctuation:

$$\mathbf{u}(\mathbf{x}) = \bar{\boldsymbol{\varepsilon}} \cdot \mathbf{x} + \tilde{\mathbf{u}}(\mathbf{x})$$

where $\bar{\boldsymbol{\varepsilon}}$ is the prescribed macroscopic strain tensor (symmetric) and $\tilde{\mathbf{u}}$ is periodic:

$$\tilde{\mathbf{u}}(\mathbf{x}^+) = \tilde{\mathbf{u}}(\mathbf{x}^-) \quad \forall \ \text{pairs on opposite boundaries}$$

This decomposition satisfies the strain averaging theorem:

$$\langle \boldsymbol{\varepsilon} \rangle = \frac{1}{|\Omega|} \int_\Omega \boldsymbol{\varepsilon} \, d\Omega = \bar{\boldsymbol{\varepsilon}}$$

### Periodic Boundary Conditions and MPC Enforcement

Periodic boundary conditions are enforced by constraining the fluctuation field on opposite boundary faces. For an RVE occupying $[0, L]^d$, the boundary is partitioned into pairs of opposite faces. The fluctuation periodicity condition reads:

$$\tilde{\mathbf{u}}(\mathbf{x}^+) - \tilde{\mathbf{u}}(\mathbf{x}^-) = \mathbf{0}$$

where $\mathbf{x}^+$ and $\mathbf{x}^-$ are corresponding points on opposite faces.

To apply this in a finite element context, the mesh must be periodic-conforming—nodes on opposite faces must match one-to-one. HomiCSx ensures this through GMSH's periodic mesh generation. For each master-slave node pair, the constraint eliminates the slave degrees of freedom from the system. If $\mathbf{u}_m$ and $\mathbf{u}_s$ are the displacement vectors of a master and slave node respectively, the total displacement constraint is:

$$\mathbf{u}_m - \mathbf{u}_s = \bar{\boldsymbol{\varepsilon}} \cdot (\mathbf{x}_m - \mathbf{x}_s)$$

Since $\mathbf{u} = \bar{\boldsymbol{\varepsilon}} \cdot \mathbf{x} + \tilde{\mathbf{u}}$, this reduces to $\tilde{\mathbf{u}}_m = \tilde{\mathbf{u}}_s$.

The MPCs are applied directly to the assembled stiffness matrix and load vector via master-slave condensation. For a constraint $\mathbf{u}_s = \mathbf{u}_m + \mathbf{c}$ (where $\mathbf{c} = \bar{\boldsymbol{\varepsilon}} \cdot (\mathbf{x}_s - \mathbf{x}_m)$), the system

$$\begin{bmatrix} \mathbf{K}_{mm} & \mathbf{K}_{ms} \\ \mathbf{K}_{sm} & \mathbf{K}_{ss} \end{bmatrix} \begin{bmatrix} \mathbf{u}_m \\ \mathbf{u}_s \end{bmatrix} = \begin{bmatrix} \mathbf{f}_m \\ \mathbf{f}_s \end{bmatrix}$$

is reduced by eliminating $\mathbf{u}_s$, adding $\mathbf{K}_{ms} + \mathbf{K}_{sm}^T + \mathbf{K}_{ss}$ into the master block and $\mathbf{K}_{ss} \mathbf{c}$ into the load vector.

Corner nodes belong to multiple face pairs and require special treatment to avoid over-constraining. HomiCSx identifies corners and assigns a single master for each, condensing all dependent constraints.

## Linear Homogenization

### Governing Equations

For linear elastic constituents, the constitutive law at any material point is:

$$\boldsymbol{\sigma}(\mathbf{x}) = \mathbb{C}(\mathbf{x}) : \boldsymbol{\varepsilon}(\mathbf{x})$$

where $\boldsymbol{\varepsilon} = \nabla^s \mathbf{u}$ is the infinitesimal strain. The effective macroscopic relation is:

$$\bar{\boldsymbol{\sigma}} = \bar{\mathbb{C}} : \bar{\boldsymbol{\varepsilon}}, \quad \bar{\boldsymbol{\sigma}} = \langle \boldsymbol{\sigma} \rangle$$

### Weak Form

The equilibrium equation $\nabla \cdot \boldsymbol{\sigma} = \mathbf{0}$ in the absence of body forces leads to the weak formulation: find $\tilde{\mathbf{u}} \in \mathcal{V}_{per}$ such that

$$\int_\Omega \nabla^s \delta \mathbf{v} : \mathbb{C} : \nabla^s \tilde{\mathbf{u}} \, d\Omega = -\int_\Omega \nabla^s \delta \mathbf{v} : \mathbb{C} : \bar{\boldsymbol{\varepsilon}} \, d\Omega \quad \forall \delta \mathbf{v} \in \mathcal{V}_{per}$$

The left-hand side defines the stiffness matrix, the right-hand side the load vector. FEniCSx assembles both from the UFL formulation and applies MPCs to enforce periodicity.

### Assembly of the Effective Stiffness Tensor

The fourth-order effective stiffness $\bar{\mathbb{C}}$ is built column by column. For 3D problems, six elementary macroscopic strain states are applied (three normal, three shear). In Voigt notation:

$$\bar{\boldsymbol{\varepsilon}}^{(1)} = [1,0,0,0,0,0]^T, \ \bar{\boldsymbol{\varepsilon}}^{(2)} = [0,1,0,0,0,0]^T, \ \dots, \ \bar{\boldsymbol{\varepsilon}}^{(6)} = [0,0,0,0,0,1]^T$$

For 2D, three states suffice. For each case $k$, the microscopic problem is solved and macroscopic stress computed by volume averaging:

$$\bar{\boldsymbol{\sigma}}^{(k)} = \frac{1}{|\Omega|} \int_\Omega \boldsymbol{\sigma}^{(k)} \, d\Omega$$

The stiffness column follows directly:

$$\bar{C}_{ik} = \bar{\sigma}_i^{(k)}$$

### Effective Moduli and Validation

For isotropic composites, effective bulk modulus $\bar{K}$ and shear modulus $\bar{G}$ are extracted from $\bar{\mathbb{C}}$ and validated against:

**Voigt bound** (isostrain, upper):
$$\bar{K}_V = \sum_r \phi_r K_r, \quad \bar{G}_V = \sum_r \phi_r G_r$$

**Reuss bound** (isostress, lower):
$$\bar{K}_R = \left( \sum_r \frac{\phi_r}{K_r} \right)^{-1}, \quad \bar{G}_R = \left( \sum_r \frac{\phi_r}{G_r} \right)^{-1}$$

**Hashin-Shtrikman bounds** for two-phase composites:
$$\bar{K}^{HS\pm} = K_m + \frac{\phi_i}{(K_i - K_m)^{-1} + 3(1 - \phi_i)/(3K_m + 4G_m)}$$

**Mori-Tanaka estimate**:
$$\bar{\mathbb{C}}^{MT} = \mathbb{C}_m + \phi_i (\mathbb{C}_i - \mathbb{C}_m) : \mathbb{A}_i^{dilute} : [ (1-\phi_i)\mathbb{I} + \phi_i \mathbb{A}_i^{dilute} ]^{-1}$$

where $\mathbb{A}_i^{dilute}$ is the dilute strain concentration tensor from Eshelby's solution.

## Nonlinear Homogenization

### Finite Strain Kinematics

The deformation gradient maps the RVE from reference $\Omega_0$ to current $\Omega$:

$$\mathbf{F}(\mathbf{X}) = \nabla_{\mathbf{X}} \boldsymbol{\varphi} = \mathbf{I} + \nabla_{\mathbf{X}} \mathbf{u}$$

Analogous to the linear case, it decomposes as:

$$\mathbf{F}(\mathbf{X}) = \bar{\mathbf{F}} + \tilde{\mathbf{F}}(\mathbf{X}), \quad \langle \tilde{\mathbf{F}} \rangle = \mathbf{0}$$

### Hyperelasticity: Neo-Hookean Model

HomiCSx provides a built-in compressible Neo-Hookean model. The strain energy density is $\Psi(\mathbf{F})$. The first Piola-Kirchhoff stress is also:

$$\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}} = \mu(\mathbf{F} - \mathbf{F}^{-T}) + \lambda (\ln J) \mathbf{F}^{-T}$$

The nominal tangent modulus for Newton-Raphson iterations is:

$$\mathbb{A} = \frac{\partial \mathbf{P}}{\partial \mathbf{F}} = \frac{\partial^2 \Psi}{\partial \mathbf{F} \partial \mathbf{F}}$$

A custom nonlinear material model can be provided by subclassing the abstract material class and implementing $\Psi(\mathbf{F})$, $\mathbf{P}(\mathbf{F})$, and $\mathbb{A}(\mathbf{F})$.

### Finite Strain Viscoelasticity

The generalized Maxwell model is available for rate-dependent materials. The total strain energy splits into equilibrium and $N$ non-equilibrium contributions:

$$\Psi = \Psi_{eq}(\mathbf{F}) + \sum_{i=1}^N \Psi_{neq}^{(i)}(\mathbf{F}_e^{(i)})$$

where $\mathbf{F}_e^{(i)} = \mathbf{F} (\mathbf{F}_v^{(i)})^{-1}$ is the elastic deformation in branch $i$, and $\mathbf{F}_v^{(i)}$ are internal viscous deformation variables. Their evolution follows:

$$\dot{\mathbf{F}}_v^{(i)} = \frac{1}{\tau_i} (\mathbf{F}_e^{(i)})^{-1} \, \text{dev}\!\left[ \frac{\partial \Psi_{neq}^{(i)}}{\partial \mathbf{F}_e^{(i)}} \right] \mathbf{F}$$

with relaxation times $\tau_i$. The equilibrium branch can use any hyperelastic model (Neo-Hookean or custom). State variables $\mathbf{F}_v^{(i)}$ are tracked at every quadrature point and updated during the solution.

### Solution Strategy

The nonlinear RVE boundary value problem is solved with adaptive load-stepping and Newton-Raphson iteration. At step $n$, iteration $k$:

$$\mathbf{K}_T^{(k)} \Delta \mathbf{u}^{(k)} = -\mathbf{R}^{(k)}$$

with tangent stiffness $\mathbf{K}_T^{(k)}$ assembled from the algorithmic consistent tangent. Convergence is tested by:

$$\|\mathbf{R}^{(k)}\|_2 \leq \epsilon_{rel} \|\mathbf{R}^{(0)}\|_2 + \epsilon_{abs}$$

Step size adapts based on iteration count: reduced on divergence or slow convergence, increased when convergence is rapid.

### Macroscopic Quantities

At each converged step, the macroscopic stress is computed by volume averaging:

$$\bar{\mathbf{P}} = \frac{1}{|\Omega_0|} \int_{\Omega_0} \mathbf{P} \, d\Omega$$

The effective tangent modulus is:

$$\bar{\mathbb{A}} = \frac{\partial \bar{\mathbf{P}}}{\partial \bar{\mathbf{F}}}$$

Recorded output includes: stress and tangent components, strain energy density $\bar{\Psi} = \langle \Psi \rangle$, Jacobian $J$, and apparent secant moduli.