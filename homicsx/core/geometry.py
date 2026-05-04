from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import math
import numpy as np


SupportedShape = Literal["circle", "sphere", "ellipse", "ellipsoid"]


@dataclass
class Inclusion:
    """
    Represents a geometric inclusion (particle) within the RVE.

    This class defines the geometry, phase properties, and interphase 
    characteristics of a single inclusion. It supports both 2D and 3D 
    shapes, including circular, spherical, and ellipsoidal geometries, 
    with built-in support for periodic boundary conditions and interphase layers.

    Attributes
    ----------
    center : np.ndarray
        The coordinates of the inclusion center (2D or 3D).
    phase_id : int
        Material phase identifier for the inclusion core.
    shape : SupportedShape
        Geometry type. Must be one of: "circle", "sphere", "ellipse", "ellipsoid".
    radii : np.ndarray
        Geometric dimensions. Represents the radius for circles/spheres, 
        or semi-axes for ellipses/ellipsoids.
    interphase_thickness_ratio : float, optional
        Ratio of the interphase thickness relative to the total radius/semi-axes.
        Must be in the range [0.0, 1.0). Default is 0.0 (no interphase).
    interphase_phase_id : int | None, optional
        Material phase identifier for the interphase layer. If not provided 
        and thickness ratio > 0, defaults to `phase_id + 100`.
    periodic_source_id : int | None, optional
        The ID of the original inclusion if this instance is a periodic image 
        across the RVE boundaries. Default is None.
    metadata : dict[str, Any], optional
        Additional user-defined data associated with the inclusion.

    Raises
    ------
    ValueError
        If dimensions, phase IDs, or radii are inconsistent with the specified shape.
    """
    center: np.ndarray
    phase_id: int
    shape: SupportedShape
    radii: np.ndarray
    interphase_thickness_ratio: float = 0.0
    interphase_phase_id: int | None = None
    periodic_source_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(-1)
        self.radii = np.asarray(self.radii, dtype=float).reshape(-1)

        # radii checks
        if self.center.size not in (2, 3):
            raise ValueError("center must have dimension 2 or 3.")
        if self.phase_id < 0:
            raise ValueError("phase_id must be non-negative.")
        if np.any(self.radii <= 0.0):
            raise ValueError("All radii must be positive.")
        
        # shape checks
        if self.shape == "circle":
            if self.center.size != 2 or self.radii.size != 1:
                raise ValueError("circle requires 2D center and one radius.")
        elif self.shape == "sphere":
            if self.center.size != 3 or self.radii.size != 1:
                raise ValueError("sphere requires 3D center and one radius.")
        elif self.shape == "ellipse":
            if self.center.size != 2 or self.radii.size != 2:
                raise ValueError("ellipse requires 2D center and two semi-axes.")
        elif self.shape == "ellipsoid":
            if self.center.size != 3 or self.radii.size != 3:
                raise ValueError("ellipsoid requires 3D center and three semi-axes.")
        else:
            raise ValueError(f"Unsupported shape: {self.shape}")
        
        # interphase checks
        if not (0.0 <= self.interphase_thickness_ratio < 1.0):
            raise ValueError("interphase_thickness_ratio must be between 0 and 1.")

        if self.interphase_thickness_ratio > 0.0 and self.interphase_phase_id is None:
            self.interphase_phase_id = self.phase_id + 100

    @property
    def dim(self) -> int:
        """The spatial dimension (2 or 3) of the inclusion."""
        return int(self.center.size)

    @property
    def radius(self) -> float:
        """If shape is circle or sphere, returns radius."""
        if self.shape not in ("circle", "sphere"):
            raise AttributeError("radius only exists for circle/sphere.")
        return float(self.radii[0])

    @property
    def has_interphase(self) -> bool:
        """True if the inclusion has an active interphase layer."""
        return self.interphase_thickness_ratio > 0.0

    @property
    def inner_radii(self) -> np.ndarray:
        """Inner radii of the inclusion. Same as outer radii if there is no interphase."""
        return self.radii * (1.0 - self.interphase_thickness_ratio)

    @property
    def outer_radii(self) -> np.ndarray:
        """Outer radii of the inclusion."""
        return self.radii

    @property
    def volume_measure(self) -> float:
        """Total volume of the RVE."""
        return self._calculate_volume(self.outer_radii)

    @property
    def core_volume(self) -> float:
        """Volume of the core of the inclusion, excluding the interphase layer from the total volume."""
        return self._calculate_volume(self.inner_radii)

    @property
    def interphase_volume(self) -> float:
        """Volume of the interphase, excluding the core from the total volume."""
        return self.volume_measure - self.core_volume
    
    @property
    def total_volume(self) -> float:
        """Total volume (or area in 2D) of the inclusion including its interphase."""
        return self.core_volume + self.interphase_volume

    def _calculate_volume(self, r_array: np.ndarray) -> float:
        if self.shape == "circle":
            return math.pi * r_array[0]**2
        if self.shape == "sphere":
            return (4.0 / 3.0) * math.pi * r_array[0]**3
        if self.shape == "ellipse":
            return math.pi * r_array[0] * r_array[1]
        if self.shape == "ellipsoid":
            return (4.0 / 3.0) * math.pi * r_array[0] * r_array[1] * r_array[2]
        raise NotImplementedError

    def is_periodic_image(self) -> bool:
        """Returns true if inclusion is a periodic image."""
        return self.periodic_source_id is not None


@dataclass
class _PeriodicityInfo:
    """
    Contains the periodic info about the inclusions in an RVE.

    Describes how inclusions that intersect boundaries of the RVE should be handled via periodic images. 
    Preventes re-computation of periodic images in the geometry and mesh construction modules and 
    allows consistent treatment of periodic images.

    Attributes
    ----------
    enabled: bool
        Whether periodicity should be applied to intersecting inclusions. Defaults to True.
    image_map: dict[int, list[int]]
        Maps the list of image inclusion IDs to the ID of the original inclusion.
    metadata: dict
        Optional dictionary for additional properties.
    
    Note
    ----
    - Periodic images are explicitly stored in 'ParticulateRVEGeometry.inclusions'.
    - The meshing module does not re-construct the periodic images using this class. 
    It assumes that the images are already generated and present.
    - The image_map attribute is primarily used for validation and debugging purposes.
    """
    enabled: bool = True
    image_map: dict[int, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryInput:
    """
    Configuration container for particulate RVE geometry generation.

    This class holds all physical and algorithmic parameters required to 
    generate a Representative Volume Element (RVE) for particulate composites.
    Supports both mono-disperse (fixed number of equal-sized particles) and 
    poly-disperse (random sizes within a range) distributions in 2D and 3D.

    Attributes
    ----------
    dim : int
        Spatial dimension of the RVE. Must be 2 or 3.
    dispersion : str
        Type of particle size distribution.
        - ``"mono"`` : Uniform size. Requires ``num_particles``.
        - ``"poly"`` : Random size within [``min_radius``, ``max_radius``] 
          or [``min_scale``, ``max_scale``]. Requires ``volume_fraction_tolerance``.
    shape : str
        Geometry of the inclusions.
        - 3D: ``"sphere"``, ``"ellipsoid"``
        - 2D: ``"circle"``, ``"ellipse"``
    volume_fraction : float
        Target volume fraction of inclusions. Must be in [0, 1].
    clearance : float
        Minimum distance between surfaces of any two inclusions. Must be positive.
    domain_size : float, tuple, list, or np.ndarray, default 1.0
        RVE domain dimensions. A single float creates a cubic/square domain.
    volume_fraction_tolerance : float, optional
        Allowable deviation from ``volume_fraction``. Required for ``"poly"``,
        must be ``None`` for ``"mono"``.
    num_particles : int, optional
        Fixed number of particles. Required for ``"mono"``, must be ``None`` 
        for ``"poly"``.
    axis_ratios : tuple of float, optional
        Semi-axis ratios for ellipsoidal/elliptical shapes (e.g., ``(1.0, 2.0)`` 
        for 2D, ``(1.0, 2.0, 3.0)`` for 3D). Must have length equal to ``dim``.
    min_radius, max_radius : float, optional
        Radius range for circles/spheres in poly-disperse mode. Required for 
        ``"poly"`` with ``"circle"``/``"sphere"`` shapes, must be ``None`` 
        for ``"mono"``.
    min_scale, max_scale : float, optional
        Scale factor range for ellipses/ellipsoids in poly-disperse mode.
        The actual semi-axes are ``scale * axis_ratios``. Required for ``"poly"`` 
        with ``"ellipse"``/``"ellipsoid"`` shapes, must be ``None`` for ``"mono"``.
    max_restarts : int, default 100
        Maximum number of global restart attempts if volume fraction cannot 
        be satisfied.
    attempts_per_restart : int, default 100000
        Number of random placement trials per restart before giving up.
    seed : int, optional
        Seed for reproducible random number generation.
    interphase_thickness_ratio : float, default 0.0
        Relative thickness of interphase layer around each inclusion.
        If 0.0, no interphase is generated. The interphase thickness is 
        ``ratio * min_semi_axis``.
    interphase_phase_id : int or None
        Phase ID assigned to interphase regions. Set automatically during 
        geometry building. Users should typically not set this.
    allow_overlap : bool, default False
        If ``True``, inclusions may overlap. Useful for open-cell foam 
        generation. Volume fraction is estimated using the Poisson overlap 
        correction: ``VF_actual = 1 - exp(-VF_nominal)``.

    Notes
    -----
    The class enforces strict mutual exclusivity between ``"mono"`` and 
    ``"poly"`` modes:

    - **Mono-disperse**: Fixed number of equal-sized particles.
    - **Poly-disperse**: Particles with sizes drawn from [min, max] range 
      are placed until the target volume fraction is reached.

    The underlying algorithm uses Random Sequential Adsorption (RSA) with 
    periodic boundary conditions.

    Raises
    ------
    ValueError
        If invalid dimensions, shapes, or conflicting dispersion parameters 
        are provided. See ``__post_init__`` for detailed validation rules.

    Examples
    --------
    >>> # 2D poly-disperse circles
    >>> geo_input = GeometryInput(
    ...     dim=2, dispersion="poly", shape="circle",
    ...     volume_fraction=0.3, volume_fraction_tolerance=0.01,
    ...     min_radius=0.05, max_radius=0.15,
    ...     clearance=0.01, domain_size=(1.0, 1.0),
    ...     seed=42,
    ... )
    
    >>> # 3D mono-disperse spheres
    >>> geo_input = GeometryInput(
    ...     dim=3, dispersion="mono", shape="sphere",
    ...     volume_fraction=0.2, num_particles=50,
    ...     clearance=0.02, domain_size=1.0,
    ... )
    
    >>> # 2D open-cell foam (overlapping circles)
    >>> geo_input = GeometryInput(
    ...     dim=2, dispersion="poly", shape="circle",
    ...     volume_fraction=0.4, volume_fraction_tolerance=0.02,
    ...     min_radius=0.03, max_radius=0.10,
    ...     clearance=0.0, allow_overlap=True,
    ... )
    """
    dim: int
    dispersion: str
    shape: str
    volume_fraction: float
    clearance: float
    domain_size: float | tuple[float, float, float] | list[float] | np.ndarray = 1
    volume_fraction_tolerance: float | None = None
    num_particles: int | None = None
    axis_ratios: tuple[int] | None = None
    min_radius: float | None = None
    max_radius: float | None = None
    min_scale: float | None = None
    max_scale: float | None = None 
    max_restarts: int = 100
    attempts_per_restart: int = 100000
    seed: int | None = None
    interphase_thickness_ratio: float = 0.0
    allow_overlap: bool = False

    def __post_init__(self) -> None:
        # dim guard
        self.interphase_phase_id = None

        if self.dim not in [2, 3]:
            raise ValueError(f'dim must be either 2 or 3. recieved {self.dim}.')
        
        # dispersity guard
        if self.dispersion not in ['mono', 'poly']:
            raise ValueError(f"wrong dispersion value. must be either 'mono' or 'poly'. recieved {self.dispersion}")

        # shape guard
        if self.dim==3:
            if self.shape not in ['sphere', 'ellipsoid']:
                raise ValueError(f"wrong 3D shape. must be either 'sphere' or 'ellipsoid'. recieved {self.shape}.")
        elif self.dim==2:
            if self.shape not in ['circle', 'ellipse']:
                raise ValueError(f"wrong 2D shape. must be either 'circle' or 'ellipse'. recieved {self.shape}.")
        
        # volume fraction guard
        if not (0 <= self.volume_fraction <= 1):
            raise ValueError("volume fraction must be in [0, 1].")
        
        # clearance guard
        if self.clearance<=0:
            raise ValueError("clearance must be non-zero positive.")
        
        # input guards
        if self.dim == 3:
            if self.dispersion == "mono":
                if self.num_particles is None:
                    raise ValueError('num_particles must not be None for monodisperse geometries.')
                if self.volume_fraction_tolerance is not None:
                    raise ValueError('volume_fraction_tolerance must be None for monodisperse geometries.')
                if self.min_radius is not None or self.max_radius is not None:
                    raise ValueError('min and max radius must be None for monodisperse geometries.')
                if self.min_scale is not None or self.max_scale is not None:
                    raise ValueError('min and max scales must be None for monodisperse geometries.')
                
            elif self.dispersion == "poly":
                if self.num_particles is not None:
                    raise ValueError('num_particles must be None for polydisperse geometries.')
                if self.volume_fraction_tolerance is None:
                    raise ValueError('volume_fraction_tolerance must not be None for polydisperse geometries.')
                if self.min_radius is None or self.max_radius is None:
                    raise ValueError('min and max radius must not be None for polydisperse geometries.')
                if self.min_scale is None or self.max_scale is None:
                    raise ValueError('min and max scales must not be None for polydisperse geometries.') 

        elif self.dim == 2:
            if self.dispersion == "mono":
                if self.num_particles is None:
                    raise ValueError('num_particles must not be None for monodisperse geometries.')
                if self.volume_fraction_tolerance is not None:
                    raise ValueError('volume_fraction_tolorance must be None for monodisperse geometries.')
                if self.min_radius is not None or self.max_radius is not None:
                    raise ValueError('min and max radius must be None for monodisperse geometries.')
                if self.min_scale is not None or self.max_scale is not None:
                    raise ValueError('min and max scales must be None for monodisperse geometries.')            
            
            elif self.dispersion == "poly":
                if self.num_particles is not None:
                    raise ValueError('num_particles must be None for polydisperse geometries.')
                if self.volume_fraction_tolerance is None:
                    raise ValueError('volume_fraction_tolorance must not be None for polydisperse geometries.')
                if self.min_radius is None or self.max_radius is None:
                    raise ValueError('min and max radius must not be None for polydisperse geometries.')
                if self.min_scale is None or self.max_scale is None:
                    raise ValueError('min and max scales must not be None for polydisperse geometries.')        
        

@dataclass
class RVEGeometry:
    """
    Represents the micro-structure of a representative volume element (RVE) collected from a macroscopic composite body.
    An object of this class contains the domain definition and all of the inclusion data (including the periodic images) 
    inside the domain required to build the mesh.

    Attributes
    ----------
    dim: int
        States the spacial dimension of the RVE (2 for 2D and 3 for 3D)
    domain_size: np.ndarray
        States the size of the domain in its dimensions:
        - Of size 2 and including Lx and Ly for 2D domains.
        - Of size 3 and including Lx, Ly, and Lz for 3D domains.
    inclusions: list[Inclusion]
        Contains all of the inclusion data, including the centers, radii, phase IDs, shapes, and source IDs if periodic.
    phase_ids: tuple[int, ...]
        Tuple of material phase identifiers present in the RVE. 
        The first entry ID is treated as the matrix phase. Defaults to (0, 1) as a 2 phase composite.
    target_volume_fraction: float
        The input target volume fraction intended for the generated RVE.
    realized_volume_fraction: float
        The output realized volume fraction of the RVE. Intended to equal to 'target_volume_fraction'.
    seed: int
        The random seed used in the RVE generation process.
    metadata: dict
        Optional dictionary for additional properties.

    Note
    ----
    - An object of this class explicitly contains the periodic images inside 'inclusions', if available.
    - Domain is assumed to start at origin.
    - No geometric validation (eg. overlap checks) is enforced here.

    See also
    --------
    Inclusion
    """
    dim: Literal[2, 3]
    domain_size: np.ndarray
    inclusions: list[Inclusion] = field(default_factory=list)
    phase_ids: tuple[int, ...] = (0, 1)
    # periodicity: PeriodicityInfo = field(default_factory=PeriodicityInfo)
    target_volume_fraction: float | None = None
    realized_volume_fraction: float | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.domain_size = np.asarray(self.domain_size, dtype=float).reshape(-1)

        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3.")
        if self.domain_size.size != self.dim:
            raise ValueError("domain_size size must match dim.")
        if np.any(self.domain_size <= 0.0):
            raise ValueError("All domain dimensions must be positive.")

        for inc in self.inclusions:
            if inc.dim != self.dim:
                raise ValueError("Inclusion dimension mismatch.")

        if self.target_volume_fraction is not None:
            self.target_volume_fraction = float(self.target_volume_fraction)
            if not (0.0 <= self.target_volume_fraction <= 1.0):
                raise ValueError("target_volume_fraction must be in [0, 1].")

        if self.realized_volume_fraction is not None:
            self.realized_volume_fraction = float(self.realized_volume_fraction)
            if not (0.0 <= self.realized_volume_fraction <= 1.0):
                raise ValueError("realized_volume_fraction must be in [0, 1].")
            
        self._update_phase_ids()

    @property
    def domain_measure(self) -> float:
        """The total area (2D) or volume (3D) of the RVE domain."""
        return float(np.prod(self.domain_size))
    
    def _update_phase_ids(self) -> None:
        present_phases = {self.phase_ids[0]}
        for inc in self.inclusions:
            present_phases.add(inc.phase_id)
            if inc.has_interphase:
                present_phases.add(inc.interphase_phase_id)
        self.phase_ids = tuple(sorted(list(present_phases)))

    @property
    def core_volume_fraction(self) -> float:
        """The volume fraction of only the inclusion cores (excluding inter-phases)."""
        total_core_vol = sum(inc.core_volume for inc in self.inclusions if not inc.is_periodic_image())
        return total_core_vol / self.domain_measure
    
    @property
    def interphase_volume_fraction(self) -> float:
        """The volume fraction occupied by the interphase layers."""
        total_inter_vol = sum(inc.interphase_volume for inc in self.inclusions if not inc.is_periodic_image())
        return total_inter_vol / self.domain_measure
    
    @property
    def total_inclusion_volume_fraction(self) -> float:
        """The sum of core and interphase volume fractions."""
        return self.core_volume_fraction + self.interphase_volume_fraction

    def add_inclusion(self, inclusion: Inclusion) -> None:
        """Add an inclusion of type `Inclusion`."""
        if inclusion.dim != self.dim:
            raise ValueError("Inclusion dimension mismatch.")
        self.inclusions.append(inclusion)


__all__ = [
    # geometry
    "Inclusion",
    "GeometryInput",
    "RVEGeometry",
]





        