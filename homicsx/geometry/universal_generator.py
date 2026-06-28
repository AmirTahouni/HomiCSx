from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from homicsx.core.geometry import (
    RVEGeometry, 
    GeometryInput,
)

from .generators_3d import _generate_mono_3d, _generate_poly_3d
from .generators_2d import _generate_mono_2d, _generate_poly_2d


def particulate_geometry_generator(
        input_data: GeometryInput
) -> RVEGeometry:
    """
    Generate a particulate RVE geometry from configuration parameters.

    This is the main entry point for creating inclusion-based Representative 
    Volume Element (RVE) geometries. It dispatches to the appropriate 
    specialized generator based on spatial dimension (2D/3D) and particle 
    size distribution (mono-disperse/poly-disperse).

    Parameters
    ----------
    input_data : GeometryInput
        Configuration object specifying all geometry parameters including 
        dimension, dispersion type, shape, volume fraction, and algorithmic 
        settings. See :class:`GeometryInput` for detailed parameter 
        documentation.

    Returns
    -------
    RVEGeometry
        An object containing the generated microstructure:
        
        - ``centers`` : List of inclusion center coordinates
        - ``axes`` : List of semi-axis lengths
        - ``shapes`` : List of shape identifiers
        - ``domain_size`` : Domain dimensions
        - ``metadata`` : Generation parameters and statistics

    Notes
    -----
    The generator uses a Random Sequential Adsorption (RSA) approach:
    
    1. Random positions are sampled within the domain
    2. Each candidate is checked for:
       - Domain boundary compliance (periodicity)
       - Clearance from existing inclusions
       - Volume fraction limits
    3. Process repeats until target volume fraction is reached or limits 
       are exceeded

    For poly-disperse distributions, particle sizes are drawn uniformly 
    from [``min_radius``, ``max_radius``] or scaled from 
    [``min_scale``, ``max_scale``] × ``axis_ratios``.

    For overlapping inclusions (``allow_overlap=True``), the effective 
    volume fraction is estimated using the Poisson overlap correction:
    ``VF_actual = 1 - exp(-VF_nominal)``.

    See Also
    --------
    GeometryInput : Configuration container for this function.
    RVEGeometry : Output container for generated geometries.

    Examples
    --------
    >>> from homicsx.geometry import GeometryInput, particulate_geometry_generator
    >>> 
    >>> # 2D poly-disperse circles at 30% volume fraction
    >>> config = GeometryInput(
    ...     dim=2, dispersion="poly", shape="circle",
    ...     volume_fraction=0.3, volume_fraction_tolerance=0.01,
    ...     min_radius=0.05, max_radius=0.15,
    ...     clearance=0.01, domain_size=(1.0, 1.0),
    ...     seed=42,
    ... )
    >>> geometry = particulate_geometry_generator(config)
    >>> print(f"Generated {len(geometry.centers)} inclusions")
    >>> 
    >>> # 3D mono-disperse spheres
    >>> config = GeometryInput(
    ...     dim=3, dispersion="mono", shape="sphere",
    ...     volume_fraction=0.2, num_particles=50,
    ...     clearance=0.02, seed=123,
    ... )
    >>> geometry = particulate_geometry_generator(config)
    >>> 
    >>> # Open-cell foam (overlapping voids)
    >>> config = GeometryInput(
    ...     dim=3, dispersion="poly", shape="sphere",
    ...     volume_fraction=0.6, volume_fraction_tolerance=0.02,
    ...     min_radius=0.05, max_radius=0.20,
    ...     clearance=0.0, allow_overlap=True,
    ... )
    >>> geometry = particulate_geometry_generator(config)
    """
    if input_data.dim == 3:
        if input_data.dispersion == "mono":
            return _generate_mono_3d(input_data)
        elif input_data.dispersion == "poly":
            return _generate_poly_3d(input_data)
            
    elif input_data.dim == 2:
        if input_data.dispersion == "mono":
            return _generate_mono_2d(input_data)
        elif input_data.dispersion == "poly":
            return _generate_poly_2d(input_data)
         

__all__ = [
    # universal_generator
    "particulate_geometry_generator",
]

