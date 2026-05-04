from __future__ import annotations

from itertools import product
import math
import numpy as np


def _log(message: str, *, lvl: int = 0, verbosity: int = 0) -> None:
    if verbosity >= lvl:
        print(message)


def as_domain_size(dim: int, domain_size: float | tuple | list | np.ndarray) -> np.ndarray:
    """
    Standardizes the RVE domain dimensions into a NumPy array.

    This utility function converts various input formats (scalar, list, tuple) 
    into a consistent 1D NumPy array representing the side lengths of the 
    domain.

    Parameters
    ----------
    dim : int
        The expected spatial dimension (e.g., 2 or 3).
    domain_size : float | tuple | list | np.ndarray
        The input defining the domain size. 
        - If a scalar (float/int) is provided, it is interpreted as a uniform 
          length for all dimensions (cubic/square domain).
        - If a sequence is provided, it must contain exactly `dim` elements 
          representing the length along each axis.

    Returns
    -------
    np.ndarray
        A 1D array of shape `(dim,)` containing the domain lengths as floats.

    Raises
    ------
    ValueError
        - If the scalar value is non-positive.
        - If the provided sequence length does not match `dim`.
        - If any of the dimension values are non-positive.

    Examples
    --------
    >>> as_domain_size(dim=2, domain_size=1.0)
    array([1.0, 1.0])
    
    >>> as_domain_size(dim=3, domain_size=[1.0, 2.0, 1.5])
    array([1.0, 2.0, 1.5])
    """
    if np.isscalar(domain_size):
        L = float(domain_size)
        if L <= 0.0:
            raise ValueError("domain_size must be positive.")
        return np.full(dim, L, dtype=float)

    arr = np.asarray(domain_size, dtype=float).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"domain_size must have length {dim}.")
    if np.any(arr <= 0.0):
        raise ValueError("All domain dimensions must be positive.")
    return arr


def circle_area(radius: float) -> float:
    """Calculates circle area."""
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    return math.pi * radius**2


def sphere_volume(radius: float) -> float:
    """Calculates sphere volume."""
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    return (4.0 / 3.0) * math.pi * radius**3


def ellipse_area(semi_axes: np.ndarray) -> float:
    """Calculates ellipse area."""
    semi_axes = np.asarray(semi_axes, dtype=float).reshape(-1)
    if semi_axes.size != 2 or np.any(semi_axes <= 0.0):
        raise ValueError("ellipse semi-axes must have length 2 and be positive.")
    a, b = semi_axes
    return math.pi * a * b


def ellipsoid_volume(semi_axes: np.ndarray) -> float:
    """Calculates ellipsoid volume."""
    semi_axes = np.asarray(semi_axes, dtype=float).reshape(-1)
    if semi_axes.size != 3 or np.any(semi_axes <= 0.0):
        raise ValueError("ellipsoid semi-axes must have length 3 and be positive.")
    a, b, c = semi_axes
    return (4.0 / 3.0) * math.pi * a * b * c


def normalize_axis_ratios(axis_ratios) -> np.ndarray:
    """
    Standardizes axis ratio inputs into a consistent NumPy array.

    Parameters
    ----------
    axis_ratios : array_like
        Input values for axis ratios (can be list, tuple, or array).

    Returns
    -------
    np.ndarray
        A flattened 1D array of ratios.

    Raises
    ------
    ValueError
        If any of the ratios are non-positive.
    """
    ratios = np.asarray(axis_ratios, dtype=float).reshape(-1)
    if np.any(ratios <= 0.0):
        raise ValueError("All axis ratios must be positive.")
    return ratios


def monodisperse_circle_radius(volume_fraction: float, num_particles: int, domain_size: np.ndarray) -> float:
    """
    Calculates the radius of circles in a 2D monodisperse system.

    The radius :math:`R` is derived from the target area fraction.

    Parameters
    ----------
    volume_fraction : float
        Target area fraction [0 to 1].
    num_particles : int
        Total number of circles to be placed.
    domain_size : np.ndarray
        Dimensions of the 2D domain [L_x, L_y].

    Returns
    -------
    float
        The calculated uniform radius for all circles.
    """
    total_area = volume_fraction * float(np.prod(domain_size))
    return float(np.sqrt((total_area / num_particles) / math.pi))


def monodisperse_sphere_radius(volume_fraction: float, num_particles: int, domain_size: np.ndarray) -> float:
    """
    Calculates the radius of spheres in a 3D monodisperse system.

    The radius :math:`R` is derived from the target volume fraction.

    Parameters
    ----------
    volume_fraction : float
        Target volume fraction (0 to 1).
    num_particles : int
        Total number of spheres to be placed.
    domain_size : np.ndarray
        Dimensions of the 3D domain [L_x, L_y, L_z].

    Returns
    -------
    float
        The calculated uniform radius for all spheres.
    """
    total_volume = volume_fraction * float(np.prod(domain_size))
    particle_volume = total_volume / num_particles
    return float(((3.0 * particle_volume) / (4.0 * math.pi)) ** (1.0 / 3.0))


def monodisperse_ellipse_axes(volume_fraction: float, num_particles: int, domain_size: np.ndarray, axis_ratios) -> np.ndarray:
    """
    Calculates the semi-axes lengths for ellipses in a 2D monodisperse system.

    Given a target area fraction and relative axis ratios, this function computes 
    the absolute lengths of the semi-axes (a, b) such that each ellipse has 
    the same area and the system reaches the desired volume fraction.

    Parameters
    ----------
    volume_fraction : float
        Target area fraction [0 to 1].
    num_particles : int
        Number of ellipses to be generated.
    domain_size : np.ndarray
        Dimensions of the 2D domain [L_x, L_y].
    axis_ratios : array_like
        Relative ratios of the axes (e.g., [1.0, 2.0] for an ellipse twice 
        as long in one direction).

    Returns
    -------
    np.ndarray
        An array containing the absolute lengths of the two semi-axes.
    """
    ratios = normalize_axis_ratios(axis_ratios)
    if ratios.size != 2:
        raise ValueError("2D ellipse axis_ratios must have length 2.")
    total_area = volume_fraction * float(np.prod(domain_size))
    particle_area = total_area / num_particles
    scale = np.sqrt(particle_area / (math.pi * np.prod(ratios)))
    return scale * ratios


def monodisperse_ellipsoid_axes(volume_fraction: float, num_particles: int, domain_size: np.ndarray, axis_ratios) -> np.ndarray:
    """
    Calculates the semi-axes lengths for ellipsoids in a 3D monodisperse system.

    Computes the absolute lengths of the semi-axes (a, b, c) for 3D ellipsoids 
    based on the target volume fraction and prescribed relative axis ratios.

    Parameters
    ----------
    volume_fraction : float
        Target volume fraction [0 to 1].
    num_particles : int
        Number of ellipsoids to be generated.
    domain_size : np.ndarray
        Dimensions of the 3D domain [L_x, L_y, L_z].
    axis_ratios : array_like
        Relative ratios of the three axes (e.g., [1.0, 1.0, 2.0] for a prolate spheroid).

    Returns
    -------
    np.ndarray
        An array containing the absolute lengths of the three semi-axes.
    """
    ratios = normalize_axis_ratios(axis_ratios)
    if ratios.size != 3:
        raise ValueError("3D ellipsoid axis_ratios must have length 3.")
    total_volume = volume_fraction * float(np.prod(domain_size))
    particle_volume = total_volume / num_particles
    scale = (particle_volume / ((4.0 / 3.0) * math.pi * np.prod(ratios))) ** (1.0 / 3.0)
    return scale * ratios


def sample_random_center(rng: np.random.Generator, domain_size: np.ndarray) -> np.ndarray:
    """Generate random center within the domain bounds."""
    return rng.random(domain_size.size) * domain_size


def _minimum_image_delta(p1: np.ndarray, p2: np.ndarray, domain_size: np.ndarray) -> np.ndarray:
    """
    Computes the minimum-image displacement vector under periodic boundary conditions.

    Given two points in a periodic domain, this function returns the shortest
    displacement vector from ``p2`` to ``p1`` by applying the minimum-image
    convention independently in each coordinate direction.

    Parameters
    ----------
    p1 : np.ndarray
        Coordinates of the first point.
    p2 : np.ndarray
        Coordinates of the second point.
    domain_size : np.ndarray
        Periodic domain lengths along each coordinate direction. Each entry
        must be positive and correspond to the size of the domain in that
        dimension.

    Returns
    -------
    np.ndarray
        The wrapped displacement vector ``p1 - p2`` adjusted so that each
        component lies in the nearest periodic image of the domain.

    Notes
    -----
    For each coordinate direction, the returned displacement is mapped into
    the interval approximately ``[-L/2, L/2]``, where ``L`` is the domain
    length in that direction. This is the standard minimum-image convention
    used in periodic particle simulations, overlap checks, and molecular / 
    microstructure computations.

    Examples
    --------
    If the domain length is 10 in 1D, and the points are at 1 and 9, the
    direct difference is ``1 - 9 = -8``. Under periodic wrapping, the minimum
    image displacement becomes ``+2``.
    """
    delta = p1 - p2
    return delta - domain_size * np.round(delta / domain_size)


def _conservative_mesh_safety_radius(semi_axes: np.ndarray) -> float:
    semi_axes = np.asarray(semi_axes, dtype=float).reshape(-1)
    return float(np.max(semi_axes))


def _is_mesh_valid_primitive(center: np.ndarray, semi_axes: np.ndarray, clearance: float, domain_size: np.ndarray) -> bool:
    center = np.asarray(center, dtype=float).reshape(-1)
    semi_axes = np.asarray(semi_axes, dtype=float).reshape(-1)
    domain_size = np.asarray(domain_size, dtype=float).reshape(-1)

    if center.size != domain_size.size:
        raise ValueError("center and domain_size must have same dimension.")
    if semi_axes.size != domain_size.size:
        raise ValueError("semi_axes and domain_size must have same dimension.")

    if np.any(center <= 0.0) or np.any(center >= domain_size):
        return False

    safety_r = _conservative_mesh_safety_radius(semi_axes)

    # Faces
    for i in range(domain_size.size):
        d1 = center[i]
        d2 = domain_size[i] - center[i]
        if abs(d1 - safety_r) < clearance:
            return False
        if abs(d2 - safety_r) < clearance:
            return False

    # Corners
    for signs in product([0.0, 1.0], repeat=domain_size.size):
        corner = np.array([signs[i] * domain_size[i] for i in range(domain_size.size)], dtype=float)
        d = float(np.linalg.norm(center - corner))
        if abs(d - safety_r) < clearance:
            return False

    return True


def _primitives_overlap_periodic(
    center_1: np.ndarray,
    semi_axes_1: np.ndarray,
    shape_1: str,
    center_2: np.ndarray,
    semi_axes_2: np.ndarray,
    shape_2: str,
    domain_size: np.ndarray,
    clearance: float = 0.0,
) -> bool:
    """
    Checks for overlap between two geometric primitives in a periodic domain.

    This function determines if two shapes intersect, accounting for the 
    'Minimum Image Convention' to handle periodicity. It supports both 
    perfectly round shapes (circles/spheres) and axis-aligned ellipses/ellipsoids.

    Parameters
    ----------
    center_1, center_2 : np.ndarray
        Coordinates of the centers of the first and second primitives.
    semi_axes_1, semi_axes_2 : np.ndarray
        Lengths of the semi-axes (or radii) for each primitive.
    shape_1, shape_2 : str
        Geometric types: "circle", "sphere", "ellipse", or "ellipsoid".
    domain_size : np.ndarray
        Dimensions of the periodic unit cell [L_x, L_y, (L_z)].
    clearance : float, default 0.0
        Minimum required separation distance between the surfaces of the 
        two primitives. Positive values enforce a safety gap.

    Returns
    -------
    bool
        True if the primitives overlap (or violate the clearance), False otherwise.

    Notes
    -----
    - For round shapes, the Euclidean distance between centers is compared 
      to the sum of radii plus clearance.
    - For axis-aligned ellipses/ellipsoids, an approximate overlap check is 
      performed by scaling the distance vector by the sum of semi-axes.
    - This implementation assumes that non-spherical shapes are not rotated
      relative to the domain axes.

    See Also
    --------
    minimum_image_delta : Computes the shortest distance vector under periodic BCs.
    """
    center_1 = np.asarray(center_1, dtype=float).reshape(-1)
    center_2 = np.asarray(center_2, dtype=float).reshape(-1)
    semi_axes_1 = np.asarray(semi_axes_1, dtype=float).reshape(-1)
    semi_axes_2 = np.asarray(semi_axes_2, dtype=float).reshape(-1)
    domain_size = np.asarray(domain_size, dtype=float).reshape(-1)

    delta = _minimum_image_delta(center_1, center_2, domain_size)

    both_round = (
        (shape_1 == "circle" and shape_2 == "circle" and delta.size == 2)
        or
        (shape_1 == "sphere" and shape_2 == "sphere" and delta.size == 3)
    )

    if both_round:
        r1 = float(semi_axes_1[0])
        r2 = float(semi_axes_2[0])
        return float(np.linalg.norm(delta)) < (r1 + r2 + clearance)

    summed_axes = semi_axes_1 + semi_axes_2 + clearance
    scaled = delta / summed_axes
    return float(np.sum(scaled**2)) < 1.0


def _overlaps_existing_periodic(
    candidate_center: np.ndarray,
    candidate_axes: np.ndarray,
    candidate_shape: str,
    existing_centers: list[np.ndarray],
    existing_axes: list[np.ndarray],
    existing_shapes: list[str],
    domain_size: np.ndarray,
    clearance: float,
    allow_overlap: bool = False
) -> bool:
    """
    Checks if a candidate particle overlaps with any previously placed particles.

    This function iterates through the collection of existing particles and 
    performs a pairwise overlap check against the candidate, accounting for 
    periodic boundary conditions. It is a core component of the Random 
    Sequential Adsorption (RSA) algorithm.

    Parameters
    ----------
    candidate_center : np.ndarray
        The proposed center coordinates for the new particle.
    candidate_axes : np.ndarray
        The semi-axes lengths (radii) of the candidate particle.
    candidate_shape : str
        The geometry type of the candidate (e.g., "circle", "ellipse").
    existing_centers : list[np.ndarray]
        A list containing the centers of all particles already accepted.
    existing_axes : list[np.ndarray]
        A list containing the semi-axes of all particles already accepted.
    existing_shapes : list[str]
        A list of geometry type strings for each existing particle.
    domain_size : np.ndarray
        The dimensions of the periodic RVE [L_x, L_y, (L_z)].
    clearance : float
        The minimum required buffer distance between any two particles.

    Returns
    -------
    bool
        True if the candidate overlaps with at least one existing particle 
        (within the clearance distance), False if the position is 'safe'.

    Notes
    -----
    The function uses a 'First-Hit' logic: it returns True immediately upon 
    detecting the first overlap, which optimizes performance during the 
    stochastic placement process.
    """
    if not allow_overlap:
        for center_i, axes_i, shape_i in zip(existing_centers, existing_axes, existing_shapes):
            if _primitives_overlap_periodic(
                candidate_center,
                candidate_axes,
                candidate_shape,
                center_i,
                axes_i,
                shape_i,
                domain_size,
                clearance=clearance,
            ):
                return True
        return False
    else:
        return False


def generate_periodic_image_centers(center: np.ndarray, extents: np.ndarray, domain_size: np.ndarray) -> list[np.ndarray]:
    """
    Generates necessary periodic image centers for a particle near domain boundaries.

    This function identifies if a particle (defined by its center and axis-aligned 
    bounding box extents) crosses any boundary of the periodic domain. If it 
    does, it calculates the centers of the 'ghost' images in the neighboring 
    periodic cells.

    Parameters
    ----------
    center : np.ndarray
        Coordinates of the particle's center.
    extents : np.ndarray
        The maximum reach of the particle from its center along each axis 
        (e.g., radii or semi-axes). Used to check boundary intersection.
    domain_size : np.ndarray
        Dimensions of the periodic unit cell [L_x, L_y, (L_z)].

    Returns
    -------
    list[np.ndarray]
        A list of center coordinates for the required periodic images. 
        Returns an empty list if the particle is fully contained within 
        the domain (no boundary crossing).

    Notes
    -----
    - The function uses a selective approach: it only generates images for 
      the specific directions where the particle exceeds the domain bounds.
    - If a particle sits in a corner, it will correctly generate images for 
      all adjacent cells (including diagonal ones) using a Cartesian product 
      of valid shifts.
    """
    center = np.asarray(center, dtype=float).reshape(-1)
    extents = np.asarray(extents, dtype=float).reshape(-1)
    domain_size = np.asarray(domain_size, dtype=float).reshape(-1)

    shift_options = []
    for i in range(domain_size.size):
        options = [0.0]
        if center[i] - extents[i] < 0.0:
            options.append(domain_size[i])
        if center[i] + extents[i] > domain_size[i]:
            options.append(-domain_size[i])
        shift_options.append(options)

    images = []
    for shift_tuple in product(*shift_options):
        shift = np.array(shift_tuple, dtype=float)
        if np.allclose(shift, 0.0):
            continue
        images.append(center + shift)
    return images


__all__ = [
    # helpers
    "as_domain_size",
    "circle_area",
    "sphere_volume",
    "ellipse_area",
    "ellipsoid_volume",
    "normalize_axis_ratios",
    "monodisperse_circle_radius",
    "monodisperse_sphere_radius",
    "monodisperse_ellipse_axes",
    "monodisperse_ellipsoid_axes",
    "sample_random_center",
    "generate_periodic_image_centers",
]





