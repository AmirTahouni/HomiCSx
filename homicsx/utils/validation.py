from __future__ import annotations

from itertools import combinations

import numpy as np

from homicsx.core.geometry import Inclusion, RVEGeometry
from homicsx.geometry.helpers import _minimum_image_delta


def original_inclusions(geometry: RVEGeometry) -> list[Inclusion]:
    return [inc for inc in geometry.inclusions if not inc.is_periodic_image()]


def periodic_image_inclusions(geometry: RVEGeometry) -> list[Inclusion]:
    return [inc for inc in geometry.inclusions if inc.is_periodic_image()]


def _axes_for_inclusion(inclusion: Inclusion) -> np.ndarray:
    if inclusion.shape in ("circle", "sphere"):
        return np.full(inclusion.dim, inclusion.radius, dtype=float)
    return np.asarray(inclusion.radii, dtype=float)


def primitive_overlap(inc1: Inclusion, inc2: Inclusion, domain_size: np.ndarray, clearance: float = 0.0) -> bool:
    c1 = np.asarray(inc1.center, dtype=float)
    c2 = np.asarray(inc2.center, dtype=float)
    a1 = _axes_for_inclusion(inc1)
    a2 = _axes_for_inclusion(inc2)

    delta = _minimum_image_delta(c1, c2, np.asarray(domain_size, dtype=float))

    both_round = (
        (inc1.shape == "circle" and inc2.shape == "circle" and inc1.dim == 2)
        or
        (inc1.shape == "sphere" and inc2.shape == "sphere" and inc1.dim == 3)
    )

    if both_round:
        return float(np.linalg.norm(delta)) < (inc1.radius + inc2.radius + clearance)

    scaled = delta / (a1 + a2 + clearance)
    return float(np.sum(scaled**2)) < 1.0


def geometry_has_periodic_overlaps(geometry: RVEGeometry, clearance: float = 0.0, originals_only: bool = True) -> bool:
    inclusions = original_inclusions(geometry) if originals_only else geometry.inclusions
    for inc1, inc2 in combinations(inclusions, 2):
        if primitive_overlap(inc1, inc2, geometry.domain_size, clearance=clearance):
            return True
    return False


def realized_volume_fraction_from_originals(geometry: RVEGeometry) -> float:
    total = sum(inc.volume_measure for inc in original_inclusions(geometry))
    return total / geometry.domain_measure


def image_map_is_consistent(geometry: RVEGeometry) -> bool:
    image_map = geometry.periodicity.image_map
    inclusions = geometry.inclusions

    for source_idx, image_indices in image_map.items():
        if source_idx < 0 or source_idx >= len(inclusions):
            return False

        source = inclusions[source_idx]
        if source.is_periodic_image():
            return False

        for image_idx in image_indices:
            if image_idx < 0 or image_idx >= len(inclusions):
                return False

            image = inclusions[image_idx]
            if not image.is_periodic_image():
                return False

            if image.periodic_source_id != source_idx:
                return False

            if image.shape != source.shape:
                return False

            if not np.allclose(image.radii, source.radii):
                return False

    return True


__all__ = [
    
]



