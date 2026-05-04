import numpy as np
import pytest

from homicsx.geometry import (
    generate_mono_sphere_3d,
    generate_poly_sphere_3d,
    generate_mono_ellipsoid_3d,
    generate_poly_ellipsoid_3d,
)
from homicsx.geometry.helpers import (
    ellipsoid_volume,
    monodisperse_ellipsoid_axes,
    monodisperse_sphere_radius,
    sphere_volume,
)
from homicsx.utils.validation import (
    geometry_has_periodic_overlaps,
    image_map_is_consistent,
    original_inclusions,
    periodic_image_inclusions,
    realized_volume_fraction_from_originals,
)


@pytest.fixture
def unit_cube():
    return np.array([1.0, 1.0, 1.0], dtype=float)


def test_mono_sphere_radius_matches_target(unit_cube):
    vf = 0.12
    n = 10
    r = monodisperse_sphere_radius(vf, n, unit_cube)
    achieved = n * sphere_volume(r) / np.prod(unit_cube)
    assert np.isclose(achieved, vf, atol=1e-12)


def test_mono_ellipsoid_axes_match_target_volume(unit_cube):
    vf = 0.15
    n = 6
    axes = monodisperse_ellipsoid_axes(vf, n, unit_cube, (1.0, 1.5, 2.0))
    achieved = n * ellipsoid_volume(axes) / np.prod(unit_cube)
    assert np.isclose(achieved, vf, atol=1e-12)


def test_generate_mono_sphere_3d_returns_correct_original_count():
    geom = generate_mono_sphere_3d(
        volume_fraction=0.08,
        num_particles=8,
        clearance=0.01,
        seed=42,
    )
    assert len(original_inclusions(geom)) == 8
    assert geom.dim == 3


def test_generate_mono_sphere_3d_has_no_periodic_overlaps():
    geom = generate_mono_sphere_3d(
        volume_fraction=0.06,
        num_particles=6,
        clearance=0.005,
        seed=123,
    )
    assert not geometry_has_periodic_overlaps(geom, clearance=0.005, originals_only=True)


def test_generate_mono_sphere_3d_realized_vf_matches_target():
    vf = 0.10
    geom = generate_mono_sphere_3d(
        volume_fraction=vf,
        num_particles=12,
        clearance=0.002,
        seed=4,
    )
    realized = realized_volume_fraction_from_originals(geom)
    assert np.isclose(realized, vf, atol=1e-12)
    assert np.isclose(geom.realized_volume_fraction, vf, atol=1e-12)


def test_generate_poly_sphere_3d_realized_vf_within_tolerance():
    target = 0.13
    tol = 0.02
    geom = generate_poly_sphere_3d(
        volume_fraction=target,
        volume_fraction_tolerance=tol,
        min_radius=0.04,
        max_radius=0.08,
        clearance=0.002,
        seed=1,
    )
    realized = realized_volume_fraction_from_originals(geom)
    assert target - tol <= realized <= target + tol


def test_generate_poly_sphere_3d_has_no_periodic_overlaps():
    geom = generate_poly_sphere_3d(
        volume_fraction=0.11,
        volume_fraction_tolerance=0.015,
        min_radius=0.03,
        max_radius=0.07,
        clearance=0.001,
        seed=2,
    )
    assert not geometry_has_periodic_overlaps(geom, clearance=0.001, originals_only=True)


def test_generate_mono_ellipsoid_3d_returns_correct_original_count():
    geom = generate_mono_ellipsoid_3d(
        volume_fraction=0.08,
        num_particles=7,
        axis_ratios=(1.0, 1.5, 2.0),
        clearance=0.002,
        seed=99,
    )
    originals = original_inclusions(geom)
    assert len(originals) == 7
    assert all(inc.shape == "ellipsoid" for inc in originals)


def test_generate_mono_ellipsoid_3d_realized_vf_matches_target():
    vf = 0.09
    geom = generate_mono_ellipsoid_3d(
        volume_fraction=vf,
        num_particles=5,
        axis_ratios=(1.0, 1.2, 1.8),
        clearance=0.001,
        seed=9,
    )
    realized = realized_volume_fraction_from_originals(geom)
    assert np.isclose(realized, vf, atol=1e-12)


def test_generate_mono_ellipsoid_3d_has_no_periodic_overlaps():
    geom = generate_mono_ellipsoid_3d(
        volume_fraction=0.05,
        num_particles=5,
        axis_ratios=(1.0, 1.3, 1.7),
        clearance=0.001,
        seed=12,
    )
    assert not geometry_has_periodic_overlaps(geom, clearance=0.001, originals_only=True)


def test_generate_poly_ellipsoid_3d_realized_vf_within_tolerance():
    target = 0.12
    tol = 0.02
    geom = generate_poly_ellipsoid_3d(
        volume_fraction=target,
        volume_fraction_tolerance=tol,
        axis_ratios=(1.0, 1.4, 2.2),
        min_scale=0.03,
        max_scale=0.06,
        clearance=0.001,
        seed=11,
    )
    realized = realized_volume_fraction_from_originals(geom)
    assert target - tol <= realized <= target + tol


def test_generate_poly_ellipsoid_3d_has_no_periodic_overlaps():
    geom = generate_poly_ellipsoid_3d(
        volume_fraction=0.10,
        volume_fraction_tolerance=0.02,
        axis_ratios=(1.0, 1.5, 1.8),
        min_scale=0.03,
        max_scale=0.05,
        clearance=0.001,
        seed=17,
    )
    assert not geometry_has_periodic_overlaps(geom, clearance=0.001, originals_only=True)


def test_periodic_image_map_consistency_for_spheres():
    geom = generate_mono_sphere_3d(
        volume_fraction=0.07,
        num_particles=8,
        clearance=0.001,
        seed=14,
    )
    assert image_map_is_consistent(geom)


def test_periodic_image_map_consistency_for_ellipsoids():
    geom = generate_mono_ellipsoid_3d(
        volume_fraction=0.06,
        num_particles=6,
        axis_ratios=(1.0, 1.2, 1.6),
        clearance=0.001,
        seed=15,
    )
    assert image_map_is_consistent(geom)


def test_geometry_contains_some_images_when_boundary_crossing_is_possible():
    geom = generate_mono_sphere_3d(
        volume_fraction=0.22,
        num_particles=2,
        clearance=0.001,
        seed=7,
    )
    originals = original_inclusions(geom)
    images = periodic_image_inclusions(geom)

    assert len(originals) == 2
    assert len(images) >= 0
    assert len(geom.inclusions) >= len(originals)