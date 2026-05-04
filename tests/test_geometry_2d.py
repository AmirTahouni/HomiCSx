import numpy as np
import pytest

from homicsx.geometry import (
    generate_mono_circle_2d,
    generate_poly_circle_2d,
    generate_mono_ellipse_2d,
    generate_poly_ellipse_2d,
)
from homicsx.geometry.helpers import (
    circle_area,
    ellipse_area,
    monodisperse_circle_radius,
    monodisperse_ellipse_axes,
)
from homicsx.utils.validation import (
    geometry_has_periodic_overlaps,
    image_map_is_consistent,
    original_inclusions,
    periodic_image_inclusions,
    realized_volume_fraction_from_originals,
)


@pytest.fixture
def unit_square():
    return np.array([1.0, 1.0], dtype=float)


def test_mono_circle_radius_matches_target(unit_square):
    vf = 0.18
    n = 12
    r = monodisperse_circle_radius(vf, n, unit_square)
    achieved = n * circle_area(r) / np.prod(unit_square)
    assert np.isclose(achieved, vf, atol=1e-12)


def test_mono_ellipse_axes_match_target_area(unit_square):
    vf = 0.20
    n = 8
    axes = monodisperse_ellipse_axes(vf, n, unit_square, (1.0, 2.0))
    achieved = n * ellipse_area(axes) / np.prod(unit_square)
    assert np.isclose(achieved, vf, atol=1e-12)


def test_generate_mono_circle_2d_returns_correct_original_count():
    geom = generate_mono_circle_2d(
        volume_fraction=0.10,
        num_particles=10,
        clearance=0.005,
        seed=42,
    )
    assert len(original_inclusions(geom)) == 10
    assert geom.dim == 2


def test_generate_mono_circle_2d_has_no_periodic_overlaps():
    geom = generate_mono_circle_2d(
        volume_fraction=0.08,
        num_particles=8,
        clearance=0.002,
        seed=123,
    )
    assert not geometry_has_periodic_overlaps(geom, clearance=0.002, originals_only=True)


def test_generate_mono_circle_2d_realized_vf_matches_target():
    vf = 0.12
    geom = generate_mono_circle_2d(
        volume_fraction=vf,
        num_particles=12,
        clearance=0.001,
        seed=4,
    )
    realized = realized_volume_fraction_from_originals(geom)
    assert np.isclose(realized, vf, atol=1e-12)
    assert np.isclose(geom.realized_volume_fraction, vf, atol=1e-12)


def test_generate_poly_circle_2d_realized_vf_within_tolerance():
    target = 0.20
    tol = 0.02
    geom = generate_poly_circle_2d(
        volume_fraction=target,
        volume_fraction_tolerance=tol,
        min_radius=0.03,
        max_radius=0.07,
        clearance=0.001,
        seed=1,
    )
    realized = realized_volume_fraction_from_originals(geom)
    assert target - tol <= realized <= target + tol


def test_generate_mono_ellipse_2d_returns_correct_original_count():
    geom = generate_mono_ellipse_2d(
        volume_fraction=0.08,
        num_particles=7,
        axis_ratios=(1.0, 1.8),
        clearance=0.001,
        seed=99,
    )
    originals = original_inclusions(geom)
    assert len(originals) == 7
    assert all(inc.shape == "ellipse" for inc in originals)


def test_generate_mono_ellipse_2d_realized_vf_matches_target():
    vf = 0.09
    geom = generate_mono_ellipse_2d(
        volume_fraction=vf,
        num_particles=6,
        axis_ratios=(1.0, 1.5),
        clearance=0.001,
        seed=9,
    )
    realized = realized_volume_fraction_from_originals(geom)
    assert np.isclose(realized, vf, atol=1e-12)


def test_generate_poly_ellipse_2d_realized_vf_within_tolerance():
    target = 0.14
    tol = 0.02
    geom = generate_poly_ellipse_2d(
        volume_fraction=target,
        volume_fraction_tolerance=tol,
        axis_ratios=(1.0, 1.7),
        min_scale=0.03,
        max_scale=0.06,
        clearance=0.001,
        seed=11,
    )
    realized = realized_volume_fraction_from_originals(geom)
    assert target - tol <= realized <= target + tol


def test_periodic_image_map_consistency_for_circles():
    geom = generate_mono_circle_2d(
        volume_fraction=0.07,
        num_particles=8,
        clearance=0.001,
        seed=14,
    )
    assert image_map_is_consistent(geom)


def test_periodic_image_map_consistency_for_ellipses():
    geom = generate_mono_ellipse_2d(
        volume_fraction=0.06,
        num_particles=6,
        axis_ratios=(1.0, 1.4),
        clearance=0.001,
        seed=15,
    )
    assert image_map_is_consistent(geom)


def test_geometry_contains_images_or_not_but_is_consistent():
    geom = generate_mono_circle_2d(
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