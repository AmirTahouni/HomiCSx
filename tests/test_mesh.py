from __future__ import annotations

import importlib.util

import numpy as np
import pytest


HAS_GMSH = importlib.util.find_spec("gmsh") is not None
HAS_DOLFINX = importlib.util.find_spec("dolfinx") is not None

pytestmark = pytest.mark.skipif(not HAS_GMSH, reason="gmsh is not installed")


if HAS_GMSH:
    import gmsh

from homicsx.core.geometry import Inclusion, RVEGeometry
from homicsx.core.mesh import PhysicalTags
from homicsx.geometry import (
    generate_mono_circle_2d,
    generate_mono_ellipse_2d,
    generate_mono_sphere_3d,
    generate_mono_ellipsoid_3d,
)
from homicsx.mesh import build_gmsh_model, generate_mesh


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _physical_groups_by_dim():
    """
    Return physical groups as a dict:
        dim -> list[(tag_id, name)]
    """
    out: dict[int, list[tuple[int, str]]] = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        out.setdefault(dim, []).append((tag, name))
    return out


def _physical_group_name_to_tag(dim: int) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for d, tag in gmsh.model.getPhysicalGroups():
        if d != dim:
            continue
        mapping[gmsh.model.getPhysicalName(dim, tag)] = tag
    return mapping


def _build_model_in_clean_gmsh(
    geometry: RVEGeometry,
    *,
    min_size: float,
    max_size: float,
    **kwargs,
):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            geometry,
            min_size=min_size,
            max_size=max_size,
            **kwargs,
        )
        return info
    finally:
        gmsh.finalize()


# ------------------------------------------------------------
# fixtures
# ------------------------------------------------------------

@pytest.fixture
def physical_tags():
    return PhysicalTags()


@pytest.fixture
def simple_circle_geom():
    return generate_mono_circle_2d(
        volume_fraction=0.08,
        num_particles=6,
        clearance=0.002,
        seed=10,
    )


@pytest.fixture
def simple_ellipse_geom():
    return generate_mono_ellipse_2d(
        volume_fraction=0.08,
        num_particles=5,
        axis_ratios=(1.5, 1.0),
        clearance=0.002,
        seed=11,
    )


@pytest.fixture
def simple_sphere_geom():
    return generate_mono_sphere_3d(
        volume_fraction=0.04,
        num_particles=4,
        clearance=0.002,
        seed=12,
    )


@pytest.fixture
def simple_ellipsoid_geom():
    return generate_mono_ellipsoid_3d(
        volume_fraction=0.04,
        num_particles=4,
        axis_ratios=(1.0, 1.3, 1.8),
        clearance=0.002,
        seed=13,
    )


@pytest.fixture
def manual_multiphase_geom_2d():
    """
    Small hand-built 2D multi-phase geometry:
      phase 0 -> matrix
      phase 1 -> circle
      phase 2 -> ellipse
    """
    return RVEGeometry(
        dim=2,
        domain_size=np.array([1.0, 1.0], dtype=float),
        phase_ids=(0, 1, 2),
        inclusions=[
            Inclusion(
                center=np.array([0.30, 0.30]),
                phase_id=1,
                shape="circle",
                radii=np.array([0.10]),
            ),
            Inclusion(
                center=np.array([0.72, 0.68]),
                phase_id=2,
                shape="ellipse",
                radii=np.array([0.08, 0.05]),
            ),
        ],
    )


@pytest.fixture
def manual_multiphase_geom_3d():
    """
    Small hand-built 3D multi-phase geometry:
      phase 0 -> matrix
      phase 1 -> sphere
      phase 2 -> ellipsoid
    """
    return RVEGeometry(
        dim=3,
        domain_size=np.array([1.0, 1.0, 1.0], dtype=float),
        phase_ids=(0, 1, 2),
        inclusions=[
            Inclusion(
                center=np.array([0.30, 0.30, 0.30]),
                phase_id=1,
                shape="sphere",
                radii=np.array([0.12]),
            ),
            Inclusion(
                center=np.array([0.72, 0.68, 0.65]),
                phase_id=2,
                shape="ellipsoid",
                radii=np.array([0.08, 0.05, 0.06]),
            ),
        ],
    )


@pytest.fixture
def periodic_crossing_geom_2d():
    """
    Hand-built geometry with a periodic image already included.
    This tests the mesher's simple design choice:
    mesh all inclusions exactly as stored in geometry.inclusions.
    """
    return RVEGeometry(
        dim=2,
        domain_size=np.array([1.0, 1.0], dtype=float),
        phase_ids=(0, 1),
        inclusions=[
            Inclusion(
                center=np.array([0.96, 0.50]),
                phase_id=1,
                shape="circle",
                radii=np.array([0.10]),
                periodic_source_id=None,
                metadata={"is_original": True},
            ),
            Inclusion(
                center=np.array([-0.04, 0.50]),
                phase_id=1,
                shape="circle",
                radii=np.array([0.10]),
                periodic_source_id=0,
                metadata={"is_original": False},
            ),
        ],
    )


# ------------------------------------------------------------
# build_gmsh_model tests: 2D
# ------------------------------------------------------------

def test_build_gmsh_model_2d_circle_returns_expected_metadata(simple_circle_geom, physical_tags):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            simple_circle_geom,
            min_size=0.02,
            max_size=0.08,
            physical_tags=physical_tags,
        )

        assert info["dim"] == 2
        assert np.allclose(info["domain_size"], np.array([1.0, 1.0]))
        assert info["matrix_phase_id"] == 0
        assert info["matrix_cell_tag"] == 1
        assert 1 in info["phase_cell_tags"]
        assert info["phase_cell_tags"][1] == 11

        assert isinstance(info["matrix_entity_tags"], list)
        assert isinstance(info["phase_entity_tags"], dict)
        assert isinstance(info["boundary_entities"], dict)
        assert isinstance(info["interface_entity_tags"], list)

        assert "left" in info["boundary_entities"]
        assert "right" in info["boundary_entities"]
        assert "bottom" in info["boundary_entities"]
        assert "top" in info["boundary_entities"]

    finally:
        gmsh.finalize()


def test_build_gmsh_model_2d_adds_expected_physical_groups(simple_circle_geom):
    gmsh.initialize()
    try:
        build_gmsh_model(
            simple_circle_geom,
            min_size=0.02,
            max_size=0.08,
        )

        groups = _physical_groups_by_dim()
        assert 2 in groups
        assert 1 in groups

        cell_names = {name for _, name in groups[2]}
        facet_names = {name for _, name in groups[1]}

        assert "Matrix" in cell_names
        assert "Phase_1" in cell_names

        assert "Left" in facet_names
        assert "Right" in facet_names
        assert "Bottom" in facet_names
        assert "Top" in facet_names

    finally:
        gmsh.finalize()


def test_build_gmsh_model_2d_ellipse_builds(simple_ellipse_geom):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            simple_ellipse_geom,
            min_size=0.02,
            max_size=0.08,
        )
        print('info created succ')
        assert info["dim"] == 2
        assert len(info["phase_entity_tags"][1]) >= 1
    finally:
        gmsh.finalize()


# ------------------------------------------------------------
# build_gmsh_model tests: 3D
# ------------------------------------------------------------

def test_build_gmsh_model_3d_sphere_returns_expected_metadata(simple_sphere_geom):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            simple_sphere_geom,
            min_size=0.05,
            max_size=0.12,
        )

        assert info["dim"] == 3
        assert np.allclose(info["domain_size"], np.array([1.0, 1.0, 1.0]))
        assert info["matrix_phase_id"] == 0
        assert info["matrix_cell_tag"] == 1
        assert info["phase_cell_tags"][1] == 11

        assert "near" in info["boundary_entities"]
        assert "far" in info["boundary_entities"]

    finally:
        gmsh.finalize()


def test_build_gmsh_model_3d_adds_expected_physical_groups(simple_sphere_geom):
    gmsh.initialize()
    try:
        build_gmsh_model(
            simple_sphere_geom,
            min_size=0.05,
            max_size=0.12,
        )

        groups = _physical_groups_by_dim()
        assert 3 in groups
        assert 2 in groups

        cell_names = {name for _, name in groups[3]}
        facet_names = {name for _, name in groups[2]}

        assert "Matrix" in cell_names
        assert "Phase_1" in cell_names

        assert "Left" in facet_names
        assert "Right" in facet_names
        assert "Bottom" in facet_names
        assert "Top" in facet_names
        assert "Near" in facet_names
        assert "Far" in facet_names

    finally:
        gmsh.finalize()


def test_build_gmsh_model_3d_ellipsoid_builds(simple_ellipsoid_geom):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            simple_ellipsoid_geom,
            min_size=0.05,
            max_size=0.12,
        )
        assert info["dim"] == 3
        assert len(info["phase_entity_tags"][1]) >= 1
    finally:
        gmsh.finalize()


# ------------------------------------------------------------
# multi-phase tests
# ------------------------------------------------------------

def test_multiphase_2d_phase_tracking_metadata(manual_multiphase_geom_2d, physical_tags):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            manual_multiphase_geom_2d,
            min_size=0.02,
            max_size=0.08,
            physical_tags=physical_tags,
        )

        assert info["matrix_phase_id"] == 0
        assert info["matrix_cell_tag"] == 1
        assert info["phase_cell_tags"][1] == 11
        assert info["phase_cell_tags"][2] == 12

        assert 1 in info["phase_entity_tags"]
        assert 2 in info["phase_entity_tags"]
        assert len(info["phase_entity_tags"][1]) >= 1
        assert len(info["phase_entity_tags"][2]) >= 1

    finally:
        gmsh.finalize()


def test_multiphase_2d_physical_group_names(manual_multiphase_geom_2d):
    gmsh.initialize()
    try:
        build_gmsh_model(
            manual_multiphase_geom_2d,
            min_size=0.02,
            max_size=0.08,
        )

        name_to_tag = _physical_group_name_to_tag(2)
        assert "Matrix" in name_to_tag
        assert "Phase_1" in name_to_tag
        assert "Phase_2" in name_to_tag

        assert name_to_tag["Matrix"] == 1
        assert name_to_tag["Phase_1"] == 11
        assert name_to_tag["Phase_2"] == 12

    finally:
        gmsh.finalize()


def test_multiphase_3d_phase_tracking_metadata(manual_multiphase_geom_3d):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            manual_multiphase_geom_3d,
            min_size=0.05,
            max_size=0.12,
        )

        assert info["matrix_cell_tag"] == 1
        assert info["phase_cell_tags"][1] == 11
        assert info["phase_cell_tags"][2] == 12

        assert len(info["phase_entity_tags"][1]) >= 1
        assert len(info["phase_entity_tags"][2]) >= 1

    finally:
        gmsh.finalize()


# ------------------------------------------------------------
# periodic-image handling
# ------------------------------------------------------------

def test_periodic_image_geometry_builds_in_2d(periodic_crossing_geom_2d):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            periodic_crossing_geom_2d,
            min_size=0.02,
            max_size=0.08,
        )

        assert info["dim"] == 2
        assert info["phase_cell_tags"][1] == 11
        assert len(info["phase_entity_tags"][1]) >= 1

    finally:
        gmsh.finalize()


# ------------------------------------------------------------
# options / refinement behavior
# ------------------------------------------------------------

def test_build_gmsh_model_without_interface_refinement(simple_circle_geom):
    gmsh.initialize()
    try:
        info = build_gmsh_model(
            simple_circle_geom,
            min_size=0.02,
            max_size=0.08,
            refine_interfaces=False,
        )

        assert info["dim"] == 2
        assert isinstance(info["interface_entity_tags"], list)
    finally:
        gmsh.finalize()


def test_build_gmsh_model_with_custom_physical_tags(simple_circle_geom):
    custom = PhysicalTags(
        matrix=100,
        phase_tag_offset=200,
        left=301,
        right=302,
        bottom=303,
        top=304,
        near=305,
        far=306,
    )

    gmsh.initialize()
    try:
        info = build_gmsh_model(
            simple_circle_geom,
            min_size=0.02,
            max_size=0.08,
            physical_tags=custom,
        )

        assert info["matrix_cell_tag"] == 100
        assert info["phase_cell_tags"][1] == 201

        name_to_tag_cells = _physical_group_name_to_tag(2)
        name_to_tag_facets = _physical_group_name_to_tag(1)

        assert name_to_tag_cells["Matrix"] == 100
        assert name_to_tag_cells["Phase_1"] == 201
        assert name_to_tag_facets["Left"] == 301
        assert name_to_tag_facets["Right"] == 302
        assert name_to_tag_facets["Bottom"] == 303
        assert name_to_tag_facets["Top"] == 304

    finally:
        gmsh.finalize()


# ------------------------------------------------------------
# full generate_mesh smoke tests
# ------------------------------------------------------------

@pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx is not installed")
def test_generate_mesh_2d_smoke(simple_circle_geom):
    mesh, cell_tags, facet_tags = generate_mesh(
        simple_circle_geom,
        min_size=0.02,
        max_size=0.08,
    )

    assert mesh.topology.dim == 2
    assert cell_tags is not None
    assert facet_tags is not None


@pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx is not installed")
def test_generate_mesh_2d_multiphase_smoke(manual_multiphase_geom_2d):
    mesh, cell_tags, facet_tags = generate_mesh(
        manual_multiphase_geom_2d,
        min_size=0.02,
        max_size=0.08,
    )

    assert mesh.topology.dim == 2
    assert cell_tags is not None
    assert facet_tags is not None


@pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx is not installed")
def test_generate_mesh_3d_smoke(simple_sphere_geom):
    mesh, cell_tags, facet_tags = generate_mesh(
        simple_sphere_geom,
        min_size=0.05,
        max_size=0.12,
    )

    assert mesh.topology.dim == 3
    assert cell_tags is not None
    assert facet_tags is not None


@pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx is not installed")
def test_generate_mesh_3d_multiphase_smoke(manual_multiphase_geom_3d):
    mesh, cell_tags, facet_tags = generate_mesh(
        manual_multiphase_geom_3d,
        min_size=0.05,
        max_size=0.12,
    )

    assert mesh.topology.dim == 3
    assert cell_tags is not None
    assert facet_tags is not None


# ------------------------------------------------------------
# direct PhysicalTags tests
# ------------------------------------------------------------

def test_physical_tags_cell_tag_mapping_defaults():
    tags = PhysicalTags()

    assert tags.cell_tag_for_phase(0, matrix_phase_id=0) == 1
    assert tags.cell_tag_for_phase(1, matrix_phase_id=0) == 11
    assert tags.cell_tag_for_phase(2, matrix_phase_id=0) == 12
    assert tags.cell_name_for_phase(0, matrix_phase_id=0) == "Matrix"
    assert tags.cell_name_for_phase(3, matrix_phase_id=0) == "Phase_3"


def test_physical_tags_boundary_mapping_2d():
    tags = PhysicalTags()
    mapping = tags.boundary_name_to_tag(2)

    assert mapping["left"] == tags.left
    assert mapping["right"] == tags.right
    assert mapping["bottom"] == tags.bottom
    assert mapping["top"] == tags.top


def test_physical_tags_boundary_mapping_3d():
    tags = PhysicalTags()
    mapping = tags.boundary_name_to_tag(3)

    assert mapping["left"] == tags.left
    assert mapping["right"] == tags.right
    assert mapping["bottom"] == tags.bottom
    assert mapping["top"] == tags.top
    assert mapping["near"] == tags.near
    assert mapping["far"] == tags.far