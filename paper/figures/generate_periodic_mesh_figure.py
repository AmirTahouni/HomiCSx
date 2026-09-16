"""Generate representative 2D/3D periodic-conforming mesh panels."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from homicsx import (
    Inclusion,
    MeshSettings,
    PhysicalTags,
    RVEGeometry,
    generate_mesh,
)


OUTPUT = Path(__file__).resolve().parent


def mesh_geometry(dim: int):
    if dim == 2:
        inclusions = [
            Inclusion(center=(0.0, 0.5), phase_id=1, shape="circle", radii=0.19),
            Inclusion(
                center=(1.0, 0.5),
                phase_id=1,
                shape="circle",
                radii=0.19,
                periodic_source_id=0,
            ),
        ]
        sizes = (0.055, 0.11)
    else:
        inclusions = [
            Inclusion(
                center=(0.5, 0.5, 0.5),
                phase_id=1,
                shape="sphere",
                radii=0.22,
            )
        ]
        sizes = (0.16, 0.27)
    geometry = RVEGeometry(
        dim=dim,
        domain_size=(1.0,) * dim,
        inclusions=inclusions,
        metadata={"figure": "periodic_mesh"},
    )
    tags = PhysicalTags()
    domain, cell_tags, unused_facet_tags = generate_mesh(
        geometry=geometry,
        mesh_settings=MeshSettings(
            min_size=sizes[0],
            max_size=sizes[1],
            physical_tags=tags,
            periodic_mesh=True,
            verbosity=0,
        ),
    )
    return domain, cell_tags, tags


def cell_vertices(domain) -> np.ndarray:
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    connectivity = domain.topology.connectivity(tdim, 0)
    count = domain.topology.index_map(tdim).size_local
    return np.asarray([connectivity.links(cell) for cell in range(count)], dtype=int)


def draw_2d(ax):
    domain, cell_tags, tags = mesh_geometry(2)
    coordinates = domain.geometry.x[:, :2]
    triangles = cell_vertices(domain)
    ax.triplot(
        coordinates[:, 0],
        coordinates[:, 1],
        triangles,
        color="#8c8c8c",
        linewidth=0.35,
    )
    inclusion_cells = cell_tags.find(tags.cell_tag_for_phase(1))
    centroids = coordinates[triangles[inclusion_cells]].mean(axis=1)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=4,
        color="#ef8a62",
        alpha=0.75,
        linewidths=0,
    )
    left = np.isclose(coordinates[:, 0], 0.0, atol=1.0e-8)
    right = np.isclose(coordinates[:, 0], 1.0, atol=1.0e-8)
    ax.scatter(coordinates[left, 0], coordinates[left, 1], s=18, facecolors="none", edgecolors="#2166ac", linewidths=0.8)
    ax.scatter(coordinates[right, 0], coordinates[right, 1], s=14, marker="+", color="#b2182b", linewidths=0.9)
    ax.set_aspect("equal")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.set_title("(a) 2D boundary-split inclusion")
    ax.text(0.02, 0.02, "master", color="#2166ac", fontsize=7, transform=ax.transAxes)
    ax.text(0.98, 0.02, "slave", color="#b2182b", fontsize=7, ha="right", transform=ax.transAxes)


def draw_3d(ax):
    domain, cell_tags, tags = mesh_geometry(3)
    coordinates = domain.geometry.x[:, :3]
    tetrahedra = cell_vertices(domain)
    edges = set()
    for tetrahedron in tetrahedra:
        for first, second in combinations(tetrahedron, 2):
            edges.add(tuple(sorted((int(first), int(second)))))
    segments = np.asarray([[coordinates[first], coordinates[second]] for first, second in edges])
    ax.add_collection3d(
        Line3DCollection(segments, colors="#a6a6a6", linewidths=0.22, alpha=0.22)
    )
    inclusion_cells = cell_tags.find(tags.cell_tag_for_phase(1))
    centroids = coordinates[tetrahedra[inclusion_cells]].mean(axis=1)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        s=7,
        color="#ef8a62",
        alpha=0.75,
        depthshade=False,
    )
    near = np.isclose(coordinates[:, 2], 0.0, atol=1.0e-8)
    far = np.isclose(coordinates[:, 2], 1.0, atol=1.0e-8)
    ax.scatter(*coordinates[near].T, s=9, facecolors="none", edgecolors="#2166ac", linewidths=0.6, depthshade=False)
    ax.scatter(*coordinates[far].T, s=9, marker="+", color="#b2182b", linewidths=0.6, depthshade=False)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="$X_1$", ylabel="$X_2$", zlabel="$X_3$")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_title("(b) 3D spherical inclusion")
    ax.grid(False)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    figure = plt.figure(figsize=(7.15, 3.45), constrained_layout=True)
    draw_2d(figure.add_subplot(1, 2, 1))
    draw_3d(figure.add_subplot(1, 2, 2, projection="3d"))
    for extension in ("pdf", "png"):
        path = OUTPUT / f"periodic_meshes.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
