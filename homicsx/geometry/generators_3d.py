from __future__ import annotations

import numpy as np

from homicsx.core.geometry import (
    Inclusion, 
    # PeriodicityInfo,
    GeometryInput, 
    RVEGeometry,
)

from .helpers import (
    as_domain_size,
    ellipsoid_volume,
    generate_periodic_image_centers,
    _is_mesh_valid_primitive,
    monodisperse_ellipsoid_axes,
    monodisperse_sphere_radius,
    normalize_axis_ratios,
    _overlaps_existing_periodic,
    sample_random_center,
    sphere_volume,
)


def _primitive_volume(shape: str, axes: np.ndarray) -> float:
    if shape == "sphere":
        return sphere_volume(float(axes[0]))
    if shape == "ellipsoid":
        return ellipsoid_volume(axes)
    raise NotImplementedError


def _build_geometry_3d(
    centers: list[np.ndarray],
    axes_list: list[np.ndarray],
    shapes: list[str],
    domain_size: np.ndarray,
    target_volume_fraction: float | None,
    seed: int | None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    allow_overlap: bool = False,
    metadata: dict | None = None,
) -> RVEGeometry:
    inclusions: list[Inclusion] = []
    image_map: dict[int, list[int]] = {}

    original_indices = []

    for center, axes, shape in zip(centers, axes_list, shapes):
        idx = len(inclusions)
        original_indices.append(idx)
        radii = np.array([axes[0]], dtype=float) if shape == "sphere" else np.asarray(axes, dtype=float)
        inclusions.append(
            Inclusion(
                center=center,
                phase_id=1,
                shape=shape,
                radii=radii,
                interphase_thickness_ratio=interphase_thickness_ratio,
                interphase_phase_id=interphase_phase_id,
                periodic_source_id=None,
                metadata={"is_original": True},
            )
        )

    for source_idx, center, axes, shape in zip(original_indices, centers, axes_list, shapes):
        image_centers = generate_periodic_image_centers(center, axes, domain_size)
        if not image_centers:
            continue
        image_map[source_idx] = []
        for image_center in image_centers:
            image_idx = len(inclusions)
            radii = np.array([axes[0]], dtype=float) if shape == "sphere" else np.asarray(axes, dtype=float)
            inclusions.append(
                Inclusion(
                    center=image_center,
                    phase_id=1,
                    shape=shape,
                    radii=radii,
                    interphase_thickness_ratio=interphase_thickness_ratio,
                    interphase_phase_id=interphase_phase_id,
                    periodic_source_id=source_idx,
                    metadata={"is_original": False},
                )
            )
            image_map[source_idx].append(image_idx)

    # realized_vf = sum(
    #     inc.total_volume for inc in inclusions if inc.periodic_source_id is None
    # ) / float(np.prod(domain_size))

    domain_volume = float(np.prod(domain_size))
    nominal_vf = sum(
        inc.total_volume for inc in inclusions if inc.periodic_source_id is None
    ) / domain_volume

    if allow_overlap:
        # Apply Poisson overlap correction for randomly overlapping inclusions
        # VF_actual = 1 - exp(-VF_nominal)
        realized_vf = 1.0 - np.exp(-nominal_vf)
    else:
        realized_vf = nominal_vf

    return RVEGeometry(
        dim=3,
        domain_size=domain_size,
        inclusions=inclusions,
        target_volume_fraction=target_volume_fraction,
        realized_volume_fraction=realized_vf,
        seed=seed,
        metadata={} if metadata is None else dict(metadata),
    )


def _generate_mono_sphere_3d(
    volume_fraction: float,
    num_particles: int,
    clearance: float,
    *,
    domain_size: float | tuple[float, float, float] | list[float] | np.ndarray = 1.0,
    max_restarts: int = 100,
    attempts_per_restart: int = 10000,
    seed: int | None = None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    verbosity: int = 0,
) -> RVEGeometry:
    domain = as_domain_size(3, domain_size)
    radius = monodisperse_sphere_radius(volume_fraction, num_particles, domain)
    fixed_axes = np.array([radius, radius, radius], dtype=float)

    rng = np.random.default_rng(seed)

    for restart in range(max_restarts):
        centers = []
        axes_list = []
        shapes = []

        success = True
        for _ in range(num_particles):
            accepted = False
            for _ in range(attempts_per_restart):
                center = sample_random_center(rng, domain)
                if not _is_mesh_valid_primitive(center, fixed_axes, clearance, domain):
                    continue
                if _overlaps_existing_periodic(center, fixed_axes, "sphere", centers, axes_list, shapes, domain, clearance):
                    continue
                centers.append(center)
                axes_list.append(fixed_axes.copy())
                shapes.append("sphere")
                accepted = True
                break
            if not accepted:
                success = False
                break

        if success:
            return _build_geometry_3d(
                centers,
                axes_list,
                shapes,
                domain,
                volume_fraction,
                seed,
                interphase_thickness_ratio,
                interphase_phase_id,
                metadata={
                    "generator": "generate_mono_sphere_3d",
                    "radius": radius,
                    "clearance": clearance,
                },
            )

    raise RuntimeError("Failed to generate monodisperse sphere geometry.")


def _generate_poly_sphere_3d(
    volume_fraction: float,
    volume_fraction_tolerance: float,
    min_radius: float,
    max_radius: float,
    clearance: float,
    *,
    domain_size: float | tuple[float, float, float] | list[float] | np.ndarray = 1.0,
    max_restarts: int = 100,
    attempts_per_restart: int = 100000,
    seed: int | None = None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    allow_overlap: bool = False,
) -> RVEGeometry:
    if min_radius <= 0.0 or max_radius <= 0.0 or min_radius > max_radius:
        raise ValueError("Invalid radius range.")

    domain = as_domain_size(3, domain_size)
    rng = np.random.default_rng(seed)

    lower_target = max(0.0, volume_fraction - volume_fraction_tolerance)
    upper_target = min(1.0, volume_fraction + volume_fraction_tolerance)
    domain_volume = float(np.prod(domain))

    for restart in range(max_restarts):
        centers = []
        axes_list = []
        shapes = []
        
        if allow_overlap:
            # For randomly overlapping inclusions:
            #   VF_actual = 1 - exp(-VF_nominal)
            # So: nominal_volume = -domain_volume * ln(1 - VF_actual)
            nominal_volume = 0.0
        else:
            current_volume = 0.0

        for _ in range(attempts_per_restart):
            # Check termination condition
            if allow_overlap:
                current_vf = 1.0 - np.exp(-nominal_volume / domain_volume)
                if current_vf >= lower_target:
                    return _build_geometry_3d(
                        centers,
                        axes_list,
                        shapes,
                        domain,
                        volume_fraction,
                        seed,
                        interphase_thickness_ratio,
                        interphase_phase_id,
                        metadata={
                            "generator": "generate_poly_sphere_3d",
                            "min_radius": min_radius,
                            "max_radius": max_radius,
                            "clearance": clearance,
                            "allow_overlap": True,
                            "target_vf": volume_fraction,
                            "estimated_vf": current_vf,
                            "num_inclusions": len(centers),
                        },
                    )
            else:
                current_vf = current_volume / domain_volume
                if current_vf >= lower_target:
                    return _build_geometry_3d(
                        centers,
                        axes_list,
                        shapes,
                        domain,
                        volume_fraction,
                        seed,
                        interphase_thickness_ratio,
                        interphase_phase_id,
                        allow_overlap=allow_overlap,
                        metadata={
                            "generator": "generate_poly_sphere_3d",
                            "min_radius": min_radius,
                            "max_radius": max_radius,
                            "clearance": clearance,
                        },
                    )

            r = float(rng.uniform(min_radius, max_radius))
            axes = np.array([r, r, r], dtype=float)
            center = sample_random_center(rng, domain)

            if not _is_mesh_valid_primitive(center, axes, clearance, domain):
                continue
            if _overlaps_existing_periodic(
                center, axes, "sphere", centers, axes_list, shapes, 
                domain, clearance, allow_overlap=allow_overlap
            ):
                continue

            inc_volume = sphere_volume(r)
            
            if allow_overlap:
                new_nominal = nominal_volume + inc_volume
                new_vf_actual = 1.0 - np.exp(-new_nominal / domain_volume)
                if new_vf_actual > upper_target:
                    continue
                nominal_volume = new_nominal
            else:
                new_volume = current_volume + inc_volume
                new_vf = new_volume / domain_volume
                if new_vf > upper_target:
                    continue
                current_volume = new_volume

            centers.append(center)
            axes_list.append(axes)
            shapes.append("sphere")

    raise RuntimeError(
        f"Failed to generate polydisperse sphere geometry. "
        f"Target VF: {volume_fraction:.3f}"
    )


def _generate_mono_ellipsoid_3d(
    volume_fraction: float,
    num_particles: int,
    axis_ratios,
    clearance: float,
    *,
    domain_size: float | tuple[float, float, float] | list[float] | np.ndarray = 1.0,
    max_restarts: int = 100,
    attempts_per_restart: int = 10000,
    seed: int | None = None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    verbosity: int = 0,
) -> RVEGeometry:
    domain = as_domain_size(3, domain_size)
    ratios = normalize_axis_ratios(axis_ratios)
    if ratios.size != 3:
        raise ValueError("axis_ratios must have length 3 for ellipsoids.")

    fixed_axes = monodisperse_ellipsoid_axes(volume_fraction, num_particles, domain, ratios)
    rng = np.random.default_rng(seed)

    for restart in range(max_restarts):
        centers = []
        axes_list = []
        shapes = []

        success = True
        for _ in range(num_particles):
            accepted = False
            for _ in range(attempts_per_restart):
                center = sample_random_center(rng, domain)
                if not _is_mesh_valid_primitive(center, fixed_axes, clearance, domain):
                    continue
                if _overlaps_existing_periodic(center, fixed_axes, "ellipsoid", centers, axes_list, shapes, domain, clearance):
                    continue
                centers.append(center)
                axes_list.append(fixed_axes.copy())
                shapes.append("ellipsoid")
                accepted = True
                break
            if not accepted:
                success = False
                break

        if success:
            return _build_geometry_3d(
                centers,
                axes_list,
                shapes,
                domain,
                volume_fraction,
                seed,
                interphase_thickness_ratio,
                interphase_phase_id,
                metadata={
                    "generator": "generate_mono_ellipsoid_3d",
                    "axis_ratios": ratios.tolist(),
                    "semi_axes": fixed_axes.tolist(),
                    "clearance": clearance,
                    "orientation_mode": "axis_aligned_only",
                },
            )

    raise RuntimeError("Failed to generate monodisperse ellipsoid geometry.")


def _generate_poly_ellipsoid_3d(
    volume_fraction: float,
    volume_fraction_tolerance: float,
    axis_ratios,
    min_scale: float,
    max_scale: float,
    clearance: float,
    *,
    domain_size: float | tuple[float, float, float] | list[float] | np.ndarray = 1.0,
    max_restarts: int = 100,
    attempts_per_restart: int = 100000,
    seed: int | None = None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    allow_overlap: bool = False,
) -> RVEGeometry:
    if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
        raise ValueError("Invalid scale range.")

    domain = as_domain_size(3, domain_size)
    ratios = normalize_axis_ratios(axis_ratios)
    if ratios.size != 3:
        raise ValueError("axis_ratios must have length 3 for ellipsoids.")

    rng = np.random.default_rng(seed)

    lower_target = max(0.0, volume_fraction - volume_fraction_tolerance)
    upper_target = min(1.0, volume_fraction + volume_fraction_tolerance)
    domain_volume = float(np.prod(domain))

    for restart in range(max_restarts):
        centers = []
        axes_list = []
        shapes = []
        
        if allow_overlap:
            # Same analytical correction for overlapping ellipsoids
            nominal_volume = 0.0
        else:
            current_volume = 0.0

        for _ in range(attempts_per_restart):
            # Check termination condition
            if allow_overlap:
                current_vf = 1.0 - np.exp(-nominal_volume / domain_volume)
                if current_vf >= lower_target:
                    return _build_geometry_3d(
                        centers,
                        axes_list,
                        shapes,
                        domain,
                        volume_fraction,
                        seed,
                        interphase_thickness_ratio,
                        interphase_phase_id,
                        allow_overlap=allow_overlap,
                        metadata={
                            "generator": "generate_poly_ellipsoid_3d",
                            "axis_ratios": ratios.tolist(),
                            "min_scale": min_scale,
                            "max_scale": max_scale,
                            "clearance": clearance,
                            "orientation_mode": "axis_aligned_only",
                            "allow_overlap": True,
                            "target_vf": volume_fraction,
                            "estimated_vf": current_vf,
                            "num_inclusions": len(centers),
                        },
                    )
            else:
                current_vf = current_volume / domain_volume
                if current_vf >= lower_target:
                    return _build_geometry_3d(
                        centers,
                        axes_list,
                        shapes,
                        domain,
                        volume_fraction,
                        seed,
                        interphase_thickness_ratio,
                        interphase_phase_id,
                        metadata={
                            "generator": "generate_poly_ellipsoid_3d",
                            "axis_ratios": ratios.tolist(),
                            "min_scale": min_scale,
                            "max_scale": max_scale,
                            "clearance": clearance,
                            "orientation_mode": "axis_aligned_only",
                        },
                    )

            scale = float(rng.uniform(min_scale, max_scale))
            axes = scale * ratios
            center = sample_random_center(rng, domain)

            if not _is_mesh_valid_primitive(center, axes, clearance, domain):
                continue
            if _overlaps_existing_periodic(
                center, axes, "ellipsoid", centers, axes_list, shapes, 
                domain, clearance, allow_overlap=allow_overlap
            ):
                continue

            inc_volume = ellipsoid_volume(axes)
            
            if allow_overlap:
                new_nominal = nominal_volume + inc_volume
                new_vf_actual = 1.0 - np.exp(-new_nominal / domain_volume)
                if new_vf_actual > upper_target:
                    continue
                nominal_volume = new_nominal
            else:
                new_volume = current_volume + inc_volume
                new_vf = new_volume / domain_volume
                if new_vf > upper_target:
                    continue
                current_volume = new_volume

            centers.append(center)
            axes_list.append(axes)
            shapes.append("ellipsoid")

    raise RuntimeError(
        f"Failed to generate polydisperse ellipsoid geometry. "
        f"Target VF: {volume_fraction:.3f}"
    )


def _generate_mono_3d(
    input_data: GeometryInput,
) -> RVEGeometry:
    volume_fraction = input_data.volume_fraction
    num_particles = input_data.num_particles
    clearance = input_data.clearance
    domain_size = input_data.domain_size
    shape = input_data.shape
    axis_ratios = input_data.axis_ratios
    max_restarts = input_data.max_restarts
    attempts_per_restart = input_data.attempts_per_restart
    seed = input_data.seed
    interphase_thickness_ratio = input_data.interphase_thickness_ratio
    interphase_phase_id = input_data.interphase_phase_id
    # verbosity = input_data.verbosity

    if shape == "sphere":
        return _generate_mono_sphere_3d(
            volume_fraction=volume_fraction,
            num_particles=num_particles,
            clearance=clearance,
            domain_size=domain_size,
            max_restarts=max_restarts,
            attempts_per_restart=attempts_per_restart,
            seed=seed,
            interphase_thickness_ratio = interphase_thickness_ratio,
            interphase_phase_id = interphase_phase_id,
            # verbosity=verbosity,
        )
    if shape == "ellipsoid":
        return _generate_mono_ellipsoid_3d(
            volume_fraction=volume_fraction,
            num_particles=num_particles,
            axis_ratios=axis_ratios,
            clearance=clearance,
            domain_size=domain_size,
            max_restarts=max_restarts,
            attempts_per_restart=attempts_per_restart,
            seed=seed,
            interphase_thickness_ratio = interphase_thickness_ratio,
            interphase_phase_id = interphase_phase_id,
            # verbosity=verbosity,
        )
    raise ValueError(f"Unsupported 3D shape: {shape}")


def _generate_poly_3d(
    input_data: GeometryInput
) -> RVEGeometry:
    volume_fraction = input_data.volume_fraction
    volume_fraction_tolerance = input_data.volume_fraction_tolerance
    clearance = input_data.clearance
    domain_size = input_data.domain_size
    shape = input_data.shape
    min_radius = input_data.min_radius
    max_radius = input_data.max_radius
    axis_ratios = input_data.axis_ratios
    min_scale = input_data.min_scale
    max_scale = input_data.max_scale
    max_restarts = input_data.max_restarts
    attempts_per_restart = input_data.attempts_per_restart
    seed = input_data.seed
    interphase_thickness_ratio = input_data.interphase_thickness_ratio
    interphase_phase_id = input_data.interphase_phase_id
    # verbosity = input_data.verbosity
    allow_overlap=input_data.allow_overlap

    if shape == "sphere":
        if min_radius is None or max_radius is None:
            raise ValueError("min_radius and max_radius are required for spheres.")
        return _generate_poly_sphere_3d(
            volume_fraction=volume_fraction,
            volume_fraction_tolerance=volume_fraction_tolerance,
            min_radius=min_radius,
            max_radius=max_radius,
            clearance=clearance,
            domain_size=domain_size,
            max_restarts=max_restarts,
            attempts_per_restart=attempts_per_restart,
            seed=seed,
            interphase_thickness_ratio = interphase_thickness_ratio,
            interphase_phase_id = interphase_phase_id,
            # verbosity=verbosity,
            allow_overlap=allow_overlap,
        )
    if shape == "ellipsoid":
        if min_scale is None or max_scale is None:
            raise ValueError("min_scale and max_scale are required for ellipsoids.")
        return _generate_poly_ellipsoid_3d(
            volume_fraction=volume_fraction,
            volume_fraction_tolerance=volume_fraction_tolerance,
            axis_ratios=axis_ratios,
            min_scale=min_scale,
            max_scale=max_scale,
            clearance=clearance,
            domain_size=domain_size,
            max_restarts=max_restarts,
            attempts_per_restart=attempts_per_restart,
            seed=seed,
            interphase_thickness_ratio = interphase_thickness_ratio,
            interphase_phase_id = interphase_phase_id,
            # verbosity=verbosity,
            allow_overlap=allow_overlap,
        )
    raise ValueError(f"Unsupported 3D shape: {shape}")


__all__ = [
    
]


