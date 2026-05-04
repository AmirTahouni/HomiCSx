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
    circle_area,
    ellipse_area,
    generate_periodic_image_centers,
    _is_mesh_valid_primitive,
    monodisperse_circle_radius,
    monodisperse_ellipse_axes,
    normalize_axis_ratios,
    _overlaps_existing_periodic,
    sample_random_center,
)


def _primitive_area(shape: str, axes: np.ndarray) -> float:
    if shape == "circle":
        return circle_area(float(axes[0]))
    if shape == "ellipse":
        return ellipse_area(axes)
    raise NotImplementedError


def _build_geometry_2d(
    centers: list[np.ndarray],
    axes_list: list[np.ndarray],
    shapes: list[str],
    domain_size: np.ndarray,
    target_volume_fraction: float | None,
    seed: int | None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    metadata: dict | None = None,
    allow_overlap: bool = False,
) -> RVEGeometry:
    inclusions: list[Inclusion] = []
    image_map: dict[int, list[int]] = {}
    original_indices = []

    for center, axes, shape in zip(centers, axes_list, shapes):
        idx = len(inclusions)
        original_indices.append(idx)
        radii = np.array([axes[0]], dtype=float) if shape == "circle" else np.asarray(axes, dtype=float)
        
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
            radii = np.array([axes[0]], dtype=float) if shape == "circle" else np.asarray(axes, dtype=float)
            
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

    domain_area = float(np.prod(domain_size))
    nominal_vf = sum(
        inc.total_volume for inc in inclusions if inc.periodic_source_id is None
    ) / domain_area
    
    if allow_overlap:
        # Apply Poisson overlap correction
        realized_vf = 1.0 - np.exp(-nominal_vf)
    else:
        realized_vf = nominal_vf

    return RVEGeometry(
        dim=2,
        domain_size=domain_size,
        inclusions=inclusions,
        target_volume_fraction=target_volume_fraction,
        realized_volume_fraction=realized_vf,
        seed=seed,
        metadata={} if metadata is None else dict(metadata),
    )


def _generate_mono_circle_2d(
    volume_fraction: float,
    num_particles: int,
    clearance: float,
    *,
    domain_size: float | tuple[float, float] | list[float] | np.ndarray = 1.0,
    max_restarts: int = 100,
    attempts_per_restart: int = 10000,
    seed: int | None = None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    allow_overlap: bool = False,
    verbosity: int = 0,
) -> RVEGeometry:
    domain = as_domain_size(2, domain_size)
    radius = monodisperse_circle_radius(volume_fraction, num_particles, domain)
    fixed_axes = np.array([radius, radius], dtype=float)

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
                if _overlaps_existing_periodic(center, fixed_axes, "circle", centers, axes_list, shapes, domain, clearance):
                    continue
                centers.append(center)
                axes_list.append(fixed_axes.copy())
                shapes.append("circle")
                accepted = True
                break
            if not accepted:
                success = False
                break

        if success:
            return _build_geometry_2d(
                centers,
                axes_list,
                shapes,
                domain,
                volume_fraction,
                seed,
                interphase_thickness_ratio,
                interphase_phase_id,
                metadata={
                    "generator": "generate_mono_circle_2d",
                    "radius": radius,
                    "clearance": clearance,
                },
            )

    raise RuntimeError("Failed to generate monodisperse circle geometry.")


def _generate_poly_circle_2d(
    volume_fraction: float,
    volume_fraction_tolerance: float,
    min_radius: float,
    max_radius: float,
    clearance: float,
    *,
    domain_size: float | tuple[float, float] | list[float] | np.ndarray = 1.0,
    max_restarts: int = 100,
    attempts_per_restart: int = 100000,
    seed: int | None = None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    allow_overlap: bool = False,
) -> RVEGeometry:
    if min_radius <= 0.0 or max_radius <= 0.0 or min_radius > max_radius:
        raise ValueError("Invalid radius range.")

    domain = as_domain_size(2, domain_size)
    rng = np.random.default_rng(seed)

    lower_target = max(0.0, volume_fraction - volume_fraction_tolerance)
    upper_target = min(1.0, volume_fraction + volume_fraction_tolerance)
    domain_area = float(np.prod(domain))

    for restart in range(max_restarts):
        centers = []
        axes_list = []
        shapes = []
        
        if allow_overlap:
            # For randomly overlapping inclusions, true volume fraction follows:
            #   VF_actual = 1 - exp(-VF_nominal)
            # So to target VF_actual, we need VF_nominal = -ln(1 - VF_actual)
            nominal_area = 0.0
            nominal_lower = -np.log(1.0 - lower_target) if lower_target < 1.0 else np.inf
            nominal_upper = -np.log(1.0 - upper_target) if upper_target < 1.0 else np.inf
        else:
            current_area = 0.0

        for _ in range(attempts_per_restart):
            # Check termination condition
            if allow_overlap:
                # Convert nominal area to actual VF estimate
                current_vf = 1.0 - np.exp(-nominal_area / domain_area)
                if current_vf >= lower_target:
                    return _build_geometry_2d(
                        centers,
                        axes_list,
                        shapes,
                        domain,
                        volume_fraction,
                        seed,
                        interphase_thickness_ratio,
                        interphase_phase_id,
                        metadata={
                            "generator": "generate_poly_circle_2d",
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
                current_vf = current_area / domain_area
                if current_vf >= lower_target:
                    return _build_geometry_2d(
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
                            "generator": "generate_poly_circle_2d",
                            "min_radius": min_radius,
                            "max_radius": max_radius,
                            "clearance": clearance,
                        },
                    )

            r = float(rng.uniform(min_radius, max_radius))
            axes = np.array([r, r], dtype=float)
            center = sample_random_center(rng, domain)

            if not _is_mesh_valid_primitive(center, axes, clearance, domain):
                continue
            if _overlaps_existing_periodic(
                center, axes, "circle", centers, axes_list, shapes, 
                domain, clearance, allow_overlap=allow_overlap
            ):
                continue

            inc_area = circle_area(r)
            
            if allow_overlap:
                new_nominal = nominal_area + inc_area
                # new_vf_actual = 1.0 - np.exp(-new_nominal / domain_area)
                # if new_vf_actual > upper_target:
                #     continue
                nominal_area = new_nominal
            else:
                new_area = current_area + inc_area
                new_vf = new_area / domain_area
                if new_vf > upper_target:
                    continue
                current_area = new_area

            centers.append(center)
            axes_list.append(axes)
            shapes.append("circle")

    raise RuntimeError(
        f"Failed to generate polydisperse circle geometry. "
        f"Target VF: {volume_fraction:.3f}, "
        f"Current VF: {current_vf:.3f}"
    )


def _generate_mono_ellipse_2d(
    volume_fraction: float,
    num_particles: int,
    axis_ratios,
    clearance: float,
    *,
    domain_size: float | tuple[float, float] | list[float] | np.ndarray = 1.0,
    max_restarts: int = 100,
    attempts_per_restart: int = 10000,
    seed: int | None = None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    verbosity: int = 0,
) -> RVEGeometry:
    domain = as_domain_size(2, domain_size)
    ratios = normalize_axis_ratios(axis_ratios)
    if ratios.size != 2:
        raise ValueError("axis_ratios must have length 2 for ellipses.")

    fixed_axes = monodisperse_ellipse_axes(volume_fraction, num_particles, domain, ratios)
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
                if _overlaps_existing_periodic(center, fixed_axes, "ellipse", centers, axes_list, shapes, domain, clearance):
                    continue
                centers.append(center)
                axes_list.append(fixed_axes.copy())
                shapes.append("ellipse")
                accepted = True
                break
            if not accepted:
                success = False
                break

        if success:
            return _build_geometry_2d(
                centers,
                axes_list,
                shapes,
                domain,
                volume_fraction,
                seed,
                interphase_thickness_ratio,
                interphase_phase_id,
                metadata={
                    "generator": "generate_mono_ellipse_2d",
                    "axis_ratios": ratios.tolist(),
                    "semi_axes": fixed_axes.tolist(),
                    "clearance": clearance,
                    "orientation_mode": "axis_aligned_only",
                },
            )

    raise RuntimeError("Failed to generate monodisperse ellipse geometry.")


def _generate_poly_ellipse_2d(
    volume_fraction: float,
    volume_fraction_tolerance: float,
    axis_ratios,
    min_scale: float,
    max_scale: float,
    clearance: float,
    *,
    domain_size: float | tuple[float, float] | list[float] | np.ndarray = 1.0,
    max_restarts: int = 100,
    attempts_per_restart: int = 100000,
    seed: int | None = None,
    interphase_thickness_ratio: float = 0.0,
    interphase_phase_id: int | None = None,
    verbosity: int = 0,
    allow_overlap: bool = False,
) -> RVEGeometry:
    if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
        raise ValueError("Invalid scale range.")

    domain = as_domain_size(2, domain_size)
    ratios = normalize_axis_ratios(axis_ratios)
    if ratios.size != 2:
        raise ValueError("axis_ratios must have length 2 for ellipses.")

    rng = np.random.default_rng(seed)

    lower_target = max(0.0, volume_fraction - volume_fraction_tolerance)
    upper_target = min(1.0, volume_fraction + volume_fraction_tolerance)
    domain_area = float(np.prod(domain))

    for restart in range(max_restarts):
        centers = []
        axes_list = []
        shapes = []
        
        if allow_overlap:
            # Same analytical correction for overlapping ellipses
            nominal_area = 0.0
        else:
            current_area = 0.0

        for _ in range(attempts_per_restart):
            # Check termination condition
            if allow_overlap:
                current_vf = 1.0 - np.exp(-nominal_area / domain_area)
                if current_vf >= lower_target:
                    return _build_geometry_2d(
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
                            "generator": "generate_poly_ellipse_2d",
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
                current_vf = current_area / domain_area
                if current_vf >= lower_target:
                    return _build_geometry_2d(
                        centers,
                        axes_list,
                        shapes,
                        domain,
                        volume_fraction,
                        seed,
                        interphase_thickness_ratio,
                        interphase_phase_id,
                        metadata={
                            "generator": "generate_poly_ellipse_2d",
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
                center, axes, "ellipse", centers, axes_list, shapes, 
                domain, clearance, allow_overlap=allow_overlap
            ):
                continue

            inc_area = ellipse_area(axes)
            
            if allow_overlap:
                new_nominal = nominal_area + inc_area
                new_vf_actual = 1.0 - np.exp(-new_nominal / domain_area)
                if new_vf_actual > upper_target:
                    continue
                nominal_area = new_nominal
            else:
                new_area = current_area + inc_area
                new_vf = new_area / domain_area
                if new_vf > upper_target:
                    continue
                current_area = new_area

            centers.append(center)
            axes_list.append(axes)
            shapes.append("ellipse")

    raise RuntimeError(
        f"Failed to generate polydisperse ellipse geometry. "
        f"Target VF: {volume_fraction:.3f}, "
        f"Current VF: {current_vf:.3f}"
    )


def _generate_mono_2d(
    input_data: GeometryInput
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

    if shape == "circle":
        return _generate_mono_circle_2d(
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
    if shape == "ellipse":
        return _generate_mono_ellipse_2d(
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
    raise ValueError(f"Unsupported 2D shape: {shape}")


def _generate_poly_2d(
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
    allow_overlap = input_data.allow_overlap

    if shape == "circle":
        if min_radius is None or max_radius is None:
            raise ValueError("min_radius and max_radius are required for circles.")
        return _generate_poly_circle_2d(
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
            allow_overlap=allow_overlap,
            # verbosity=verbosity,
        )
    if shape == "ellipse":
        if min_scale is None or max_scale is None:
            raise ValueError("min_scale and max_scale are required for ellipses.")
        return _generate_poly_ellipse_2d(
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
            allow_overlap=allow_overlap,
            # verbosity=verbosity,
        )
    raise ValueError(f"Unsupported 2D shape: {shape}")


__all__ = [
    
]




