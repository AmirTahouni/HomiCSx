"""Deterministic generated and prescribed geometry examples."""

from __future__ import annotations

import json
import math

from homicsx import GeometryInput, Inclusion, RVEGeometry, particulate_geometry_generator


def run_example() -> dict:
    generated = particulate_geometry_generator(
        GeometryInput(
            dim=2,
            dispersion="mono",
            shape="circle",
            volume_fraction=0.12,
            num_particles=4,
            clearance=0.02,
            domain_size=(1.0, 1.0),
            seed=42,
        )
    )
    rotated_ellipse = Inclusion(
        center=(0.5, 0.5),
        phase_id=1,
        shape="ellipse",
        radii=(0.22, 0.08),
        orientation=math.pi / 6.0,
    )
    prescribed = RVEGeometry(
        dim=2,
        domain_size=(1.0, 1.0),
        inclusions=[rotated_ellipse],
        metadata={"example": "geometry_generation"},
    )
    return {
        "generated_original_inclusions": sum(
            not inclusion.is_periodic_image() for inclusion in generated.inclusions
        ),
        "generated_total_inclusions": len(generated.inclusions),
        "generated_phase_ids": list(generated.phase_ids),
        "prescribed_shape": prescribed.inclusions[0].shape,
        "prescribed_orientation_radians": prescribed.inclusions[0].orientation,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
