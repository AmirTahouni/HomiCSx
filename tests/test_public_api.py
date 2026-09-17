import pytest

import homicsx


SUPPORTED_NONLINEAR_API = {
    "NonlinearHomogenizationDriver",
    "NonlinearHomogenizationResult",
    "AdaptiveSettings",
    "SimulationState",
    "PreLoadCaseData",
    "PreStepData",
    "PostConvergenceData",
    "PostStressData",
    "PostTangentData",
    "PostLoadCaseData",
    "StepFailureData",
    "HyperelasticMaterial",
    "NeoHookeanIsotropic",
    "ViscoelasticGeneralizedMaxwell",
}


def test_supported_nonlinear_api_is_exported_from_top_level():
    assert SUPPORTED_NONLINEAR_API <= set(homicsx.__all__)
    for name in SUPPORTED_NONLINEAR_API:
        assert getattr(homicsx, name) is not None


def test_problem_settings_rejects_plane_stress_with_guidance():
    with pytest.raises(NotImplementedError, match="use 'plane_strain' or a 3D model"):
        homicsx.ProblemSettings(
            dim=2,
            two_dimensional_formulation="plane_stress",
        )


@pytest.mark.parametrize(
    ("dim", "formulation", "message"),
    [
        (1, None, "dim must be either 2 or 3"),
        (2, None, "must be 'plane_strain'"),
        (2, "axisymmetric", "must be 'plane_strain'"),
        (3, "plane_strain", "must be None for 3D"),
    ],
)
def test_problem_settings_rejects_invalid_dimension_formulation_pairs(
    dim,
    formulation,
    message,
):
    with pytest.raises(ValueError, match=message):
        homicsx.ProblemSettings(
            dim=dim,
            two_dimensional_formulation=formulation,
        )
