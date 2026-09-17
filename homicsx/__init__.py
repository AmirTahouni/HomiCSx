from .core import (
    Inclusion, 
    # PeriodicityInfo,
    GeometryInput, 
    RVEGeometry,

    PhysicalTags,
    MeshSettings,

    LinearElasticIsotropic,
    NonlinearMaterialModel,
    HyperelasticMaterial,
    NeoHookeanIsotropic,
    ViscoelasticGeneralizedMaxwell,
    MaterialAssignment, 

    ProblemSettings,

    LinearHomogenizationResult,
    NonlinearHomogenizationResult,
    AdaptiveSettings,
    SimulationState,
    PreLoadCaseData,
    PreStepData,
    PostConvergenceData,
    PostStressData,
    PostTangentData,
    PostLoadCaseData,
    StepFailureData,

    EnsembleStatSummary,
    EnsembleStudyResult,
)

from .geometry import (
    particulate_geometry_generator,
)

from .mesh import (
    generate_mesh,
)

from .homogenization import (
    LinearHomogenizationDriver,
    NonlinearHomogenizationDriver,
)

from .stochastic import (
    perform_ensemble_study,
    sweep_volume_fraction_linear,
    sweep_stiffness_contrast_linear,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # core
    "Inclusion", 
    "GeometryInput", 
    "RVEGeometry",
    "PhysicalTags",
    "MeshSettings",
    "LinearElasticIsotropic",
    "NonlinearMaterialModel",
    "HyperelasticMaterial",
    "NeoHookeanIsotropic",
    "ViscoelasticGeneralizedMaxwell",
    "MaterialAssignment", 
    "ProblemSettings",
    "LinearHomogenizationResult",
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
    "EnsembleStatSummary",
    "EnsembleStudyResult",

    # geometry
    "particulate_geometry_generator",

    # mesh
    "generate_mesh",

    # homogenization
    "LinearHomogenizationDriver",
    "NonlinearHomogenizationDriver",

    # stochastic
    "perform_ensemble_study",
    "sweep_volume_fraction_linear",
    "sweep_stiffness_contrast_linear",
]

