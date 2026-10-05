"""MultiEcho: a counterfactual framework for experiments on learned worlds.

Everything is communication. Worlds (physical, simulation, learned) receive, evolve
and emit artifacts. Interfaces (masking, clipping, bridges, translators) carry
artifacts between them. An experiment is a chain of worlds that ends at an analyst,
who observes under an explicit scope and fits forward and backward laws.

Engines, models, translators and analysis methods are plugins. `EXPERIMENTAL` lists
symbols outside the protocol.
"""

from ._version import __version__ as __version__
from .analysis import (
    BackwardLaw,
    BackwardMethod,
    ContextView,
    ForwardLaw,
    ForwardMethod,
    Labeled,
    ResponseSet,
    Task,
)
from .artifacts import Artifact, Channel, Modality, Region, Scope, Statement
from .chain import CONTEXT_CHANNEL, Chain, ReferenceRoute
from .communication import (
    Alignment,
    CommunicationReport,
    CommunicationRequest,
    CommunicationResult,
    CommunicationRunner,
    TranslationReport,
    evaluate_communication,
    evaluate_translation,
)
from .contracts import (
    AccessMode,
    Axis,
    Context,
    DataRole,
    Intervention,
    ObservationSpec,
    PairRequest,
    SeedMode,
    Space,
    StimulusSpec,
    WorldSpec,
)
from .datasets import PairDataset, require_disjoint
from .evaluation import (
    BranchPrediction,
    Comparison,
    DistributionReport,
    EvaluationPlan,
    EvaluationReport,
    EvaluationTarget,
    EvidenceStatus,
    energy_score,
    evaluate_distributions,
    evaluate_predictions,
)
from .execution import (
    PairResult,
    PairRunner,
    SequenceRequest,
    SequenceRunner,
    TraceResult,
    WorldAdapter,
)
from .experiment import Experiment, dumps_report, loads_report
from .interfaces import (
    ComposedInterface,
    Interface,
    InterfaceSpec,
    ParameterBridge,
    ParameterMap,
    ScopeInterface,
)
from .keyframes import KeyframeAlignment, KeyframeGallery, load_gallery, save_gallery
from .laws import (
    LawFitter,
    LawQuery,
    LawSpec,
    LawTestPlan,
    LawTestReport,
    PathLaw,
    PathPair,
    ProposedQuery,
    ResidualLaw,
    ResponseLaw,
    TrajectoryLaw,
    fit_response_law,
    predict_responses,
    propose_disagreement,
    test_path_law,
    test_regimes,
    test_residual_law,
    test_trajectory_law,
)
from .masking import (
    EnergyMask,
    Grouping,
    ManualMask,
    MaskedScope,
    Masker,
    MaskerCard,
    MaskInput,
    MonotonicSNRMask,
    NoMask,
    RepeatNoiseSNRMask,
    Sweep,
)
from .protocol import (
    PROTOCOL_VERSION,
    Claim,
    ConditionDimension,
    Declaration,
    Outcome,
    adjudicate,
)
from .serialization import (
    dumps_artifact,
    dumps_dataset,
    dumps_trace,
    loads_artifact,
    loads_dataset,
    loads_trace,
)
from .study import StudyLock, compare_declarations, compare_reports, dumps_study, loads_study
from .training import (
    TrainingBatch,
    TrainingConsumer,
    TrainingTarget,
    branch_energy_loss,
    export_training,
    response_mse,
)
from .worlds import (
    LearnedContract,
    PhysicalScene,
    World,
    WorldCard,
    WorldKind,
    scene_context,
    scene_statements,
)

__protocol_version__ = PROTOCOL_VERSION
EXPERIMENTAL = (
    "TrainingBatch", "TrainingConsumer", "TrainingTarget", "branch_energy_loss", "export_training",
    "response_mse",
)

__all__ = [
    "EXPERIMENTAL", "PROTOCOL_VERSION",
    # communication architecture
    "Artifact", "Channel", "Modality", "Region", "Scope", "Statement",
    "World", "WorldCard", "WorldKind", "LearnedContract", "PhysicalScene", "scene_context", "scene_statements",
    "Interface", "InterfaceSpec", "ScopeInterface", "ParameterBridge", "ParameterMap", "ComposedInterface",
    "Chain", "ReferenceRoute", "CONTEXT_CHANNEL", "Alignment", "TranslationReport", "evaluate_translation",
    "CommunicationRequest", "CommunicationResult", "CommunicationRunner", "CommunicationReport",
    "evaluate_communication",
    # experiments
    "AccessMode", "Axis", "Context", "DataRole", "Intervention", "ObservationSpec", "PairRequest", "SeedMode",
    "Space", "StimulusSpec", "WorldSpec", "WorldAdapter", "PairRunner", "PairResult", "SequenceRunner", "SequenceRequest",
    "TraceResult", "PairDataset", "require_disjoint", "Declaration", "ConditionDimension", "Experiment",
    "Outcome", "Claim", "adjudicate", "dumps_report", "loads_report",
    # studies and keyframe alignment
    "StudyLock", "compare_declarations", "compare_reports", "dumps_study", "loads_study",
    "KeyframeAlignment", "KeyframeGallery", "save_gallery", "load_gallery",
    # observation masking
    "Grouping", "Sweep", "MaskInput", "MaskerCard", "Masker", "MaskedScope", "NoMask", "ManualMask",
    "MonotonicSNRMask", "RepeatNoiseSNRMask", "EnergyMask",
    # analysis
    "Task", "Labeled", "ContextView", "ResponseSet", "ForwardLaw", "BackwardLaw", "ForwardMethod",
    "BackwardMethod", "EvaluationPlan", "EvaluationReport", "EvaluationTarget", "EvidenceStatus", "Comparison",
    "evaluate_predictions", "BranchPrediction", "DistributionReport", "energy_score", "evaluate_distributions",
    # laws on trajectories
    "LawSpec", "LawQuery", "LawFitter", "ResponseLaw", "TrajectoryLaw", "ResidualLaw", "PathLaw", "PathPair",
    "LawTestPlan", "LawTestReport", "ProposedQuery", "fit_response_law", "predict_responses",
    "propose_disagreement", "test_path_law", "test_regimes", "test_residual_law", "test_trajectory_law",
    # exchange
    "dumps_artifact", "loads_artifact", "dumps_dataset", "loads_dataset", "dumps_trace", "loads_trace",
    # experimental
    "TrainingBatch", "TrainingConsumer", "TrainingTarget", "branch_energy_loss", "export_training",
    "response_mse",
]
