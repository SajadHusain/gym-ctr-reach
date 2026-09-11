"""Experimental equilibrium mechanics; importing does not change CTR-Reach-v1."""
from .geometry import TubeParameters, JointConstraints, segment_tubes
from .solver import Equilibrium, EquilibriumError, EquilibriumSolver, SolverOptions
from .sensitivity import PaperJacobian, Sensitivity, SensitivityError, equilibrium_sensitivity

# Archived experiments retain their public imports without loading branch
# tracking, elastic analysis or controllers in the joint-only training path.
_LEGACY = {**{name: "stability" for name in ("StabilityDiagnostic", "elastic_stability")},
           **{name: "tracking" for name in ("BranchTracker", "TrackingOptions", "TrackingStep", "TrackedState",
                                            "BranchInitializationError", "GoalProgress")},
           **{name: "control" for name in ("GoalController", "ControlOptions", "ControlStep")}}


def __getattr__(name):
    if name not in _LEGACY:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(f".{_LEGACY[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = ["TubeParameters", "JointConstraints", "segment_tubes", "Equilibrium",
           "EquilibriumError", "EquilibriumSolver", "SolverOptions", "Sensitivity", "PaperJacobian",
           "SensitivityError", "equilibrium_sensitivity", "StabilityDiagnostic", "elastic_stability",
           "BranchTracker", "TrackingOptions", "TrackingStep", "TrackedState", "BranchInitializationError", "GoalProgress",
           "GoalController", "ControlOptions", "ControlStep"]
