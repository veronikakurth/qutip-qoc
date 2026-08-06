from qutip import Qobj

from qoc.performance.base import PerformanceMeasure


class Objective:
    """
    Defines what the optimization is trying to achieve.

    Holds the initial condition, target, and the measure used to score how close
    the achieved result is to the target. compute() delegates to performance_measure
    by default; subclasses may override it for completely custom scoring logic.

    Subclasses (StateTransfer, GateSynthesis) add domain-specific validation
    of initial and target in their constructors.
    """

    def __init__(
        self,
        initial: Qobj,
        target: Qobj,
        performance_measure: PerformanceMeasure, # ~ figure of merit
    ):
        if not isinstance(performance_measure, PerformanceMeasure):
            raise TypeError(
                f"performance_measure must be a PerformanceMeasure, "
                f"got {type(performance_measure)}"
            )
        self.initial = initial
        self.target = target
        self.performance_measure = performance_measure
    # TODO: is worth renaming to reflect what is exactly computed
    def compute(self, current: Qobj) -> float:
        return self.performance_measure.compute(current, self.target)

    def loss(self, current: Qobj) -> float:
        return 1.0 - self.compute(current)

    def check_compatible(self, system) -> None:
        """Validate this objective against the system it's paired with"""
        if self.initial.dims[0] != system.dims[0]:
            raise ValueError(
                f"objective acts on Hilbert space {self.initial.dims[0]}, "
                f"but system is defined on {system.dims[0]}"
            )
