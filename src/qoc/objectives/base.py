from qutip import Qobj

from qoc.performance.base import PerformanceMeasure


class Objective:
    """
    Defines what the optimization is trying to achieve.

    Holds the initial condition, target, and the measure used to score how close
    the achieved result is to the target.

    Everything delegates to ``performance_measure``, which owns both the loss
    and its gradient. ``loss`` is what an optimizer minimizes;
    ``fidelity`` is for reporting only. Each method defaults to this
    objective's own ``target`` when none is passed.

    Subclasses (StateTransfer, GateSynthesis) add domain-specific validation
    of initial and target in their constructors.
    """

    def __init__(
        self,
        initial: Qobj,
        target: Qobj,
        performance_measure: PerformanceMeasure, # ~ figure of merit/quality function
    ):
        if not isinstance(performance_measure, PerformanceMeasure):
            raise TypeError(
                f"performance_measure must be a PerformanceMeasure, "
                f"got {type(performance_measure)}"
            )
        self.initial = initial
        self.target = target
        self.performance_measure = performance_measure
    
    def loss(self, current: Qobj, target: Qobj = None) -> float:
        """Scalar the optimizer minimizes; 0.0 at a perfect match."""
        return self.performance_measure.loss(current, self._target_or_own(target))

    def loss_gradient(self, current: Qobj, target: Qobj = None) -> Qobj:
        """d(loss)/d(current), in the same encoding as ``current``."""
        return self.performance_measure.loss_gradient(
            current, self._target_or_own(target)
        )

    def fidelity(self, current: Qobj, target: Qobj = None) -> float:
        """Figure of merit for reporting; 1 = perfect match."""
        return self.performance_measure.fidelity(current, self._target_or_own(target))

    #: Retained so existing scripts keep working; `fidelity` is the name to use.
    compute = fidelity

    def _target_or_own(self, target: Qobj | None) -> Qobj:
        # `is None`, not truthiness: bool(Qobj) is always True, so `if target:`
        # only happened to behave correctly.
        return self.target if target is None else target

    def check_compatible(self, system) -> None:
        """Validate this objective against the system it's paired with"""
        if self.initial.dims[0] != system.dims[0]:
            raise ValueError(
                f"objective acts on space {self.initial.dims[0]}, "
                f"but system is defined on {system.dims[0]}"
            )
