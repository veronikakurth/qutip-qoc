from qutip import Qobj

from qoc.performance.base import PerformanceMeasure
from qoc.performance.state_fidelity import StateFidelity
from qoc.utils.objective_helpers import validate_states, state_type
from .base import Objective


class StateTransfer(Objective):
    """
    Objective for driving an initial state to a target state.

    Both initial and target must be kets (pure states) or density matrices (mixed states)
    with matching dimensions.
    If a performance measure is not selected, StateFidelity is chosen by default.
    """

    def __init__(
        self,
        initial: Qobj,
        target: Qobj,
        performance_measure: None | PerformanceMeasure = None,
    ):
        if performance_measure is None:
            performance_measure = StateFidelity()
        else:
            if not isinstance(performance_measure, PerformanceMeasure):
                raise TypeError(
                    f"Expected PerformanceMeasure as type of performance_measure, got={type(performance_measure)}"
                )
        validate_states(initial, target)
        super().__init__(initial, target, performance_measure)

    def check_compatible(self, system) -> None:
        super().check_compatible(system)
        type_ = state_type(self.initial)
        if type_ != system.state_type:
            raise ValueError(
                f"System expects {system.state_type} states, but objective provides {type_}"
            )
