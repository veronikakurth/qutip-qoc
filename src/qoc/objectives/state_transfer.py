from qutip import Qobj

from qoc.performance.base import PerformanceMeasure
from qoc.performance.state_fidelity import state_measure_for
from qoc.utils.objective_helpers import validate_states, state_type
from .base import Objective


class StateTransfer(Objective):
    """
    Objective for driving an initial state to a target state.

    Both initial and target must be kets (pure states) or density matrices (mixed states)
    with matching dimensions.
    If a performance measure is not selected, the default formula
    matching the state representation is chosen: KetStateFidelity for kets,
    SquaredHSDistance for density matrices (half the squared Hilbert-Schmidt
    distance as the loss; the reported fidelity is its complementary score).
    """

    propagates = "state"

    def __init__(
        self,
        initial: Qobj,
        target: Qobj, # TODO: a target may become optional -> for certain objectives, it might not be necessary to provide explicit one. Update this contract in base class
        performance_measure: None | PerformanceMeasure = None,
    ):
        validate_states(initial, target)
        if performance_measure is None:
            performance_measure = state_measure_for(state_type(initial))
        else:
            if not isinstance(performance_measure, PerformanceMeasure):
                raise TypeError(
                    f"Expected PerformanceMeasure as type of performance_measure, got={type(performance_measure)}"
                )
        super().__init__(initial, target, performance_measure)

    def check_compatible(self, system) -> None:
        super().check_compatible(system)
        type_ = state_type(self.initial)
        if type_ != system.state_type:
            raise ValueError(
                f"System expects {system.state_type} states, but objective provides {type_}"
            )
