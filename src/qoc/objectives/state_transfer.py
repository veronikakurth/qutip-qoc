from qutip import Qobj

from qoc.performance.base import PerformanceMeasure
from qoc.performance.state_fidelity import is_density_matrix, StateFidelity
from .base import Objective


class StateTransfer(Objective):
    """
    Objective for driving an initial state to a target state.

    Both initial and target must be kets (pure states) or density matrices (mixed states)
    with matching dimensions.
    """

    def __init__(
        self,
        initial: Qobj,
        target: Qobj,
        performance_measure: PerformanceMeasure=StateFidelity()
    ):
        _validate_state(initial, "initial")
        _validate_state(target, "target")
        if initial.dims != target.dims:
            raise ValueError(
                f"initial and target dimensions do not match: "
                f"{initial.dims} vs {target.dims}"
            )
        super().__init__(initial, target, performance_measure)


def _validate_state(obj: Qobj, name: str) -> None:
    if not isinstance(obj, Qobj):
        raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
    if not obj.isket and not (obj.isoper and is_density_matrix(obj)):
        raise ValueError(f"{name} must be a ket or density matrix")
