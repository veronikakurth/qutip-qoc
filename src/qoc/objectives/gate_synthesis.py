from qutip import Qobj

from qoc.performance.base import PerformanceMeasure
from qoc.performance.gate_fidelity import is_unitary, GateFidelity
from .base import Objective


class GateSynthesis(Objective):
    """
    Objective for finding controls that implement a target unitary gate.

    Both initial (typically identity) and target must be unitary operators
    with matching dimensions.
    """

    def __init__(
        self,
        initial: Qobj,
        target: Qobj,
        performance_measure: None | PerformanceMeasure = None
    ):
        if performance_measure is None:
            performance_measure = GateFidelity()
        _validate_operator(initial, "initial")
        _validate_operator(target, "target")
        if initial.dims != target.dims:
            raise ValueError(
                f"initial and target dimensions do not match: "
                f"{initial.dims} vs {target.dims}"
            )
        super().__init__(initial, target, performance_measure)


def _validate_operator(obj: Qobj, name: str) -> None:
    if not isinstance(obj, Qobj):
        raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
    if not obj.isoper:
        raise ValueError(f"{name} must be an operator, got type '{obj.type}'")
    if not is_unitary(obj):
        raise ValueError(f"{name} must be a unitary operator")
