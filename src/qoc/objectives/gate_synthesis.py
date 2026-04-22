from qutip import Qobj

from .base import Objective
from qoc.fom.base import FigureOfMerit


class GateSynthesis(Objective):
    """
    Objective for unitary gate synthesis.

    Both initial and target must be square operators with matching dimensions.
    The initial operator is typically the identity; the target is the desired gate.
    """

    def __init__(self, initial: Qobj, figure_of_merit: FigureOfMerit, target: Qobj = None):
        if not target:
            target = self._infer_target_from_figure_of_merit(figure_of_merit)
        _validate_operators(initial, target)
        super().__init__(initial, target, infidelity)


def is_unitary(obj, tol=1e-10):
    I = qeye(obj.shape[0])
    return (obj.dag() * obj - I).norm() < tol

def _validate_operators(initial: Qobj, target: Qobj) -> None:
    for name, obj in [("initial", initial), ("target", target)]:
        if not isinstance(obj, Qobj):
            raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
        if not obj.isoper:
            raise ValueError(f"{name} must be an operator (square matrix), got type {obj.type}")
        if not is_unitary(obj):
            raise ValueError(f"Operator must be unitary")

    if initial.dims != target.dims:
        raise ValueError(
            f"initial and target must have the same dimensions, "
            f"got {initial.dims} and {target.dims}"
        )
