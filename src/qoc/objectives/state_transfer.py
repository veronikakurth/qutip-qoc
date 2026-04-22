from qutip import Qobj

from .base import Objective
from qoc.fom.base import FigureOfMerit

class StateTransfer(Objective):
    """
    Objective for state-to-state transfer.

    Both initial and target must be quantum states: either kets (pure,
    closed system) or density matrices (mixed, open system). They must
    be the same type and have matching dimensions.
    """

    def __init__(self, initial: Qobj, figure_of_merit: FigureOfMerit, target: Qobj = None):
        # If target is not provided, it must be inferrable from the figure of merit
        if not target:
            target = self._infer_target_from_figure_of_merit(figure_of_merit)
        _validate_states(initial, target)
        super().__init__(initial, figure_of_merit, target)


def is_density_matrix(obj, tol=1e-10):
    return (
        obj.isherm and
        abs(obj.tr() - 1) < tol and
        min(obj.eigenenergies()) >= -tol
    )

def _validate_states(initial: Qobj, target: Qobj) -> None:
    for name, obj in [("initial", initial), ("target", target)]:
        if not isinstance(obj, Qobj):
            raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
        if not obj.isket:
            if not obj.isoper or not is_density_matrix(obj):
                raise ValueError(f"{name} must be a ket or density matrix")

    if initial.isket != target.isket:
        raise ValueError("initial and target must both be kets or both be density matrices")

    if initial.dims != target.dims:
        raise ValueError(
            f"initial and target must have the same dimensions, "
            f"got {initial.dims} and {target.dims}"
        )
