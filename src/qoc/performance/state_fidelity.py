from qutip import Qobj, fidelity

from .base import PerformanceMeasure
from ..utils.objective_helpers import state_type


# TODO: check if tackles open system. For this, check where and how used exactly
# Answer: in the optimisation procedure, PerformanceMeasure.loss function is used
# It's possible to compute fidelity for open system if qutip.fidelity supports it
class StateFidelity(PerformanceMeasure):
    """State fidelity: 1 = perfect match, 0 = orthogonal."""

    def compute(self, current: Qobj, target: Qobj) -> float:
        _validate_states(current, target)
        return fidelity(current, target)


def is_density_matrix(obj: Qobj, tol: float = 1e-10) -> bool:
    return obj.isoper and obj.isherm


def _validate_states(current: Qobj, target: Qobj) -> None:
    current_type, target_type = state_type(current), state_type(target)

    if current_type != target_type:
        raise ValueError(
            f"initial and target must be of the same type, got {current_type} and {target_type}"
        )
    if initial.dims != target.dims:
        raise ValueError(
            f"initial and target dims differ: {initial.dims} vs {target.dims}"
        )
