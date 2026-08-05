from qutip import Qobj, fidelity

from .base import PerformanceMeasure
from ..utils.objective_helpers import validate_states


# TODO: check if tackles open system. For this, check where and how used exactly
# Answer: in the optimisation procedure, PerformanceMeasure.loss function is used
# It's possible to compute fidelity for open system if qutip.fidelity supports it
class StateFidelity(PerformanceMeasure):
    """State fidelity: 1 = perfect match, 0 = orthogonal."""

    def compute(self, current: Qobj, target: Qobj) -> float:
        return fidelity(current, target)
