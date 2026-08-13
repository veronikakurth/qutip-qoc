from qutip import Qobj, fidelity
import numpy as np
 

from .base import PerformanceMeasure


class StateFidelity(PerformanceMeasure):
    """State fidelity: 1 = perfect match, 0 = orthogonal."""

    def compute(self, current: Qobj, target: Qobj) -> float:
        if current.type != "operator-ket":
            overlap = target.dag() @ current
            return float(np.abs(overlap) ** 2)
        else:
            """Non-normalized Hilbert-Schmidt overlap."""
            return float(np.real(target.dag() @ current))
