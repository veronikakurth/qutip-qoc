from qutip import Qobj, fidelity
import numpy as np
 

from .base import PerformanceMeasure


class StateFidelity(PerformanceMeasure):
    """State fidelity: 1 = perfect match, 0 = orthogonal."""

    def compute(self, current: Qobj, target: Qobj) -> float:
        if current.type != "operator-ket":
            overlap = target.dag() @ current
            return float(np.abs(complex(overlap)) ** 2)
        """Non-normalized Hilbert-Schmidt overlap."""
        diff = current - target
        overlap = Qobj(diff.dag() * diff).tr()
        return np.real(overlap / target.norm())

    def gradient(self, current, target):
        if current.type != "operator-ket":
            # f = |<t|psi>|**2 -> df = 2Re(conj(c) <t|dpsi>), c = <t|psi>
            c = complex(target.dag() @ current)
            return 2 * c * target
        return target # f = Re<t|rho> (linear) -> df = Re<t|drho>
