from qutip import Qobj

from .base import FigureOfMerit


class GateInfidelity(FigureOfMerit):
    """
    Infidelity measure for unitary gate synthesis.

    Uses the process infidelity from Khaneja et al. (2005):
        infidelity = 1 - |Tr[U†V]|² / N²

    where U is the target unitary, V is the propagated unitary, and N is
    the Hilbert space dimension.

    This normalisation ensures the fidelity is 1 when U = V (up to global phase)
    and 0 when they are orthogonal.

    Returns a value in [0, 1]. 0 = perfect gate.
    """

    def compute(self, propagated: Qobj, target: Qobj) -> float:
        N = target.shape[0]
        overlap = (target.dag() * propagated).tr()
        return 1.0 - abs(overlap) ** 2 / N ** 2
