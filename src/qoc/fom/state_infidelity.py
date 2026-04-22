import qutip
from qutip import Qobj

from .base import FigureOfMerit


class StateInfidelity(FigureOfMerit):
    """
    Infidelity measure for state-to-state transfer.

    For kets (pure states, closed systems):
        infidelity = 1 - |<target|propagated>|²

    For density matrices (mixed states, open systems):
        infidelity = 1 - F(propagated, target)²
        where F is the Uhlmann fidelity, computed via QuTiP.

    Returns a value in [0, 1]. 0 = perfect transfer.
    """

    def compute(self, propagated: Qobj, target: Qobj) -> float:
        if propagated.isket:
            # In QuTiP 5, bra * ket returns a complex scalar directly
            overlap = target.dag() * propagated
            return 1.0 - abs(overlap) ** 2
        else:
            # Uhlmann fidelity for density matrices: Tr[sqrt(sqrt(rho) sigma sqrt(rho))]
            # qutip.fidelity returns F (not F²), so we square it to stay consistent
            # with the pure-state convention above.
            return 1.0 - qutip.fidelity(propagated, target) ** 2
