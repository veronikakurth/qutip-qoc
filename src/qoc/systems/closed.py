import numpy as np
from qutip import Qobj

from .base import System


class ClosedSystem(System):
    """
    Closed quantum system described by a drift + control Hamiltonian.

    H(t) = H0 + sum_k u_k(t) * H_k

    where H0 is the drift (time-independent) and H_k are the control
    Hamiltonians scaled by the piecewise-constant amplitudes u_k(t).
    """

    def __init__(self, H0: Qobj, H_controls: list[Qobj]):
        """
        Parameters
        ----------
        H0 : Qobj
            Drift Hamiltonian.
        H_controls : list[Qobj]
            Control Hamiltonians.
        """
        super().__init__(H0, H_controls)

    # TODO: Would it be better to use Coefficient class for expressing time dependency in coefficients
    def build_generator(
        self, control_amplitudes: np.ndarray
    ) -> list:
        """Build the QuTiP time-dependent Hamiltonian list."""
        H = [self._H0]
        for k, H_k in enumerate(self._H_controls):
            H.append([H_k, control_amplitudes[k]])
        return H

    # Currently, only used in GRAPE
    def build_generator_time_j(self, control_amplitudes: np.ndarray, j: int) -> Qobj:
        H = self._H0 + sum(control_amplitudes[k][j] * self._H_controls[k] for k in range(self.n_controls))
        return H
