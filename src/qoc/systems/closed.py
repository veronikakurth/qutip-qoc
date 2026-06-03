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
        _validate_hamiltonians(H0, H_controls)
        self.H0 = H0
        self.H_controls = H_controls

    @property
    def n_controls(self) -> int:
        return len(self.H_controls)

    @property
    def dims(self) -> int: # TODO: it's another type from qutip.
        return self.H0.dims

    @property
    def shape(self):
        return self.H0.shape

    # TODO: Would it be better to use Coefficient class for expressing time dependency in coefficients
    def build_hamiltonian(
        self, control_amplitudes: np.ndarray
    ) -> list:
        """Build the QuTiP time-dependent Hamiltonian list."""
        H = [self.H0]
        for k, H_k in enumerate(self.H_controls):
            H.append([H_k, control_amplitudes[k]])
        return H

    def build_hamiltonian_time_j(self, control_amplitudes: np.ndarray, j: int) -> Qobj:
        H = self.H0 + sum(control_amplitudes[k][j] * self.H_controls[k] for k in range(self.n_controls))
        return H



def _validate_hamiltonians(H0: Qobj, H_controls: list) -> None: # TODO: I believe there may be more properties we need to check in order to propagate the system. Or are these done by the solver?
    if not isinstance(H0, Qobj) or not H0.isoper:
        raise TypeError("H0 must be a square operator Qobj")

    for k, H_k in enumerate(H_controls):
        if not isinstance(H_k, Qobj) or not H_k.isoper:
            raise TypeError(f"H_controls[{k}] must be a square operator Qobj")
        if H_k.dims != H0.dims:
            raise ValueError(
                f"H_controls[{k}] has dims {H_k.dims}, expected {H0.dims}"
            )
