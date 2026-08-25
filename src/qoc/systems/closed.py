import numpy as np
from qutip import Qobj

from .base import System, StateType, ClassVar


class ClosedSystem(System):
    """
    Closed quantum system described by a drift + control Hamiltonian.

    Hilbert space, dim n.

    H(t) = H0 + sum_k u_k(t) * H_k

    where H0 is the drift (time-independent) and H_k are the control
    Hamiltonians scaled by the piecewise-constant amplitudes u_k(t).
    """

    _GENERATOR_PREFACTOR = -1j # Schroedinger: d|psi>/dt = -i H |psi>

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
        self._motion_controls = [self._GENERATOR_PREFACTOR * Hk for Hk in H_controls]

    # System representation
    state_type: ClassVar[StateType] = "ket"

    def encode_state(self, state: Qobj) -> Qobj:
        return state # shape (n, 1) for a ket, (n, n) for a gate-synthesis operator
    # TODO: review encoding/decoding strategy for closed systems 
    def decode_state(self, state: Qobj) -> Qobj:
        space = self._H0.dims[0]
        target_dims = [space, [1] * len(space)]
        return Qobj(state, dims=target_dims)

    def decode_operator(self, operator: Qobj) -> Qobj:
        space = self._H0.dims[0]
        return Qobj(operator, dims=[space, space])

    def control_generators(self) -> list[Qobj]:
        return self._motion_controls

    # Is not used in GRAPE, may prove useful in other algorithms
    def build_generator(
        self, u: np.ndarray
    ) -> list:
        """Build the QuTiP time-dependent Hamiltonian list based on 
        drift and control Hamiltonians, as well as time-dependent control amplitudes."""
        H = [self._H0]
        for k, H_k in enumerate(self._H_controls):
            H.append([H_k, u[k]])
        return H

    def motion_generator_time_j(self, u: np.ndarray, j: int) -> Qobj:
        """Full Hamiltonian with a prefactor term and control amplitudes at time j. """
        H_j = self._H0 + sum(u[k][j] * self._H_controls[k] for k in range(self.n_controls))
        return self._GENERATOR_PREFACTOR * H_j
