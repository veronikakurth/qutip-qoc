import numpy as np
from qutip import Qobj, QobjEvo, liouvillian

from .base import System

from qutip.typing import QobjEvoLike

class OpenSystem(System):
    """
    Open quantum system described by Lindbladian dynamics.

    H(t) = H0 + sum_k u_k(t) * H_k, with Lindblad dissipators c_ops.
    """
    # TODO: allow user to pass Liouvillian as in mesolve. There, H can be Liouvillian
    # TODO: for smooth integration with mesolve:  
    def __init__(self, H0: Qobj, H_controls: list[Qobj], c_ops: QobjEvoLike | list[QobjEvoLike]):
        """
        Parameters
        ----------
        H0 : Qobj
            Drift Hamiltonian.
        H_controls : list[Qobj]
            Control Hamiltonians.
        c_ops: QobjEvoLike | list[QobjEvoLike]
            Collapse operators defining dissipation into the environment.
        """
        super().__init__(H0, H_controls)
        self._validate_c_ops(c_ops)
        self._c_ops = c_ops or []

    @property
    def c_ops(self):
        return self._c_ops

    # The output here can be a full Hamiltonian or Liouvillian - see how the latter is assembled in MESolver
    def build_generator(self, control_amplitudes: np.ndarray) -> list:
        H = [self._H0]
        for k, H_k in enumerate(self._H_controls):
            H.append([H_k, control_amplitudes[k]])
        return H
    
    # Currently, only used in GRAPE
    # TODO: Would it generally work with other parameterisations? And is it needed?
    def build_generator_time_j(self, control_amplitudes: np.ndarray, j: int) -> Qobj:
        H = self._H0 + sum(control_amplitudes[k][j] * self.H_controls[k] for k in range(self.n_controls))
        L_j = liouvillian(H, self._c_ops)
        return L_j

    def _validate_c_ops(self, c_ops):
        c_ops = c_ops or []
        c_ops = [c_ops] if isinstance(c_ops, (Qobj, QobjEvo)) else c_ops
        for c_op in c_ops:
            if not isinstance(c_op, (Qobj, QobjEvo)):
                raise TypeError("All `c_ops` must be a Qobj or QobjEvo")
