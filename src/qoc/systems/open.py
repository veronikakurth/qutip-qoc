import numpy as np
from qutip import Qobj

from .base import System

class OpenSystem(System):
    """
    Open quantum system described by Lindbladian dynamics.

    H(t) = H0 + sum_k u_k(t) * H_k, with Lindblad dissipators c_ops.
    """

    def __init__(self, H0: Qobj, H_controls: list[Qobj], c_ops: list[Qobj]):
        _validate_hamiltonians(H0, H_controls)
        self.H0 = H0
        self.H_controls = H_controls
        self.c_ops = c_ops

    @property
    def n_controls(self) -> int:
        return len(self.H_controls)

    def build_hamiltonian(self, control_amplitudes: np.ndarray) -> list:
        H = [self.H0]
        for k, H_k in enumerate(self.H_controls):
            H.append([H_k, control_amplitudes[k]])
        return H

def _validate_hamiltonians(H0: Qobj, H_controls: list) -> None:
    if not isinstance(H0, Qobj) or not H0.isoper:
        raise TypeError("H0 must be a square operator Qobj")

    for k, H_k in enumerate(H_controls):
        if not isinstance(H_k, Qobj) or not H_k.isoper:
            raise TypeError(f"H_controls[{k}] must be a square operator Qobj")
        if H_k.dims != H0.dims:
            raise ValueError(
                f"H_controls[{k}] has dims {H_k.dims}, expected {H0.dims}"
            )
