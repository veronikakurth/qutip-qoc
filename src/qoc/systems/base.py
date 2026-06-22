from abc import ABC, abstractmethod

import numpy as np


class System(ABC):
    """Physics of the controlled system.

    Concrete subclasses (ClosedSystem, OpenSystem, ...) must expose the
    control-affine Hamiltonian H(t) = H0 + sum_k u_k(t) H_k in qutip's
    nested-list format, so the same Simulator implementations can drive
    either sesolve or mesolve.
    """

    @abstractmethod
    def build_hamiltonian(self, control_amplitudes: np.ndarray) -> list:
        """Return the time-dependent Hamiltonian in qutip's nested-list format.

        Parameters
        ----------
        control_amplitudes : np.ndarray
            Shape (n_controls, n_timesteps). Coefficient samples on the
            solver's `tlist` grid.

        Returns
        -------
        list
            ``[H0, [H_1, u_1], [H_2, u_2], ...]`` consumable by sesolve/mesolve.
        """
