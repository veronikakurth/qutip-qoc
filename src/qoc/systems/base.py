from abc import ABC, abstractmethod

import numpy as np
from qutip import Qobj


class System(ABC):
    """Physics of the controlled system.

    Concrete subclasses (ClosedSystem, OpenSystem, ...) must expose the
    control-affine Hamiltonian H(t) = H0 + sum_k u_k(t) H_k in qutip's
    nested-list format, so the same Simulator implementations can drive
    either sesolve or mesolve.
    """
    
    def __init__(self, H0: Qobj, H_controls: list[Qobj]):
        self._validate_hamiltonians(H0, H_controls)
        self._H0 = H0
        self._H_controls = H_controls
    
    @property
    def n_controls(self) -> int:
        return len(self._H_controls)
    
    @property
    def drift(self):
        return self._H0

    @property
    def controls(self):
        return self._H_controls
    
    @property
    def dims(self) -> int: # TODO: align type with qutip core
        return self._H0.dims

    @property
    def shape(self):
        return self._H0.shape

    @abstractmethod
    def build_generator(self, control_amplitudes: np.ndarray) -> list:
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

    @abstractmethod
    def build_generator_time_j(self, control_amplitudes, j):
        """
        """

    def _validate_hamiltonians(self, H0: Qobj, H_controls: list) -> None:
        if not isinstance(H0, Qobj) or not H0.isoper:
            raise TypeError("H0 must be a square operator Qobj")

        for k, H_k in enumerate(H_controls):
            if not isinstance(H_k, Qobj) or not H_k.isoper:
                raise TypeError(f"H_controls[{k}] must be a square operator Qobj")
            if H_k.dims != H0.dims:
                raise ValueError(
                    f"H_controls[{k}] has dims {H_k.dims}, expected {H0.dims}"
                )
