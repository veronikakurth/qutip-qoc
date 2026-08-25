from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import numpy as np
from qutip import Qobj

StateType = Literal["ket", "dm"]

class System(ABC):
    """
    Defines physics of the controlled system.
    Non-user-facing class.
    """

    def __init__(self, H0: Qobj, H_controls: list[Qobj]):
        self._validate_hamiltonians(H0, H_controls)
        self._H0 = H0
        self._H_controls = H_controls

    # System representation

    state_type: ClassVar[StateType]

    @abstractmethod
    def encode_state(self, state: Qobj) -> Qobj:
        """Depending on the system type,
           it is either a column-vector representation or identity. Inverse of decode_state""" 

    @abstractmethod
    def decode_state(self, arr: np.ndarray) -> Qobj:
        """Reconstruct the physical Qobj from encode_state's output """

    # decode_state(encode_state(x)) ~= x

    def encode_operator(self, op: Qobj) -> Qobj:
        return op

    @abstractmethod
    def decode_operator(self, arr: np.ndarray) -> Qobj:
        """Dual method to encode_operator, implementation is system-specific"""

    @abstractmethod
    def control_generators(self) -> list[Qobj]:
        """Control operators in the generator space (H_k vs L_k) """

    @abstractmethod
    def motion_generator_time_j(self, u: np.ndarray, j: int) -> Qobj:
        """To be used for propagator computation based on drift term,
        time-dependent control terms with injected pulse and pre-factor term,
        which depends on system type"""
    
    @property
    def n_controls(self) -> int:
        return len(self._H_controls)
    
    @property
    def drift(self) -> Qobj:
        """Drift term of ``build_generator``, in this system's generator space.

        ``H0`` for a closed system, ``L0`` for an open one. Every concrete
        System must supply this on every construction path.
        """
        return self._H0

    @property
    def controls(self) -> list[Qobj]:
        return self._H_controls

    @property
    def dims(self) -> list:
        """Hilbert-space dims of the controlled system, as ``[[n], [n]]``.

        Always the *Hilbert* space, never the Liouville space, so that
        ``Objective.check_compatible`` can compare ``dims[0]`` against the
        dims of a ket or density matrix regardless of system type.
        """
        return self._H0.dims

    @property
    def shape(self) -> tuple[int, int]:
        """Hilbert-space shape of the controlled system."""
        return self._H0.shape

    @abstractmethod
    def build_generator(self, control_amplitudes: np.ndarray) -> list:
        """Return the time-dependent Hamiltonian in qutip's nested-list format.

        Parameters
        ----------
        control_amplitudes : np.ndarray
            Shape (n_controls, n_timesteps).

        Returns
        -------
        list
            ``[H0, [H_1, u_1], [H_2, u_2], ...]``
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
