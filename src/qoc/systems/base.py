from abc import ABC, abstractmethod

import numpy as np
from qutip import Qobj

# Model of a system to be controlled. Described by Hamiltonians or Lindbladians and time-dependent pulse coefficients 
# Structure: controlled + uncontrolled Hamiltonians/Lindbladians
# Given time-dependent pulse amplitudes and initial state, can compute time evolution

class ControlledSystem(ABC):

    @abstractmethod
    def evolve(self,
                  initial_state: Qobj,
                  control_amplitudes: np.ndarray,
                  times: np.ndarray) -> Qobj:
        """
        Propagate the quantum state/operator under given controls.

        Parameters
        ----------
        initial_state : Qobj
            Initial quantum state or operator
        control_amplitudes : np.ndarray
            Control pulse amplitudes, shape (n_controls, n_timesteps)
        times : np.ndarray
            Time points, shape (n_timesteps, )

        Returns
        -------
        Qobj # TODO: potentially, this is Result object returned by the Solver
            Final state/operator after propagation
        """
        pass

    # @abstractmethod # TODO: maybe this method would belong to ABC class in case we find a way to generalize Hamiltonians composition
    # def build_hamiltonian():
    #     pass
    # TODO: How do we do validation of system's Hamiltonians? And which kind of they have to be. Unitary is enough or also commutative?
