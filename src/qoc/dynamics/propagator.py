from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import expm

from qutip import Qobj, sesolve, mesolve
from qoc.systems.base import System

class Propagator(ABC):
   
    @abstractmethod
    def propagate(system: System, **kwargs):  # TODO: maybe the better name would be "compute": Propagator.compute()
        pass

def eigendecomposition(H: Qobj, dt: float) -> np.ndarray:
    U_j = expm(-1j * H.full() * dt)
    return U_j

class StepPropagator(Propagator):

    def propagate(self, system: System, N: int, control_amplitudes, dt, **kwargs) -> list:
        """
            system : System
            N : int
                Number of time slices.
            dt : float
                Discretization step.
            control_amplitudes : np.ndarray
        """
        # Dev note: Propagator is meant to be used privately by different solvers, so it can expect parts of optimal control problem definition be passed directly, not wrapped into a container class. Hence our motivation to expand all needed arguments in the signature.

        propagators = [None] * N
        # Build a full system Hamiltonian for time j (remember: controls are time dependent)
        for j in range(N):
            step_Hamiltonian = system.build_hamiltonian_time_j(control_amplitudes, j)
            # Compute a propagator for the jth time slice (corresponds to (j, j + 1) time interval)
            step_propagator = eigendecomposition(step_Hamiltonian, dt)
            # Collect a propagator into the list
            propagators[j] = step_propagator

        return propagators

class FinalStatePropagator(Propagator):

    def propagate(system: System, initial: Qobj, control_amplitudes: np.ndarray, times: np.ndarray) -> Qobj:
        """
        Propagate a state or operator under the time-dependent Hamiltonian.

        Parameters
        ----------
        system: System
            Required part of the physics of the system will be accesses from this object.
        initial : Qobj
            Initial state (ket/DM) for state transfer, or initial operator
            (typically identity) for gate synthesis.
        control_amplitudes : np.ndarray
            Control pulse amplitudes, shape (n_controls, n_timesteps).
            Must match len(H_controls) and len(times).
        times : np.ndarray
            Time grid, shape (n_timesteps,).

        Returns
        -------
        Qobj
            Final state (ket/DM) or final unitary operator after system evolution.
        """
        _validate_propagation_inputs(system.n_controls, initial, control_amplitudes, times)

        H = self.build_hamiltonian(control_amplitudes, times)
        if initial.isket:
            result = sesolve(H, initial, times)
        else:
            result = mesolve(H, initial, times, c_ops=[])
        
        return result.states[-1]

def _validate_control_amplitudes(
    n_controls: int,
    control_amplitudes: np.ndarray,
    times: np.ndarray,
) -> None:
    control_amplitudes = np.asarray(control_amplitudes)
    if control_amplitudes.shape != (n_controls, len(times)):
        raise ValueError(
            f"control_amplitudes must have shape (n_controls={len(H_controls)}, "
            f"n_timesteps={len(times)}), got {control_amplitudes.shape}"
        )


def _validate_propagation_inputs(
    n_controls: int,
    initial: Qobj,
    control_amplitudes: np.ndarray,
    times: np.ndarray,
) -> None:
    if not isinstance(initial, Qobj):
        raise TypeError(f"initial must be a Qobj, got {type(initial)}")
    _validate_control_amplitudes(n_controls, control_amplitudes, times)
