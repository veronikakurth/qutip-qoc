import numpy as np

from qoc.systems.controlled_system import ControlledSystem
from qoc.objectives.base import Objective


class OptimalControlProblem:
    """Full description of an OCP: what system, what goal, over what time, starting from what parameter guess."""

    def __init__(
        self,
        system: ControlledSystem,
        objective: Objective,
        times: np.ndarray,
        initial_parameters: np.ndarray,
    ):
        """
        Parameters
        ----------
        system : ControlledSystem
            The quantum system to be controlled (closed or open).
        objective : Objective
            Defines target and figure of merit.
        times : np.ndarray
            Time grid, shape (n_timesteps,).
        initial_parameters : np.ndarray
            1D initial guess vector in the parameterization chosen by the algorithm.
            (e.g. flattened amplitudes for piecewise-constant, basis
            coefficients for Fourier). 
            For piecewise-constant control, use
            ``PiecewiseConstant(K, N).initial_theta(amplitudes_2d)``.
            Algorithms validate its length against their
            PulseParameterization.n_parameters at solve time.
        """
        self._validate(system, objective, times, initial_parameters)
        self.system = system
        self.objective = objective
        self.times = times
        self.initial_parameters = np.asarray(initial_parameters).ravel()

    @staticmethod
    def _validate(system, objective, times, initial_parameters):
        if not isinstance(system, ControlledSystem):
            raise TypeError(f"system must be a ControlledSystem, got {type(system)}")
        if not isinstance(objective, Objective):
            raise TypeError(f"objective must be an Objective, got {type(objective)}")

        times = np.asarray(times)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times must be a 1D array with at least 2 points")
        if not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")

        initial_parameters = np.asarray(initial_parameters)
        if initial_parameters.size == 0:
            raise ValueError("initial_parameters must be non-empty")
