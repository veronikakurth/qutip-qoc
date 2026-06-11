import numpy as np

from qoc.systems.controlled_system import ControlledSystem
from qoc.objectives.base import Objective


class OptimalControlProblem:
    """
    Full description of an OCP: what system, what goal, over what time, starting from what pulse guess.

    This is the central validation point where all inputs are checked before
    being handed to a solver.
    """

    def __init__(
        self,
        system: ControlledSystem,
        objective: Objective,
        times: np.ndarray,
        initial_pulses: np.ndarray, # TODO: can be also initial pulse parameters - make the argument more general
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
        initial_pulses : np.ndarray
            Initial guess for control amplitudes, shape (n_controls, n_timesteps).
        """
        self._validate(system, objective, times, initial_pulses)
        self.system = system
        self.objective = objective
        self.times = times
        self.initial_pulses = initial_pulses

    @staticmethod
    def _validate(system, objective, times, initial_pulses):
        if not isinstance(system, ControlledSystem):
            raise TypeError(f"system must be a ControlledSystem, got {type(system)}")
        if not isinstance(objective, Objective):
            raise TypeError(f"objective must be an Objective, got {type(objective)}")

        times = np.asarray(times)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times must be a 1D array with at least 2 points")
        if not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")

        initial_pulses = np.asarray(initial_pulses)
        if initial_pulses.ndim != 2:
            raise ValueError("initial_pulses must be 2D with shape (n_controls, n_timesteps)")
        if initial_pulses.shape[1] != len(times):
            raise ValueError(
                f"initial_pulses has {initial_pulses.shape[1]} timesteps "
                f"but times has {len(times)}"
            )
