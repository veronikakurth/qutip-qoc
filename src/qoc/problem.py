import numpy as np

from qoc.systems.base import System
from qoc.objectives.base import Objective

# A so-called compatibility layer
class OptimalControlProblem:
    """Full description of an OCP: what system, what goal, over what time, starting from what parameter guess."""
    # TODO: maybe we should simplify to Problem(system, objective)
    # TODO: move initial_parameters description to a better place
    def __init__(
        self,
        system: System,
        objective: Objective,
    ):
        """
        Parameters
        ----------
        system : System
            The quantum system to be controlled, as returned by
            ``ControlledSystem.closed`` / ``.open`` / ``.open_liouvillian``.
        objective : Objective
            Defines initial, target and performance measure.
        times : np.ndarray
            Time grid, shape (n_timesteps,). # Time grid sounds too GRAPE specific.
        initial_parameters : np.ndarray
            1D initial guess vector in the parameterization chosen by the algorithm.
            (e.g. flattened amplitudes for piecewise-constant, basis
            coefficients for Fourier). 
            For piecewise-constant control, use
            ``PiecewiseConstant(K, N).initial_theta(amplitudes_2d)``.
            Algorithms validate its length against their
            PulseParameterization.n_parameters at solve time.
        """
        self._validate(system, objective)
        self.system = system
        self.objective = objective

    @staticmethod
    def _validate(system, objective):
        if not isinstance(system, System):
            raise TypeError(
                f"system must be a System (build one with "
                f"ControlledSystem.closed/.open/.open_liouvillian), "
                f"got {type(system)}"
            )
        if not isinstance(objective, Objective):
            raise TypeError(f"objective must be an Objective, got {type(objective)}")
        objective.check_compatible(system)
