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
