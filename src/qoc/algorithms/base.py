from abc import ABC, abstractmethod

from qoc.problem import OptimalControlProblem
from qoc.algorithms.result import Result
from qoc.dynamics.simulator import Simulator, default_simulator_for


class Algorithm(ABC):
    """Base class for optimal control algorithms.

    Subclasses that rely on forward simulation (CRAB, GOAT, ...) should use
    self._simulator(system) to obtain the configured Simulator. Adjoint-based
    algorithms (GRAPE) construct their own per-slice propagators and ignore it.
    """

    def __init__(self, simulator: Simulator | None = None):
        self._simulator = simulator

    def _get_simulator(self, system) -> Simulator:
        """Return the user-supplied Simulator, or the system-appropriate default."""
        return self._simulator or default_simulator_for(system)

    @abstractmethod
    def solve(self, problem: OptimalControlProblem) -> Result:
        raise NotImplementedError()
