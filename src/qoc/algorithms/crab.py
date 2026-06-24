from qoc.algorithms.base import Algorithm
from qoc.dynamics.simulator import Simulator
from qoc.problem import OptimalControlProblem
from qoc.algorithms.result import Result


class CRAB(Algorithm):

    def __init__(self, simulator: Simulator | None = None):
        super().__init__(simulator)

    def solve(self, problem: OptimalControlProblem) -> Result:
        simulator = self._get_simulator(problem.system.dynamics)
        # TODO: parameterization + optimizer loop. Loss evaluation will call:
        #   simulator.evolve(problem.system.dynamics, u, problem.times,
        #                    problem.objective.initial)
        raise NotImplementedError
