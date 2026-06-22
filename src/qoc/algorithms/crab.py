from qoc.algorithms.base import Algorithm
from qoc.dynamics.simulator import Simulator, default_simulator_for
from qoc.problem import OptimalControlProblem
from qoc.algorithms.result import Result


class CRAB(Algorithm):

    def __init__(self, simulator: Simulator | None = None):
        self.simulator = simulator

    def solve(self, problem: OptimalControlProblem) -> Result:
        simulator = self.simulator or default_simulator_for(problem.system.dynamics)
        # TODO: parameterization + optimizer loop. Loss evaluation will call:
        #   simulator.evolve(problem.system.dynamics, u, problem.times,
        #                    problem.objective.initial)
        raise NotImplementedError
