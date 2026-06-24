from qoc.algorithms.base import Algorithm
from qoc.algorithms.result import Result
from qoc.dynamics.simulator import Simulator
from qoc.problem import OptimalControlProblem
from qoc.pulse.base import PulseParameterization


class CRAB(Algorithm):

    def __init__(
        self,
        parameterization: PulseParameterization,
        simulator: Simulator | None = None,
    ):
        super().__init__(simulator)
        self.parameterization = parameterization

    def solve(self, problem: OptimalControlProblem) -> Result:
        simulator = self._get_simulator(problem.system.dynamics)
        param = self.parameterization
        theta0 = problem.initial_parameters
        if theta0.size != param.n_parameters:
            raise ValueError(
                f"initial_parameters has size {theta0.size}, parameterization "
                f"expects {param.n_parameters}"
            )
        # TODO: optimizer loop. Loss evaluation:
        #   u = param.to_amplitudes(theta, problem.times)
        #   out = simulator.evolve(problem.system.dynamics, u, problem.times,
        #                          problem.objective.initial)
        #   loss = problem.objective.loss(out.final)
        raise NotImplementedError
