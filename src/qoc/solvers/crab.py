from .base import OCPSolver
from qoc.problem import OptimalControlProblem
from qoc.result import Result

class CRAB(OCPSolver):

    def solve(self, problem: OptimalControlProblem) -> Result:
        pass
