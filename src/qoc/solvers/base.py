from abc import ABC, abstractmethod

from qoc.problem import OptimalControlProblem
from qoc.result import Result

class OCPSolver(ABC):
    
    @abstractmethod
    def solve(self, problem: OptimalControlProblem) -> Result:
        raise NotImplementedError()
