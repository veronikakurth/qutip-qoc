from abc import ABC, abstractmethod

from qoc.problem import OptimalControlProblem
from qoc.algorithms.result import Result

class Algorithm(ABC):
    
    @abstractmethod
    def solve(self, problem: OptimalControlProblem) -> Result:
        raise NotImplementedError()
