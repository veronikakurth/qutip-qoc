from abc import ABC, abstractmethod
from inspect import signature
from numpy import isscalar, isfinite
from qutip import Qobj

def _validate_objective_function(func: callable):
    sig = signature(func)
    params = list(sig.parameters.values())

    if len(params) < 2:
        raise TypeError("Too few arguments. Objective function must accept at least (current_state, target)") 

class Objective(ABC):
    """
    Holds the target and the measure scoring how close the current system's state is to the target.

    Design approach: inheritance. An (advanced) user may extend the class hierarchy to implement custom objectives.

    """

    def __init__(self, target: Qobj):
        self.target = target

    @abstractmethod
    def compute(self, current: Qobj):
        pass

    def gradient(self, current: Qobj):
        # Gradient of state vector
        raise NotImplemented

class FunctionalObjective(Objective):
    """
    Holds the target and the measure scoring how close the current system's state is to the target.
    Design approach: composition (function injection) - a user may plug and play with custom objectives.
    """

    def __init__(self, target, func, grad_func = None):
        super().__init__(target)
        self.objective_func = func
        self.grad_func = grad_func
        self._validate()
    
    def _validate(self):
        _validate_objective_function(self.objective_func)
        # TODO: checks for grad func 

    def compute(self, current: Qobj):
        if current.dims != self.target.dims:
            raise ValueError("Mismatching dimensions of current and target")

        result = self.objective_func(current, self.target)

        if not isscalar(result):
            raise TypeError(f"Return result of objective must be a scalar.\n Instead, got type {type(result)}")
        
        # Numerical sanity check
        if not isfinite(result):
            raise ValueError(f"Not a finite result. Instead, {result}")

        return result
    
    def gradient(self, current: Qobj):
        result = self.grad_func(current)
        # TODO: checks for result
        return result
