from abc import ABC, abstractmethod
from inspect import signature

import numpy as np
from qutip import Qobj


class PerformanceMeasure(ABC):
    """
    Scores how close the achieved state or operator is to the target.

    Subclasses define the specific measure (fidelity, infidelity, distance, etc.)
    and its sign convention. The only contract is: compute() returns a finite scalar.
    """

    @abstractmethod
    def compute(self, current: Qobj, target: Qobj) -> float:
        pass


class FunctionalPerformanceMeasure(PerformanceMeasure):
    """
    Wraps a user-supplied callable as a PerformanceMeasure.

    The callable must accept (current, target) and return a finite scalar.
    This is the escape hatch for custom scoring logic without subclassing.
    """

    def __init__(self, func):
        _validate_func(func)
        self.func = func

    def compute(self, current: Qobj, target: Qobj) -> float:
        result = self.func(current, target)
        if not np.isscalar(result):
            raise TypeError(
                f"Performance measure must return a scalar, got {type(result)}"
            )
        if not np.isfinite(result):
            raise ValueError(
                f"Performance measure returned a non-finite value: {result}"
            )
        return float(result)


def _validate_func(func) -> None:
    sig = signature(func)
    if len(sig.parameters) < 2:
        raise TypeError(
            "Performance measure function must accept at least (current, target)"
        )
