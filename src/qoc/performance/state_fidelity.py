from qutip import Qobj, fidelity
import numpy as np
 

from .base import PerformanceMeasure


class StateFidelity(PerformanceMeasure):
    """State fidelity: 1 = perfect match, 0 = orthogonal."""

    def compute(self, current: Qobj, target: Qobj) -> float:
        return abs(target.overlap(current))**2
        #return fidelity(current, target) # for state transfer closed system
        
