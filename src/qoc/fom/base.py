from abc import ABC, abstractmethod

import numpy as np
from qutip import Qobj


class FigureOfMerit(ABC):
    """
    Defines quantity being optimised in a control problem.
    Computes its scalar value based on a propagated and a target state/operator.

    Might compute a gradient w.r.t. parameters and time. 
    Convention: returns 1 for perfect closeness to target, 1 for worst case.
    Solvers minimize 1 - F.
    """


    @abstractmethod
    def compute(self, propagated: Qobj, target: Qobj) -> float:
        """
        Function defining figure of merit between propagated and target.

        Parameters
        ----------
        propagated : Qobj
            State or operator produced by the system propagation.
        target : Qobj
            Desired state or operator.

        Returns
        -------
        float
            Value in [0, 1]. 1 = perfect, 0 = worst.
        """
        pass

    # @abstractmethod
    # def gradient(self, control_pulse):
    #     # Gradient with respect to control pulse
    #     pass
