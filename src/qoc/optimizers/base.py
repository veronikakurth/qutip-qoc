from abc import ABC, abstractmethod

import scipy

class OptimizerResult:
    # Any optimizer specific diagnostics
    def __init__(self, **kwargs):
        pass

class Optimizer(ABC):

    @abstractmethod
    def minimize(self, loss_and_grad: callable, x0, max_iter, tol, **kwargs) -> OptimizerResult:
        pass


class ScipyLBFGS(Optimizer):

    def minimize(self):
        return scipy.optimize.minimize(method="L-BFGS-B")
