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

    def minimize(self, fun, x0, max_iter, tol, **kwargs):
        opt_options = {"maxiter": max_iter, "gtol": tol}
        return scipy.optimize.minimize(fun=fun, x0=x0.flatten(), method="L-BFGS-B", options = (opt_options | kwargs))
