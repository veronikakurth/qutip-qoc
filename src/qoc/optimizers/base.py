from abc import ABC, abstractmethod
from dataclasses import dataclass

import scipy

@dataclass
class OptimizerResult:
    # Any optimizer specific diagnostics
    n_iters: int

class Optimizer(ABC):

    @abstractmethod
    def minimize(self, fun: callable, x0, max_iter, tol, **kwargs) -> OptimizerResult:
        pass


class ScipyLBFGS(Optimizer):
    # TODO: add information about interface requirements for `fun` argument or link to the SciPy's documentation.
    def minimize(self, fun, x0, max_iter, tol, **kwargs):
        opt_options = {"maxiter": max_iter, "gtol": tol}
        return scipy.optimize.minimize(fun=fun, x0=x0.flatten(), jac=True, method="L-BFGS-B", options = (opt_options | kwargs))
