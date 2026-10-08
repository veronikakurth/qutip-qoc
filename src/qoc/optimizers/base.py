from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import scipy


@dataclass
class OptimizerResult:
    # Any optimizer-specific diagnostics
    n_iters: int


class Optimizer(ABC):

    @abstractmethod
    def minimize(
        self,
        fun: callable,
        x0,
        max_iter,
        tol,
        bounds: list[tuple[float | None, float | None]] | None = None,
        callback: Callable | None = None,
        **kwargs,
    ) -> OptimizerResult:
        """``callback(x)``, if given, is called once per iteration with the accepted iterate."""


class ScipyLBFGS(Optimizer):
    # TODO: add information about interface requirements for `fun` argument or link to the SciPy's documentation.
    def minimize(self, fun, x0, max_iter, tol, bounds=None, callback=None, **kwargs):
        opt_options = {"maxiter": max_iter, "gtol": tol}
        return scipy.optimize.minimize(
            fun=fun,
            x0=x0.flatten(),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options=(opt_options | kwargs),
        )
