from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

# TODO: there could be also a class method approach (like in ControlledSystem) on creating an exact instance of a parameterisation
# Plus a class factory approach for creating custom parameterisations on the fly
# TODO: make the base class a thin wrapper around functions and its derivatives
class PulseParameterization(ABC):
    """Map an optimizer parameter vector theta -> control amplitudes u(t).

    theta is what the optimizer sees and updates (1-D, length n_parameters).
    u has shape (n_controls, n_timesteps) on the algorithm's time grid.
    """

    @property
    @abstractmethod
    def n_parameters(self) -> int: ...

    def control_func(self, *args) -> callable: ...
    
    # w.r.t. control parameters
    def control_funcs_derivatives(self, *args) -> list[callable]: ...

    def dt_control_func(self, *args): ...
    
    # TODO: this could also return functions
    def bounds(self) -> list | None:
        """Per-parameter bounds for constrained optimizers.

        Returns one ``(min, max)`` pair per entry of theta. Use ``None`` on
        either side of a pair to leave that direction unbounded. 
        If the whole return value is ``None`` -> the parameterization is fully
        unconstrained.
        """
        return None

    @abstractmethod
    def initial_theta(self, *args, **kwargs) -> np.ndarray:
        """Build a 1-D parameter vector from user input.

        Subclass-specific signature.
        """


# Idea: parameterisations = PulseParameterization()
# import parameterizations
# parameterizations.fourier(...)
# parameterizations.piecewise_constant(...)
