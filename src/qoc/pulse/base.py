from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

# TODO: there could be also a class method approach (like in ControlledSystem) on creating an exact instance of a parameterisation
# Plus a class factory approach for creating custom parameterisations on the fly

class PulseParameterization(ABC):
    """Map an optimizer parameter vector theta -> control amplitudes u(t).

    theta is what the optimizer sees and updates (1-D, length n_parameters).
    u has shape (n_controls, n_timesteps) on the algorithm's time grid.
    """

    @property
    @abstractmethod
    def n_parameters(self) -> int: ... # Does it include time?

    def control_function(self, *args): ...
    
    # This method only works for len(theta) == n_timesteps, which is not the case in general
    def to_amplitudes(self, theta: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Return amplitudes, shape (1, len(times))."""

    def jacobian(
        self, theta: np.ndarray, times: np.ndarray
    ) -> np.ndarray | None:
        """Returns derivatives of control function w.r.t. control parameters.
        Shape: (len(times), n_parameters)."""
        return None
      
    def time_derivative() -> np.ndarray | None: ...

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
