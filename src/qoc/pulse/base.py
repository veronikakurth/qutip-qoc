from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class PulseParameterization(ABC):
    """Map an optimizer parameter vector theta -> control amplitudes u(t).

    theta is what the optimizer sees and updates (1-D, length n_parameters).
    u has shape (n_controls, n_timesteps) on the algorithm's time grid.
    """

    n_controls: int

    @property
    @abstractmethod
    def n_parameters(self) -> int: ...

    @abstractmethod
    def to_amplitudes(self, theta: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Return amplitudes, shape (n_controls, len(times))."""

    def amplitude_jacobian(
        self, theta: np.ndarray, times: np.ndarray
    ) -> np.ndarray | None:
        """Return du/dtheta with shape (n_controls, len(times), n_parameters).

        Default None: algorithms that need analytic Jacobians must fall back
        to finite differences or raise.
        """
        return None

    def to_callable(self, theta: np.ndarray) -> Callable[[float], np.ndarray] | None:
        """Return a callable t -> amplitudes(t) of shape (n_controls,).

        Optional: required by algorithms (e.g. GOAT) that integrate the
        equation of motion with an analytic control profile rather than
        sampling on a fixed time grid. Default None means "not supported".
        """
        return None

    def bounds(self) -> list[tuple[float | None, float | None]] | None:
        """Per-parameter bounds for constrained optimizers.

        Returns one ``(min, max)`` pair per entry of theta. Use ``None`` on
        either side of a pair to leave that direction unbounded. The whole
        return value being ``None`` means the parameterization is fully
        unconstrained.
        """
        return None

    @abstractmethod
    def initial_theta(self, *args, **kwargs) -> np.ndarray:
        """Build a 1-D parameter vector from user-friendly input.

        Subclass-specific signature.
        """
