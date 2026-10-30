import numpy as np

from qoc.pulse.base import PulseParameterization
# Due to n_controls attribute of the base class, this means we impose a one-to-many relationship: 
# one parameterisation for many controls. This might not be the case in general
# Maybe we need a simple way to define parameterisation per control via factory approach
# Example of workflow: 
# User provides a control function per Hamiltonian
# If the goal is to use a gradient driven method, we ask for derivatives with respect to every parameter of control function (alpha, time)
# (Unless the method supports auto-differentiation)
# Based on it, a class is created with API that is necessary to be used by Algorithm etc.
# Can we have an alternative class to the base class that would accept arguments in a more general form? Like FunctionalPerformance measure, for example

# Potentially, we could use a class factory function pattern to generate different classes from configuration
# This solves, however, only the problem of the user interface to pulse parameterisation
# There is another problem left: binding parameterisation to a control Hamiltonian and keeping this information together.
class Fourier(PulseParameterization):
    
    def control_func(self, a1, a2, t):
        # a1 - amplitude, a2 - frequency 
        return a1 * np.sin(a2 * t)
   
    def da1_control_func(self, a1, a2, t):
        return np.sin(a2 * t)

    def da2_control_func(self, a1, a2, t):
        return np.cos(a2 * t) * t

    def dt_control_func(self, a1, a2, t):
        return np.cos(a2 * t) * a1 * a2
    
    def jacobian(self, theta, times):
        da1_derivatives = self.da1_control_func(*theta, times).reshape(len(times), 1)
        da2_derivatives = self.da2_control_func(*theta, times).reshape(len(times), 1)
        return np.concat([da1_derivatives, da2_derivatives], axis=1)

    
    def initial_theta(self, amplitudes: np.ndarray) -> np.ndarray:
        return np.array([self.control_func(*amplitudes, t=0)])
    # For now, this API seems to be enough. The most of the API of the base class is not used. As well as n_controls

    # how useful is this property
    @property
    def n_parameters(self) -> int:
        return 2 # do we have to include time?

class PiecewiseConstant(PulseParameterization):
    """Piecewise-constant amplitudes on a fixed time grid.

    The parameter vector theta is (1, n_timesteps) amplitude array.
    Main use: in GRAPE, where every time slice carries one independently-tuned amplitude per control.

    Parameters
    ----------
    T : 
    N : 
    times :
    amplitude_range : tuple[float | None, float | None] | None, optional
        (min, max) bounds applied to every amplitude. Use ``None`` on
        either side to leave that direction unbounded. Default ``None`` means
        the controls are fully unconstrained.

    Attributes
    ----------
    n_parameters : int
        K * N - the size of theta, parameter vector.
    """

    def __init__(
        self,
        T: int = None,
        N: int = None,
        times: np.ndarray = None,
        amplitude_range: tuple[float | None, float | None] | None = None
    ):
        if times is None:
            times = np.linspace(0, T, N) # TODO: add validation that checks that either times or T + N are passed (exclusive or)
        self.times = np.asarray(times)
        self.n_timesteps = len(self.times)
        self.amplitude_range = amplitude_range
    
    @property
    def dt(self) -> float:
        return self.times[1] - self.times[0] # uniform grid assumption

    @property
    def n_parameters(self) -> int:
        return self.n_timesteps

    def bounds(self) -> list[tuple[float | None, float | None]] | None:
        if self.amplitude_range is None:
            return None
        return [self.amplitude_range] * self.n_parameters

    def to_amplitudes(self, theta: np.ndarray) -> np.ndarray:
        """Reshape theta into a (1, n_timesteps) amplitude array.
        """
        return np.asarray(theta).reshape(1, self.n_timesteps)

    def jacobian(self, *theta: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Dense identity Jacobian, shape (N, N)."""
        # TODO: returned in a dense form (numpy array) - can we do better?
        N = self.n_timesteps
        J = np.diag((N, N))
        return J

    def initial_theta(self, amplitudes: np.ndarray) -> np.ndarray:
        """Build theta from a (n_controls, n_timesteps) amplitude array.

        This is the user-facing entry point for constructing
        `OptimalControlProblem.initial_parameters` when working with
        piecewise-constant controls.
        """
        amplitudes = np.asarray(amplitudes)
        if amplitudes.shape != (1, self.n_timesteps):
            raise ValueError(
                f"amplitudes must have shape (1, "
                f"{self.n_timesteps}), got {amplitudes.shape}"
            )
        return amplitudes.ravel()
