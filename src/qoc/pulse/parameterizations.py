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
        
    # how useful is this property; how not to hardcode it for the user-created parameterisation
    @property
    def n_parameters(self) -> int:
        return 2

    def control_func(self, *args):
        a = args["a"]
        t = args["t"]
        # a1 - amplitude, a2 - frequency 
        return a[0] * np.sin(a[1] * t)
   
    def control_funcs_derivatives(self, *args) -> list[callable]:
        # could be replaced with autograd!
        a = args["a"]
        t = args["t"]
        return [lambda t, a: np.sin(a[1] * t),                         # dc/da1
                lambda t, a: a[0] * t * np.cos(a[1] * t)]     # dc/da2
    #
    # def dt_control_func(self, *args):
    #     return np.cos(a2 * t) * a1 * a2
    
    def initial_theta(self, values: np.ndarray) -> np.ndarray:
        return np.array([self.control_func(*values, t=0)])


class PiecewiseConstant(PulseParameterization):
    """Piecewise-constant amplitudes on a fixed time grid.

    The parameter vector theta is (1, n_timesteps) amplitude array.

    Parameters
    ----------
    T : 
    N : 
    times :
    bounds : tuple[float | None, float | None] | None, optional
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
        bounds: tuple[float | None, float | None] | None = None
    ):
        if times is None:
            times = np.linspace(0, T, N) # TODO: add validation that checks that either times or T + N are passed (exclusive or)
        self.times = np.asarray(times)
        self.n_timesteps = len(self.times)
        self.bounds = bounds
    
    @property
    def dt(self) -> float:
        return self.times[1] - self.times[0] # uniform grid assumption

    @property
    def n_parameters(self) -> int:
        return self.n_timesteps

    def get_bounds(self) -> list[tuple[float | None, float | None]] | None:
        if self.bounds is None:
            return None
        return [self.bounds] * self.n_parameters

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


class Parameterisation:

    @staticmethod
    def fourier():
        return Fourier()

    @staticmethod
    def piecewise_constant(T, N, times, bounds):
        return PiecewiseConstant(T, N, times, bounds)
