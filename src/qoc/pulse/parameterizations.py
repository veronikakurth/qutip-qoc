import numpy as np

from qoc.pulse.base import PulseParameterization


class PiecewiseConstant(PulseParameterization):
    """Piecewise-constant amplitudes on a fixed time grid.

    The parameter vector theta is (n_controls, n_timesteps) amplitude array.
    Main use: in GRAPE, where every time slice carries one independently-tuned amplitude per control.

    Parameters
    ----------
    n_controls : int
        Number of controls.
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
        n_controls: int,
        T: int = None,
        N: int = None,
        times: np.ndarray = None,
        amplitude_range: tuple[float | None, float | None] | None = None
    ):
        self.n_controls = n_controls
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
        return self.n_controls * self.n_timesteps

    def bounds(self) -> list[tuple[float | None, float | None]] | None:
        if self.amplitude_range is None:
            return None
        return [self.amplitude_range] * self.n_parameters

    def to_amplitudes(self, theta: np.ndarray) -> np.ndarray:
        """Reshape theta into a (n_controls, n_timesteps) amplitude array.
        """
        return np.asarray(theta).reshape(self.n_controls, self.n_timesteps)

    def amplitude_jacobian(self, theta: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Dense identity Jacobian, shape (K, N, K*N)."""
        # TODO: returned in a dense form (numpy array) - can we do better?
        K, N = self.n_controls, self.n_timesteps
        J = np.zeros((K, N, K * N))
        for k in range(K):
            for j in range(N):
                J[k, j, k * N + j] = 1.0
        return J

    def initial_theta(self, amplitudes: np.ndarray) -> np.ndarray:
        """Build theta from a (n_controls, n_timesteps) amplitude array.

        This is the user-facing entry point for constructing
        `OptimalControlProblem.initial_parameters` when working with
        piecewise-constant controls.
        """
        amplitudes = np.asarray(amplitudes)
        if amplitudes.shape != (self.n_controls, self.n_timesteps):
            raise ValueError(
                f"amplitudes must have shape ({self.n_controls}, "
                f"{self.n_timesteps}), got {amplitudes.shape}"
            )
        return amplitudes.ravel()
