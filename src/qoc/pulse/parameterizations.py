import numpy as np

from qoc.pulse.base import PulseParameterization


class PiecewiseConstant(PulseParameterization):
    """Piecewise-constant amplitudes on a fixed time grid.

    The parameter vector theta is interpreted directly as a (n_controls,
    n_timesteps) amplitude array — the parameterization is an identity
    reshape. This is GRAPE's native form: every time slice carries one
    independently-tuned amplitude per control.

    Parameters
    ----------
    n_controls : int
        Number of control channels K.
    n_timesteps : int
        Number of time slices N (must match len(times) at call time).

    Attributes
    ----------
    n_parameters : int
        K * N — the size of theta.
    """

    def __init__(self, n_controls: int, n_timesteps: int):
        self.n_controls = n_controls
        self.n_timesteps = n_timesteps

    @property
    def n_parameters(self) -> int:
        return self.n_controls * self.n_timesteps

    def to_amplitudes(self, theta: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Reshape theta into a (n_controls, n_timesteps) amplitude array.

        `times` is accepted only to validate that the algorithm's grid matches
        the parameterization; its values are not used.
        """
        if len(times) != self.n_timesteps:
            raise ValueError(
                f"times has {len(times)} points, parameterization expects "
                f"{self.n_timesteps}"
            )
        return np.asarray(theta).reshape(self.n_controls, self.n_timesteps)

    def amplitude_jacobian(self, theta: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Dense identity Jacobian, shape (K, N, K*N).

        du[k, j] / dtheta[k', j'] = delta_{kk'} delta_{jj'}. Returned in dense
        form for interface uniformity with non-trivial parameterizations;
        hot-loop callers should special-case the identity rather than
        materializing this.
        """
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
