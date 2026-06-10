from dataclasses import dataclass

import numpy as np

# TODO: think of usability of optimized pulses - can they be easily put into solver? This is a probable use case after optimisation. Are there more use cases
@dataclass
class Result:
    """
    Output of a completed optimisation procedure.

    Attributes
    ----------
    optimized_pulses : np.ndarray
        Control amplitudes after optimisation, shape (n_controls, n_timesteps).
    fidelity : float
        Achieved fidelity (1 = perfect).
    n_iters : int
        Number of iterations until convergence or termination.
    optimizer_info : dict
        Optimizer-specific metadata (convergence message, etc.).
    """

    optimized_pulses: np.ndarray
    fidelity: float
    n_iters: int
    optimizer_info: dict
