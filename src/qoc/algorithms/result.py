from dataclasses import dataclass

import numpy as np
from qutip import Qobj


@dataclass
class Result:
    """Optimization output with loss and independent named reporting metrics.

    ``loss_history`` contains the initial loss and one loss per accepted
    optimizer iterate, excluding line-search trials. ``metrics`` describes
    ``final_state``; values have measure-specific meanings and need not improve
    with the loss. No fidelity is inferred for distance or custom loss measures.
    """

    optimized_pulses: np.ndarray
    loss: float
    n_iters: int
    optimizer_info: object
    loss_history: list[float]
    final_state: Qobj
