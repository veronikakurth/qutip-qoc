import numpy as np
from qutip import Qobj, sesolve

from .base import System


class ClosedSystem(System):
    """
    Closed quantum system described by a drift + control Hamiltonian.

    H(t) = H0 + sum_k u_k(t) * H_k

    where H0 is the drift (time-independent) and H_k are the control
    Hamiltonians scaled by the piecewise-constant amplitudes u_k(t).
    """

    def __init__(self, H0: Qobj, H_controls: list[Qobj]):
        """
        Parameters
        ----------
        H0 : Qobj
            Drift Hamiltonian.
        H_controls : list[Qobj]
            Control Hamiltonians.
        """
        _validate_hamiltonians(H0, H_controls)
        self.H0 = H0
        self.H_controls = H_controls
    # TODO: may need to rather return Result object
    def evolve(
        self,
        initial: Qobj,
        control_amplitudes: np.ndarray,
        times: np.ndarray,
    ) -> Qobj:
        """
        Propagate a state or operator under the time-dependent Hamiltonian.

        Parameters
        ----------
        initial : Qobj
            Initial state (ket/DM) for state transfer, or initial operator
            (typically identity) for gate synthesis.
        control_amplitudes : np.ndarray
            Control pulse amplitudes, shape (n_controls, n_timesteps).
            Must match len(H_controls) and len(times).
        times : np.ndarray
            Time grid, shape (n_timesteps,).

        Returns
        -------
        Qobj
            Final state (ket/DM) or final unitary operator after system evolution.
        """
        _validate_propagation_inputs(self.H_controls, initial, control_amplitudes, times)

        H = self._build_hamiltonian(control_amplitudes, times)
        result = sesolve(H, initial, times)
        return result.states[-1]
    
    # TODO: this will be different for different systems (open / closed); but the procedure is potentially generalizable to one method in base class
    def _build_hamiltonian(
        self, control_amplitudes: np.ndarray, times: np.ndarray
    ) -> list:
        """Build the QuTiP time-dependent Hamiltonian list."""
        H = [self.H0]
        for k, H_k in enumerate(self.H_controls):
            H.append([H_k, control_amplitudes[k]])
        return H


def _validate_hamiltonians(H0: Qobj, H_controls: list) -> None: # TODO: I believe there may be more properties we need to check in order to propagate the system. Or are these done by the solver?
    if not isinstance(H0, Qobj) or not H0.isoper:
        raise TypeError("H0 must be a square operator Qobj")

    for k, H_k in enumerate(H_controls):
        if not isinstance(H_k, Qobj) or not H_k.isoper:
            raise TypeError(f"H_controls[{k}] must be a square operator Qobj")
        if H_k.dims != H0.dims:
            raise ValueError(
                f"H_controls[{k}] has dims {H_k.dims}, expected {H0.dims}"
            )

def _validate_control_amplitudes(
    H_controls: list,
    control_amplitudes: np.ndarray,
    times: np.ndarray,
) -> None:
    control_amplitudes = np.asarray(control_amplitudes)
    if control_amplitudes.shape != (len(H_controls), len(times)):
        raise ValueError(
            f"control_amplitudes must have shape (n_controls={len(H_controls)}, "
            f"n_timesteps={len(times)}), got {control_amplitudes.shape}"
        )


def _validate_propagation_inputs(
    H_controls: list,
    initial: Qobj,
    control_amplitudes: np.ndarray,
    times: np.ndarray,
) -> None:
    if not isinstance(initial, Qobj):
        raise TypeError(f"initial must be a Qobj, got {type(initial)}")
    _validate_control_amplitudes(H_controls, control_amplitudes, times)
