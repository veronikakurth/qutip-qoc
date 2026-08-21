import numpy as np
from qutip import Qobj

from .base import PerformanceMeasure


class StateFidelity(PerformanceMeasure):
    """State infidelity: loss 0 = perfect match, fidelity 1 = perfect match.

    Two functionals, picked by how the state is encoded (see ADR 0001):

    * **closed** (`current` is a ket)::

          F = |<t|psi>|**2

    * **open** (`current` is a vectorized density matrix, type
      ``"operator-ket"``)::

          F = Re tr(rho_t^dag rho) / tr(rho_t**2)

      Normalizing by ``tr(rho_t**2)`` makes F == 1 exactly at ``rho == rho_t``
      for any target, pure or mixed. Without it the "fidelity" is unbounded and
      ``1 - F`` is not an infidelity.

    In both cases ``loss = 1 - F``.
    """

    # --- loss ---

    def loss(self, current: Qobj, target: Qobj) -> float:
        return 1.0 - self.fidelity(current, target)

    def fidelity(self, current: Qobj, target: Qobj) -> float:
        overlap = complex(_overlap(target, current))
        if _is_vectorized(current):
            return float(np.real(overlap) / _target_norm_sq(target))
        return float(np.abs(overlap) ** 2)

    # --- gradient ---

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """d(loss)/d(current). See the class docstring for the two functionals.

        Closed: loss = 1 - |c|**2 with c = <t|psi>, so
        d(loss) = -2 Re(conj(c) <t|dpsi>) = Re<-2 c t, dpsi>.

        Open: loss = 1 - Re<<t|rho>> / s, linear in rho, so
        d(loss) = Re<-t/s, drho>.
        """
        if _is_vectorized(current):
            return -target / _target_norm_sq(target)
        c = complex(_overlap(target, current))
        return -2.0 * c * target


def _is_vectorized(state: Qobj) -> bool:
    """True for an open system's vectorized density matrix."""
    return state.type == "operator-ket"


def _overlap(target: Qobj, current: Qobj) -> Qobj:
    """<t|psi> for kets, tr(rho_t^dag rho) for vectorized density matrices."""
    return target.dag() @ current


def _target_norm_sq(target: Qobj) -> float:
    """tr(rho_t**2); the scale that makes F(rho_t) == 1."""
    norm_sq = float(target.norm()) ** 2
    if norm_sq == 0.0:
        raise ValueError("target state has zero norm; fidelity is undefined")
    return norm_sq
