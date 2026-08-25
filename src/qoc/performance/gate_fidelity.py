import numpy as np
from qutip import Qobj, qeye

from .base import PerformanceMeasure


class GateFidelity(PerformanceMeasure):
    """Average gate infidelity for a unitary gate.

    For unitaries ``U`` and target ``U_t`` in dimension ``d``, with
    ``g = tr(U_t^dag U)``::

        F_avg = (|g|**2 / d + 1) / (d + 1)
        loss  = 1 - F_avg

    This is the closed form of ``qutip.average_gate_fidelity`` restricted to
    unitary channels. It is used directly rather than via qutip so that the
    loss and its gradient come from one expression; the two agree to machine
    precision.
    """

    def loss(self, current: Qobj, target: Qobj) -> float:
        return 1.0 - self.fidelity(current, target)

    def fidelity(self, current: Qobj, target: Qobj) -> float:
        _validate_operators(current, target)
        d = current.shape[0]
        g = complex(_overlap(target, current))
        return float((np.abs(g) ** 2 / d + 1.0) / (d + 1.0))

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """d(loss)/d(U).

        loss = 1 - (|g|**2 / d + 1) / (d + 1) with g = tr(U_t^dag U), so
        d(loss) = -2 Re(conj(g) <U_t, dU>) / (d(d+1)) = Re<G, dU> with
        G = -2 g U_t / (d(d+1)).
        """
        _validate_operators(current, target)
        d = current.shape[0]
        g = complex(_overlap(target, current))
        return -2.0 * g * target / (d * (d + 1.0))


def _overlap(target: Qobj, current: Qobj) -> complex:
    """tr(U_t^dag U), the Hilbert-Schmidt overlap of the two gates."""
    return (target.dag() @ current).tr()


def is_unitary(obj: Qobj, tol: float = 1e-10) -> bool:
    I = qeye(obj.shape[0])
    return (obj.dag() * obj - I).norm() < tol


def _validate_operators(current: Qobj, target: Qobj) -> None:
    for name, obj in [("current", current), ("target", target)]:
        if not isinstance(obj, Qobj):
            raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
        if not obj.isoper:
            raise ValueError(f"{name} must be an operator, got type '{obj.type}'")
        if not is_unitary(obj):
            raise ValueError(f"{name} must be a unitary operator")

    if current.dims != target.dims:
        raise ValueError(
            f"current and target dimensions do not match: {current.dims} vs {target.dims}"
        )
