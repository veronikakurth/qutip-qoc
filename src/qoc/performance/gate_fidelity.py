from qutip import Qobj, qeye, average_gate_fidelity

from .base import PerformanceMeasure


class GateFidelity(PerformanceMeasure):
    """Average gate fidelity: 1 = perfect match."""

    def compute(self, current: Qobj, target: Qobj) -> float:
        _validate_operators(current, target)
        return average_gate_fidelity(current, target)



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
