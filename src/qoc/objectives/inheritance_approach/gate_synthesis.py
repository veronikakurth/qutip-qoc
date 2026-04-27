from qutip import Qobj, qeye, average_gate_fidelity
from base import Objective

class GateFidelityObjective(Objective):
    """
    Using fidelity of a quantum gate as a measure of closeness to the target.

    Validation:
    Current (achieved) and target gates must be valid unitary operators and have matching dimensions.
    """

    def __init__(self, target: Qobj = None):
        if not target:
            pass
        super().__init__(target)

    def compute(self, current):
        _validate_operators(current, self.target)
        return average_gate_fidelity(current, self.target)

def is_unitary(obj, tol=1e-10):
    I = qeye(obj.shape[0])
    return (obj.dag() * obj - I).norm() < tol

def _validate_operators(current: Qobj, target: Qobj) -> None:
    for name, obj in [("current", current), ("target", target)]:
        if not isinstance(obj, Qobj):
            raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
        if not obj.isoper:
            raise ValueError(f"{name} must be an operator (square matrix), got type {obj.type}")
        if not is_unitary(obj):
            raise ValueError(f"Operator must be unitary")

    if current.dims != target.dims:
        raise ValueError(
            f"current and target must have the same dimensions, "
            f"got {current.dims} and {target.dims}"
        )
