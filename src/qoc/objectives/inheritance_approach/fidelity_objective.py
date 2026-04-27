from qutip import Qobj, fidelity
from base import Objective

class FidelityObjective(Objective):
    """
    Using fidelity of a quantum state as a measure of closeness to the target.

    Validation:
    Current (achieved) and target quantum states (kets for pure states, density matrices
    for mixed states) must have the same type and matching dimensions.
    """

    def __init__(self, target: Qobj = None):
        # If target is not provided, it must be inferrable from the figure of merit
        if not target: # TODO: decide what to do: do we actually allow not providing it for state transfer? 
            pass
        super().__init__(target)

    def compute(self, current):
        _validate_states(current, self.target)
        return fidelity(current, self.target)

def is_density_matrix(obj, tol=1e-10):
    return (
        obj.isherm and
        abs(obj.tr() - 1) < tol and
        min(obj.eigenenergies()) >= -tol
    )

def _validate_states(current, target: Qobj) -> None:
    for name, obj in [("current", current), ("target", target)]:
        if not isinstance(obj, Qobj):
            raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
        if not obj.isket:
            if not obj.isoper or not is_density_matrix(obj):
                raise ValueError(f"{name} must be a ket or density matrix")

    if current.isket != target.isket:
        raise ValueError("current and target must both be kets or both be density matrices")

    if current.dims != target.dims:
        raise ValueError(
            f"current and target must have the same dimensions, "
            f"got {current.dims} and {target.dims}"
        )
