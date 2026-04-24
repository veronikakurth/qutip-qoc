from qutip import Qobj, fidelity
from utility_functions import is_density_matrix

def state_transfer_fidelity(current, target):
    # Can be applied to states and density matrices
    _validate_states(current, target)

    if current.isket:
        overlap = target.dag() * current
        return abs(overlap) ** 2
    else:
        # Uhlmann fidelity for density matrices
        return fidelity(current, target) ** 2


def _validate_states(current: Qobj, target: Qobj) -> None:
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
