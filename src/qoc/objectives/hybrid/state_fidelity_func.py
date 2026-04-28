from qutip import Qobj, fidelity

def state_fidelity(current, target):
    # Can be applied to states and density matrices
    _validate_states(current, target)
    return fidelity(current, target)

def is_density_matrix(obj, tol=1e-10):
    return (
        obj.isherm and
        abs(obj.tr() - 1) < tol and
        min(obj.eigenenergies()) >= -tol
    )

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
