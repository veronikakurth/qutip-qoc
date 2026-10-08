from qutip import Qobj


def state_type(obj: Qobj, tol: float = 1e-8) -> str:
    """Classify a physical state as a ket or density matrix"""
    if not isinstance(obj, Qobj):
        raise TypeError(f"Expected a Qobj, got {type(obj)}.")

    if obj.isket:
        if abs(obj.norm() - 1) > tol:
            raise ValueError(f"Ket must be normalized, got norm {obj.norm():.6g}")
        return "ket"

    if obj.isoper and obj.isherm:
        if abs(obj.tr() - 1) > tol:
            raise ValueError(f"Density matrix must have trace 1, got {obj.tr():.6g}")
        if obj.eigenenergies().min() < -tol:
            raise ValueError("Density matrix must be positive semidefinite")
        return "dm"

    raise ValueError(
        f"Expected a ket or density matrix, got type={obj.type!r}, dims={obj.dims}"
    )

def validate_states(current: Qobj, target: Qobj) -> None:
    current_type, target_type = state_type(current), state_type(target)

    if current_type != target_type:
        raise ValueError(
            f"initial and target must be of the same type, got {current_type} and {target_type}"
        )
    if current.dims != target.dims:
        raise ValueError(
            f"initial and target dims differ: {current.dims} vs {target.dims}"
        )
