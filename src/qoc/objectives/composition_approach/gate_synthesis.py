from qutip import average_gate_fidelity, fidelity, Qobj, qeye

def gate_synthesis_fidelity(current, target):
    _validate_operators(current, target)
    return fidelity(current, target) # Might want to use average_gate_fidelity instead (requires conversion to superoperator)

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
