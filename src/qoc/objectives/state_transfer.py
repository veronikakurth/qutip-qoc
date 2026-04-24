from qutip import Qobj
from .base import Objective


def state_fidelity(current, target):
    # TODO: what do we do if target is not provided?
    if current.isket:
        # In QuTiP 5, bra * ket returns a complex scalar directly
        overlap = target.dag() * current
        return 1.0 - abs(overlap) ** 2
    else:
        # Uhlmann fidelity for density matrices: Tr[sqrt(sqrt(rho) sigma sqrt(rho))]
        # qutip.fidelity returns F (not F²), so we square it to stay consistent
        # with the pure-state convention above.
        return 1.0 - qutip.fidelity(current, target) ** 2


class StateTransfer(Objective):
    """
    Objective for state-to-state transfer.

    Both initial and target must be quantum states: either kets (pure,
    closed system) or density matrices (mixed, open system). They must
    be the same type and have matching dimensions.
    """

    def __init__(self, initial: Qobj, target: Qobj = None):
        # If target is not provided, it must be inferrable from the figure of merit
        if not target: # TODO: decide what to do: do we actually allow not providing it for state transfer? 
            pass
        _validate_states(initial, target)
        super().__init__(initial, target)
        # TODO: should we try to infer here whether we are dealing with closed or open system?


# Here, trying to infer if input it's a density matrix (-> the objective function has to be adopted for open system case)
def is_density_matrix(obj, tol=1e-10):
    return (
        obj.isherm and
        abs(obj.tr() - 1) < tol and
        min(obj.eigenenergies()) >= -tol
    )

def _validate_states(initial: Qobj, target: Qobj) -> None:
    for name, obj in [("initial", initial), ("target", target)]:
        if not isinstance(obj, Qobj):
            raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
        if not obj.isket:
            if not obj.isoper or not is_density_matrix(obj):
                raise ValueError(f"{name} must be a ket or density matrix")

    if initial.isket != target.isket:
        raise ValueError("initial and target must both be kets or both be density matrices")

    if initial.dims != target.dims:
        raise ValueError(
            f"initial and target must have the same dimensions, "
            f"got {initial.dims} and {target.dims}"
        )
