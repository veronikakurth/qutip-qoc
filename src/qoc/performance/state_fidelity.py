from qutip import Qobj, fidelity

from .base import PerformanceMeasure


class StateFidelity(PerformanceMeasure):
    """State fidelity: 1 = perfect match, 0 = orthogonal."""

    def compute(self, current: Qobj, target: Qobj) -> float:
        _validate_states(current, target)
        return fidelity(current, target)



def is_density_matrix(obj: Qobj, tol: float = 1e-10) -> bool:
    return obj.isoper and obj.isherm and abs(obj.tr() - 1) < tol


def _validate_states(current: Qobj, target: Qobj) -> None:
    for name, obj in [("current", current), ("target", target)]:
        if not isinstance(obj, Qobj):
            raise TypeError(f"{name} must be a Qobj, got {type(obj)}")
        if not obj.isket and not is_density_matrix(obj):
                raise ValueError(f"{name} must be a ket or density matrix")

    if current.isket != target.isket:
        raise ValueError(
            "current and target must both be kets or both be density matrices"
        )

    if current.dims != target.dims:
        raise ValueError(
            f"current and target dimensions do not match: {current.dims} vs {target.dims}"
        )
