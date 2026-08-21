from abc import ABC, abstractmethod
from inspect import signature

import numpy as np
from qutip import Qobj


class PerformanceMeasure(ABC):
    """
    Defines *what* the optimizer minimizes for a given objective.

    The primitive is the **loss**, not a fidelity. A measure owns both the loss
    and its gradient, so the two cannot drift apart (see
    ``docs/adr/0002-performance-measure-owns-the-loss.md``).

    Contract
    --------
    ``loss(current, target)``
        A real scalar that is **minimized**, ``0.0`` at a perfect match and
        larger the worse ``current`` is. Nothing else is assumed: it need not
        be bounded above, and it need not be ``1 - fidelity``. A distance is a
        perfectly good loss.

    ``loss_gradient(current, target)``
        d(loss)/d(current), returned in the *same encoding* as ``current`` (a
        ket for closed systems, a vectorized density matrix for open ones, an
        operator for gate synthesis). Formally the Qobj ``G`` for which
        ``d(loss) = Re<G, d(current)>`` under the Hilbert-Schmidt inner
        product. Optional: gradient-free algorithms never call it.

    ``fidelity(current, target)``
        Reporting only, never optimized against. Defaults to ``1 - loss``,
        which is right whenever the loss is an infidelity; override it when it
        is not.
    """

    @abstractmethod
    def loss(self, current: Qobj, target: Qobj) -> float:
        """Scalar to minimize; 0.0 at a perfect match."""

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """d(loss)/d(current), in the same encoding as ``current``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement loss_gradient, so it "
            f"cannot be used with a gradient-based algorithm such as GRAPE. "
            f"Implement loss_gradient, or use a gradient-free algorithm."
        )

    def fidelity(self, current: Qobj, target: Qobj) -> float:
        """Figure of merit for reporting: 1 = perfect match."""
        return 1.0 - self.loss(current, target)


# TODO: think of the way not to use the class by users and operate on functions instead
class FunctionalPerformanceMeasure(PerformanceMeasure):
    """
    Wraps a user-supplied callable as a PerformanceMeasure.

    The escape hatch for custom scoring without subclassing. Two flavours,
    because the sign convention has to be explicit — leaving it to the
    subclass is what produced the bug recorded in ADR 0002:

    * ``FunctionalPerformanceMeasure(func)`` — ``func`` is a **figure of
      merit**: 1 = perfect, higher is better. The loss is ``1 - func(...)``.
    * ``FunctionalPerformanceMeasure.from_loss(func)`` — ``func`` **is** the
      loss: 0 = perfect, lower is better. Use this for distances.

    Neither supplies a gradient, so both are for gradient-free algorithms only.
    """

    def __init__(self, func, *, _is_loss: bool = False):
        _validate_func(func)
        self.func = func
        self._is_loss = _is_loss

    @classmethod
    def from_loss(cls, func) -> "FunctionalPerformanceMeasure":
        """Wrap a callable that already returns a loss (0 = perfect match)."""
        return cls(func, _is_loss=True)

    def loss(self, current: Qobj, target: Qobj) -> float:
        value = _validated_scalar(self.func(current, target))
        return value if self._is_loss else 1.0 - value


def _validated_scalar(result) -> float:
    if not np.isscalar(result):
        raise TypeError(
            f"Performance measure must return a scalar, got {type(result)}"
        )
    if not np.isfinite(result):
        raise ValueError(
            f"Performance measure returned a non-finite value: {result}"
        )
    return float(result)


def _validate_func(func) -> None:
    try:
        sig = signature(func)
    except (TypeError, ValueError):
        return  # builtins / C callables have no introspectable signature
    if any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()):
        return  # *args accepts any arity
    if len(sig.parameters) < 2:
        raise TypeError(
            "Performance measure function must accept at least (current, target)"
        )
