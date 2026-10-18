from abc import ABC, abstractmethod
from inspect import signature
from typing import Iterable, Literal

import numpy as np
from qutip import Qobj

#: Kind of physical object a measure scores. ``"dm"`` and ``"unitary"`` are both
#: Qobj type ``"oper"``; a measure can tell them apart only by its declaration.
Kind = Literal["ket", "dm", "unitary"]

_KINDS: tuple[Kind, ...] = ("ket", "dm", "unitary")
_QOBJ_TYPE = {"ket": "ket", "dm": "oper", "unitary": "oper"}
_KIND_NAME = {"ket": "ket", "dm": "density matrix", "unitary": "unitary"}


class PerformanceMeasure(ABC):
    """
    Defines *what* the optimizer minimizes for a given objective.

    The primitive is the **loss**, not a fidelity. A measure owns both the loss
    and its gradient.

    A measure works on **physical objects only**: kets, density matrices, or
    operators, exactly as the user wrote ``initial`` and ``target``. It never sees
    a system's encoded representation (e.g. a vectorized density matrix); the
    algorithm encodes and decodes around it (ADR 0004). A formula is therefore
    written once, in physics notation, and works for any system encoding.

    API
    --------
    ``loss(current, target)``
        A real scalar that is **minimized**, ``0.0`` at a perfect match and
        larger the further ``current`` is from the ``target``. 

    ``loss_gradient(current, target)``
        d(loss)/d(current): the ``G`` with ``d(loss) = Re<G, d(current)>`` under
        the Hilbert-Schmidt inner product ``<A, B> = tr(A^dag B)``, returned in
        the same *physical* representation as ``current`` (a ket, a density-matrix
        shaped operator, or an operator for gate synthesis). It need not be
        Hermitian or normalized.

    ``fidelity(current, target)``
        Used for reporting only, is not optimized against. Defaults to ``1 - loss``,
        which is right whenever the loss is an infidelity; override it when it
        is not.

    ``accepts``
        The kinds of physical object the measure is defined on, as a frozenset
        of ``"ket"``, ``"dm"``, ``"unitary"``; ``None`` if undeclared. A measure
        that declares it rejects any other kind via ``_check_kind``.
    """

    accepts: frozenset[Kind] | None = None

    @abstractmethod
    def loss(self, current: Qobj, target: Qobj) -> float:
        """Scalar to minimize; 0.0 at a perfect match."""

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """d(loss)/d(current), in the same physical representation as ``current``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement loss_gradient, so it "
            f"cannot be used with a gradient-based algorithm such as GRAPE. "
            f"Implement loss_gradient, or use a gradient-free algorithm."
        )

    def fidelity(self, current: Qobj, target: Qobj) -> float:
        """Figure of merit for reporting: 1 = perfect match."""
        return 1.0 - self.loss(current, target)

    def _check_kind(self, current: Qobj) -> None:
        """Validate that ``current`` is a kind this measure accepts; no-op if undeclared.

        Checks the Qobj type only, so it cannot tell a density matrix from a
        unitary. Validation, not dispatch: the measure is already chosen.
        """
        if self.accepts is None:
            return
        if current.type not in {_QOBJ_TYPE[kind] for kind in self.accepts}:
            names = " or ".join(_KIND_NAME[k] for k in _KINDS if k in self.accepts)
            raise TypeError(
                f"{type(self).__name__} accepts a {names}, got Qobj type "
                f"{current.type!r}. This usually means the objective's "
                f"performance measure does not match its kind of state."
            )


class FidelityMeasure(PerformanceMeasure):
    """Base for explicit fidelities, with ``loss = 1 - fidelity``.

    Subclasses define the fidelity formula and declare its physical domain
    via ``accepts``. Gradients are optional, just as on PerformanceMeasure:
    gradient-free algorithms need only the fidelity implementation, while
    gradient-based algorithms require ``loss_gradient`` as well.
    """

    def loss(self, current: Qobj, target: Qobj) -> float:
        return 1.0 - self.fidelity(current, target)

    @abstractmethod
    def fidelity(self, current: Qobj, target: Qobj) -> float:
        """Fidelity on the declared domain; 1 at a perfect match."""


# TODO: think of the way not to use the class by users and operate on functions instead
class FunctionalPerformanceMeasure(PerformanceMeasure):
    """
    Wraps a user-supplied callable as a PerformanceMeasure.

    Presents a way to implement a custom scoring without subclassing.

    * ``FunctionalPerformanceMeasure(func)`` — ``func`` is a **figure of
      merit**: 1 = perfect (higher is better). The loss is ``1 - func(...)``.
    * ``FunctionalPerformanceMeasure.from_loss(func)`` — ``func`` **is** the
      loss: 0 = perfect, lower is better.

    Both take an optional ``gradient(current, target) -> Qobj``, required by
    gradient-based algorithms such as GRAPE. It is the gradient of ``func``
    itself, whichever of the two ``func`` is: the ``G`` with
    ``d(func) = Re<G, d(current)>`` under ``<A, B> = tr(A^dag B)``, in the same
    physical representation as ``current``. For a figure of merit the loss
    gradient is therefore ``-G``.

    Both also take an optional ``accepts``: a kind (``"ket"``, ``"dm"``,
    ``"unitary"``) or several. When given, ``current`` is checked against it
    before ``func`` or ``gradient`` runs; when omitted, nothing is checked.
    """

    def __init__(self, func, *, gradient=None, accepts=None, _is_loss: bool = False):
        _validate_func(func, "Performance measure function")
        if gradient is not None:
            _validate_func(gradient, "Performance measure gradient")
        self.func = func
        self.gradient = gradient
        self.accepts = _normalized_kinds(accepts)
        self._is_loss = _is_loss

    @classmethod
    def from_loss(cls, func, *, gradient=None, accepts=None) -> "FunctionalPerformanceMeasure":
        """Wrap a callable that already returns a loss (0 = perfect match)."""
        return cls(func, gradient=gradient, accepts=accepts, _is_loss=True)

    def loss(self, current: Qobj, target: Qobj) -> float:
        self._check_kind(current)
        value = _validated_scalar(self.func(current, target))
        return value if self._is_loss else 1.0 - value

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        if self.gradient is None:
            raise NotImplementedError(
                "This FunctionalPerformanceMeasure has no gradient, so it cannot "
                "be used with a gradient-based algorithm such as GRAPE. Pass "
                "gradient=... when constructing it, or use a gradient-free algorithm."
            )
        self._check_kind(current)
        G = _validated_gradient(self.gradient(current, target), current)
        return G if self._is_loss else -G


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


def _normalized_kinds(accepts: Kind | Iterable[Kind] | None) -> frozenset[Kind] | None:
    if accepts is None:
        return None
    kinds = frozenset([accepts] if isinstance(accepts, str) else accepts)
    unknown = kinds - set(_KINDS)
    if unknown or not kinds:
        raise ValueError(
            f"accepts must be one or more of {_KINDS}, got {accepts!r}"
        )
    return kinds


def _validated_gradient(result, current: Qobj) -> Qobj:
    if not isinstance(result, Qobj):
        raise TypeError(
            f"Performance measure gradient must return a Qobj, got {type(result)}"
        )
    if result.dims != current.dims:
        raise ValueError(
            f"Performance measure gradient must have the dims of current, "
            f"{current.dims}, got {result.dims}"
        )
    if not np.all(np.isfinite(result.full())):
        raise ValueError("Performance measure gradient has non-finite entries")
    return result


def _validate_func(func, name: str) -> None:
    try:
        sig = signature(func)
    except (TypeError, ValueError):
        return  # builtins / C callables have no introspectable signature
    if any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()):
        return  # *args accepts any arity
    if len(sig.parameters) < 2:
        raise TypeError(f"{name} must accept at least (current, target)")
