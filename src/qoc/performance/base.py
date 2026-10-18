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
        A real scalar that is **minimized**. Its scale, offset and minimum
        are defined by the measure.

    ``loss_gradient(current, target)``
        d(loss)/d(current): the ``G`` with ``d(loss) = Re<G, d(current)>`` under
        the Hilbert-Schmidt inner product ``<A, B> = tr(A^dag B)``, returned in
        the same *physical* representation as ``current`` (a ket, a density-matrix
        shaped operator, or an operator for gate synthesis). It need not be
        Hermitian or normalized.

    ``report(current, target)``
        Optional named scalar metrics for reporting only. Defaults to an empty
        dictionary; no fidelity or normalization is inferred from the loss.

    ``accepts``
        The kinds of physical object the measure is defined on, as a frozenset
        of ``"ket"``, ``"dm"``, ``"unitary"``; ``None`` if undeclared. A measure
        that declares it is checked against the objective's semantic domain
        at construction via ``check_domain``. ``_check_kind`` additionally
        checks Qobj representation at evaluation time.
    """

    accepts: frozenset[Kind] | None = None

    @abstractmethod
    def loss(self, current: Qobj, target: Qobj) -> float:
        """Scalar to minimize, with measure-specific scale and offset."""

    def loss_gradient(self, current: Qobj, target: Qobj) -> Qobj:
        """d(loss)/d(current), in the same physical representation as ``current``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement loss_gradient, so it "
            f"cannot be used with a gradient-based algorithm such as GRAPE. "
            f"Implement loss_gradient, or use a gradient-free algorithm."
        )

    def report(self, current: Qobj, target: Qobj) -> dict[str, float]:
        """Named reporting metrics; independent of the optimization contract."""
        return {}

    def check_domain(self, kind: Kind) -> None:
        """Check the objective's semantic domain without evaluating the measure.

        Objectives supply their validated domain: ket, density matrix or
        unitary. Undeclared domains (accepts=None) permit custom measures.
        """
        if self.accepts is None:
            return
        if kind not in self.accepts:
            supported = ", ".join(sorted(self.accepts))
            raise TypeError(
                f"{type(self).__name__} accepts [{supported}], "
                f"but the objective uses {kind!r}."
            )

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

    def report(self, current: Qobj, target: Qobj) -> dict[str, float]:
        return {"fidelity": self.fidelity(current, target)}

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
      loss: lower is better; no normalization is assumed.

    Both take an optional ``gradient(current, target) -> Qobj``, required by
    gradient-based algorithms such as GRAPE. It is the gradient of ``func``
    itself, whichever of the two ``func`` is: the ``G`` with
    ``d(func) = Re<G, d(current)>`` under ``<A, B> = tr(A^dag B)``, in the same
    physical representation as ``current``. For a figure of merit the loss
    gradient is therefore ``-G``.

    Both accept an optional ``report(current, target) -> dict[str, float]``
    callable for named finite scalar metrics. Without it, reporting is empty;
    even a figure of merit is not implicitly labelled fidelity.

    Both also take an optional ``accepts``: a kind (``"ket"``, ``"dm"``,
    ``"unitary"``) or several. When given, ``current`` is checked against it
    before ``func`` or ``gradient`` runs; when omitted, nothing is checked.
    """

    def __init__(self, func, *, gradient=None, accepts=None, report=None, _is_loss: bool = False):
        _validate_func(func, "Performance measure function")
        if gradient is not None:
            _validate_func(gradient, "Performance measure gradient")
        if report is not None:
            _validate_func(report, "Performance measure report")
        self._report = report
        self.func = func
        self.gradient = gradient
        self.accepts = _normalized_kinds(accepts)
        self._is_loss = _is_loss

    @classmethod
    def from_loss(cls, func, *, gradient=None, accepts=None, report=None) -> "FunctionalPerformanceMeasure":
        """Wrap a callable that already returns a loss to minimize."""
        return cls(func, gradient=gradient, accepts=accepts, report=report, _is_loss=True)

    def report(self, current: Qobj, target: Qobj) -> dict[str, float]:
        self._check_kind(current)
        if self._report is None:
            return {}
        metrics = self._report(current, target)
        if not isinstance(metrics, dict) or not all(isinstance(k, str) for k in metrics):
            raise TypeError("Performance measure report must return a dict with string keys")
        return {name: _validated_scalar(value) for name, value in metrics.items()}

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
    if not np.isscalar(result) or np.iscomplexobj(result):
        raise TypeError(
            f"Performance measure must return a real scalar, got {type(result)}"
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
