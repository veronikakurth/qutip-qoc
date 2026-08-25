"""User-facing constructors for controlled quantum systems."""

from qutip import Qobj
from qutip.typing import QobjEvoLike

from .base import System
from .closed import ClosedSystem
from .open import OpenSystem


class ControlledSystem:
    """Entry point for building a controlled quantum system.
    
    Pick the constructor that matches your physics::

        ControlledSystem.closed(H0, H_controls) -> ClosedSystem
        ControlledSystem.open(H0, H_controls, c_ops) -> OpenSystem
        ControlledSystem.open_liouvillian(L0, H_controls) -> OpenSystem
    """

    def __init__(self):
        raise TypeError(
            "ControlledSystem is a namespace of constructors and is not "
            "instantiable. Use ControlledSystem.closed(...), .open(...) or "
            ".open_liouvillian(...), or instantiate a System subclass directly."
        )

    @staticmethod
    def closed(H0: Qobj, H_controls: list[Qobj]) -> System:
        """Closed system with dynamics ``H(t) = H0 + sum_k u_k(t) * H_k``."""
        return ClosedSystem(H0=H0, H_controls=H_controls)

    @staticmethod
    def open(
        H0: Qobj,
        H_controls: list[Qobj],
        c_ops: QobjEvoLike | list[QobjEvoLike] | None = None,
    ) -> System:
        """Open (Lindbladian) system with optional collapse operators ``c_ops``."""
        return OpenSystem(H0=H0, H_controls=H_controls, c_ops=c_ops)

    @staticmethod
    def open_liouvillian(L0: QobjEvoLike, H_controls: list[Qobj]) -> System:
        """Open system from a pre-assembled drift Liouvillian ``L0``.

        Alternative to `open`: instead of drift + collapse operators, the
        caller supplies the drift Liouvillian directly. This mirrors how
        qutip's ``mesolve`` accepts a Liouvillian in place of Hamiltonians.
        ``H_controls`` may be operators or superoperators.
        """
        return OpenSystem.from_liouvillian(L0=L0, H_controls=H_controls)
