from .base import System, StateType
from .closed import ClosedSystem
from .open import OpenSystem
from qutip import Qobj
from qutip.typing import QobjEvoLike

# TODO: Do we also want to give possibility to define custom model in addition to working via pre-defined class methods? Currently, it would be possible via __init__, but this is for advanced usage.

# Cost of maintenance: update the class every time
# 1) new public attributes are added to System hierarchy
# 2) add/update classmethods (in case of new System subclasses) or if anything changes in the constructor signature 
# Facade class
class ControlledSystem:
    """User-facing class to a controlled quantum system.
    The creation of the system of the required model (open or closed or other)
    is controlled via class methods: ``closed``, ``open``.
    """

    def __init__(self, model: System):
        # Prefer the typed constructors below over calling __init__ directly.
        # Calling __init__ is reserved for power usage cases when a user interacts with internal System hierarchy (which implies custom class implementation)
        self._model = model

    @classmethod
    def closed(cls, H0: Qobj, H_controls: list[Qobj]) -> "ControlledSystem":
        """Closed system with dynamics H(t) = H0 + sum_k u_k(t) * H_k."""
        return cls(ClosedSystem(H0=H0, H_controls=H_controls))

    @classmethod
    def open(
        cls,
        H0: Qobj,
        H_controls: list[Qobj],
        c_ops: QobjEvoLike | list[QobjEvoLike] | None = None,
    ) -> "ControlledSystem":
        """Open (Lindbladian) system with optional collapse operators ``c_ops``."""
        return cls(OpenSystem(H0=H0, H_controls=H_controls, c_ops=c_ops))

    @classmethod
    def open_liouvillian(
        cls,
        L0: QobjEvoLike,
        H_controls: list[Qobj],
    ) -> "ControlledSystem":
        """Open system from a pre-assembled drift Liouvillian ``L0``.

        Alternative to `open` instead of drift + collapse operators, the
        caller supplies the drift Liouvillian directly.
        This mirrors how qutip's ``mesolve`` accepts a
        Liouvillian in place of Hamiltonians. 
        ``H_controls`` may be operators or superoperators.
        """
        return cls(OpenSystem.from_liouvillian(L0=L0, H_controls=H_controls))

    # --- bringing System interface to the user ---
    
    @property
    def model(self) -> System:
        return self._model

    @property
    def state_type(self) -> StateType:
        return self._model.state_type

    @property
    def n_controls(self) -> int:
        return self._model.n_controls

    @property
    def dims(self):
        return self._model.dims

    @property
    def shape(self):
        return self._model.shape

    @property
    def drift(self) -> Qobj:
        return self._model.drift

    @property
    def controls(self) -> list[Qobj]:
        return self._model.controls

    def build_generator(self, control_amplitudes) -> list:
        return self._model.build_generator(control_amplitudes)

    def build_generator_time_j(self, control_amplitudes, j: int):
        return self._model.build_generator_time_j(control_amplitudes, j)
    
    # TODO: will be needed for 
    # def default_simulator(self):
    #     """Return the Simulator appropriate for this system's dynamics."""
    #     # Import lazily to avoid an import cycle
    #     # between systems and dynamics modules
    #     from qoc.dynamics.simulator import default_simulator_for

    #     return default_simulator_for(self._model)

    def __repr__(self) -> str:
        kind = type(self._model).__name__
        return (
            f"ControlledSystem({kind}, dims={self.dims}, "
            f"n_controls={self.n_controls})"
        )
    # Some properties are not universal across System subclasses, hence a fallback to a default value
    @property 
    def c_ops(self) -> list:
        return getattr(self._model, "c_ops")
