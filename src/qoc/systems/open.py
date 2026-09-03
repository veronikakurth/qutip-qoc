import numpy as np
from qutip import Qobj, QobjEvo, liouvillian, operator_to_vector, vector_to_operator

from .base import System, StateType, ClassVar

from qoc.utils.display import describe_qobj_list

from qutip.typing import QobjEvoLike


class OpenSystem(System):
    """
    Open quantum system evolving under a Lindblad master equation.

    Liouville space, dim n^2

    Two equivalent ways to build an open system are supported:

    * **component form** (``__init__``): a drift ``H0``, control Hamiltonians
      ``H_controls`` and collapse operators ``c_ops``. 
      Internally, the generator will be assembled as
      ``L0 = liouvillian(H0, c_ops)`` with each control
      converted to a Liouvillian via ``liouvillian(H_k)``.
    * **pre-assembled form** (``from_liouvillian``): a drift Liouvillian ``L0``
      (a superoperator) plus controls, mirroring how qutip's ``mesolve`` accepts
      a Liouvillian directly in place of a Hamiltonian.

    Both forms are mapped to the *same* internal representation: a drift
    superoperator ``_L0`` and control superoperators ``_L_controls``
    
    Note on empty ``c_ops``:
    We consider an open system with empty ``c_ops`` a valid case.
    In this case, the system has the equivalent dynamics as its closed counterpart.
    """

    state_type: ClassVar[StateType] = "dm"
    
    def __init__(
        self,
        H0: Qobj,
        H_controls: list[Qobj],
        c_ops: QobjEvoLike | list[QobjEvoLike] | None = None,
    ):
        """
        Parameters
        ----------
        H0 : Qobj
            Drift Hamiltonian.
        H_controls : list[Qobj]
            Control Hamiltonians.
        c_ops: QobjEvoLike | list[QobjEvoLike] | None, optional
            Collapse operators defining dissipation into the environment.
            Optional; defaults to none (see the class note on empty ``c_ops``).
        """
        super().__init__(H0, H_controls)
        c_ops = self._validate_c_ops(c_ops)
        self._c_ops = c_ops
        # Map user-passed arguments to an internal super-operator representation for drift and controls
        self._L0 = liouvillian(H0, c_ops)
        self._L_controls = [liouvillian(H_k) for H_k in H_controls]
    
    def _summary_rows(self) -> list[tuple[str, str]]:
        # The summary reports what the user passed in (H_controls, c_ops); the
        # internal Liouvillians are reachable via ``system.drift`` and
        # ``system.control_generators()``.
        if self._H0 is None:  # built by from_liouvillian: no separate c_ops
            c_ops = "absorbed into L0"
        else:
            c_ops = describe_qobj_list(self._c_ops)
        return super()._summary_rows() + [("c_ops", c_ops)]


    @classmethod
    def from_liouvillian(
        cls,
        L0: QobjEvoLike,
        H_controls: list[Qobj],
    ) -> "OpenSystem":
        """Build an open system from a pre-assembled drift Liouvillian.

        Parameters
        ----------
        L0 : Qobj | QobjEvo
            Drift Liouvillian (a superoperator). Any dissipation is assumed to
            be already captured in it, so no ``c_ops`` are taken separately.
        H_controls : list[Qobj]
            Control terms. Plain operators are promoted via ``liouvillian``;
            terms that are already superoperators are used as given.
        """
        if not (isinstance(L0, (Qobj, QobjEvo)) and L0.issuper):
            raise TypeError("L0 must be a superoperator (Liouvillian)")

        obj = cls.__new__(cls)  # bypass component-form validation in __init__
        # No component-form Hamiltonian exists on this path. Nothing reads
        # _H0 on OpenSystem: drift/dims/shape are derived from _L0 below.
        obj._H0 = None
        obj._H_controls = list(H_controls)
        obj._c_ops = []
        obj._L0 = L0
        obj._L_controls = [
            H_k if H_k.issuper else liouvillian(H_k) for H_k in H_controls
        ]
        return obj

    # System representation (implementing abstract methods from System)
    def encode_state(self, state: Qobj) -> Qobj:
        encoded = operator_to_vector(state) # shape (n**2, 1)
        return encoded

    def decode_state(self, state: Qobj) -> Qobj:
        if state.type != "operator-ket":
            raise TypeError("decode_state expects an operator-ket")
        return vector_to_operator(state)
    
    def decode_operator(self, state: Qobj) -> Qobj:
        space = self._L0.dims[0][0]
        op_dims = [space, space]
        return Qobj(state, dims=[op_dims, op_dims], superrep="super")
    
    def control_generators(self):
        return self._L_controls

    # Derived from _L0, not _H0, so that both construction paths agree.
    # (from_liouvillian has no component-form H0 to fall back on.)

    @property
    def drift(self) -> Qobj:
        """Drift Liouvillian, i.e. the drift term of ``build_generator``."""
        return self._L0

    @property
    def dims(self) -> list:
        # _L0.dims is [hilbert_dims, hilbert_dims]; its first entry is the
        # Hilbert-space dims pair [[n], [n]] we owe callers.
        return self._L0.dims[0]

    @property
    def shape(self) -> tuple[int, int]:
        n = int(np.prod(self._L0.dims[0][0]))
        return (n, n)

    @property
    def c_ops(self) -> list[Qobj | QobjEvo]:
        """Collapse operators (component form). 
        Empty in case the system was initiatilized via ```from_liouvillian``"""
        return self._c_ops
    
    # Is not used in GRAPE, may prove useful in other algorithms
    def build_generator(self, u: np.ndarray) -> list:
        """Time-dependent Liouvillian in qutip's nested-list form (with superoperators).
        """
        L = [self._L0]
        for k in range(self.n_controls):
            L.append([self._L_controls[k], u[k]])
        return L

    def motion_generator_time_j(self, u: np.ndarray, j: int) -> Qobj:
        return self._L0 + sum(
            u[k][j] * self._L_controls[k]
            for k in range(self.n_controls)
        )

    def _validate_c_ops(self, c_ops):
        c_ops = c_ops or []
        c_ops = [c_ops] if isinstance(c_ops, (Qobj, QobjEvo)) else c_ops
        for c_op in c_ops:
            if not isinstance(c_op, (Qobj, QobjEvo)):
                raise TypeError("All `c_ops` must be a Qobj or QobjEvo")
        return c_ops
