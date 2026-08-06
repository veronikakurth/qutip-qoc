import numpy as np
from qutip import Qobj, QobjEvo, liouvillian, operator_to_vector, vector_to_operator

from .base import System, StateType, ClassVar

from qutip.typing import QobjEvoLike


class OpenSystem(System):
    """
    Open quantum system evolving under a Lindblad master equation.

    Liouville space, dim n^2

    There are two equivalent ways to build one:

    * **component form** (``__init__``): a drift ``H0``, control Hamiltonians
      ``H_controls`` and collapse operators ``c_ops``. Internally, the generator is
      assembled ``L0 = liouvillian(H0, c_ops)`` with each control
      converted to a Liouvillian via ``liouvillian(H_k)``.
    * **pre-assembled form** (``from_liouvillian``): a drift Liouvillian ``L0``
      (a superoperator) plus controls, mirroring how qutip's ``mesolve`` accepts
      a Liouvillian directly in place of a Hamiltonian.

    Both forms are mapped to the *same* internal representation: a drift
    superoperator ``_L0`` and control superoperators ``_L_controls``
    
    Note on empty ``c_ops``:
    We consider an open system with empty ``c_ops`` a valid case.
    For purely coherent evolution of a pure state, however,
    ``ClosedSystem`` should be preferred.
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
        self._L0 = liouvillian(H0, c_ops)
        self._L_controls = [liouvillian(H_k) for H_k in H_controls]

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
            be already folded in, so no ``c_ops`` are taken separately.
        H_controls : list[Qobj]
            Control terms. Plain operators are lifted with ``liouvillian``;
            terms that are already superoperators are used as given (e.g. a
            control that modulates a dissipation rate rather than a coherent
            term).
        """
        if not (isinstance(L0, (Qobj, QobjEvo)) and L0.issuper):
            raise TypeError("L0 must be a superoperator (Liouvillian)")

        obj = cls.__new__(cls)  # bypass component-form validation in __init__
        obj._H0 = None
        obj._H_controls = list(H_controls)
        obj._c_ops = []
        obj._L0 = L0
        obj._L_controls = [
            H_k if H_k.issuper else liouvillian(H_k) for H_k in H_controls
        ]
        return obj
    # System representation

    def encode_state(state: Qobj) -> np.ndarray:
        return operator_to_vector(state).full() # shape (n**2, 1)

    def decode_state(arr: np.ndarray) -> Qobj:
        return vector_to_operator(Qobj(arr, dims=self.dims)) # shape (n,n) (DM)

    def control_generators():
        return self._L_controls

    @property
    def c_ops(self) -> list[Qobj | QobjEvo]:
        """Collapse operators (component form). Empty for the Liouvillian form,
        where dissipation is already folded into the drift Liouvillian."""
        return self._c_ops

    def build_generator(self, u: np.ndarray) -> list:
        """Time-dependent Liouvillian in qutip's nested-list (superoperator) form.
        """
        L = [self._L0]
        for k in range(self.n_controls):
            L.append([self._L_controls[k], u[k]])
        return L

    # Currently, only used in GRAPE
    def build_generator_time_j(self, u: np.ndarray, j: int) -> Qobj:
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
