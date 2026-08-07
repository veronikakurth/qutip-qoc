import numpy as np
from typing import Union
from qutip import Qobj, basis, sigmax, sigmaz
from scipy.linalg import expm
from dataclasses import dataclass, field, fields

from qoc.algorithms.base import Algorithm
from qoc.algorithms.result import Result
from qoc.objectives.state_transfer import StateTransfer
from qoc.optimizers.base import Optimizer, ScipyLBFGS
from qoc.problem import OptimalControlProblem
from qoc.pulse.base import PulseParameterization
from qoc.pulse.parameterizations import PiecewiseConstant
from qoc.systems.base import System
from qoc.systems.controlled_system import ControlledSystem


def adj(A):
    return np.conj(A).T


# Dev note: we pass system here since we need its motion generator method
def _step_propagators(system: System, u: np.ndarray, dt: float) -> list[np.ndarray]:
    """Calculate per-step unitaries U_j = expm(prefactor * H_j * dt).
       Prefactor term depends on the system type"""
    # Infer number of time steps from control amplitudes shape
    steps_n = u.shape[1]
    propagators = [None] * steps_n
    for j in range(steps_n):
        # Get a motion generator for time j (time-dependent Hamiltonian/Liouvillian with pulse and prefactor term)
        G_j = system.motion_generator_time_j(u, j)
        # Compute matrix exponential
        propagators[j] = expm(system.encode_operator(G_j) * dt)
    return propagators

# TODO: relocate it onto optimizer later
@dataclass
class OptimizerParams:
    max_iter: int = 100
    tol: float = 1e-8
    extra: dict = field(default_factory=dict) # optimizer-specific options, forwarded as **kwargs
    # maps to scipy options

    @classmethod
    def from_dict(cls, d: dict | None) -> "OptimizerParams":
        # Map known keys to fields, everything else goes into `extra`
        d = dict(d or {})
        known = {f.name for f in fields(cls)} - {"extra"}
        extra = {k: d.pop(k) for k in list(d) if k not in known}
        return cls(**d, extra=extra)


# The procedure must be compatible with both state transfer and gate synthesis tasks
class GRAPE(Algorithm):

    def __init__(
        self,
        parameterization: PulseParameterization,
        optimizer: Optimizer | None = None,
        optimizer_params: dict | None = None,
    ):
        super().__init__()
        if not isinstance(parameterization, PiecewiseConstant):
            raise NotImplementedError(
                f"GRAPE currently supports only parameterization of type PiecewiseConstant, "
                f"got {type(param).__name__}"
            )
        self.parameterization = parameterization
        # Set optimizer or instantiate a default one (LBFGS by SciPy)
        self.optimizer = optimizer or ScipyLBFGS()
        self.optimizer_params = OptimizerParams.from_dict(optimizer_params)

    def build_loss_and_grad(self, problem: OptimalControlProblem, fidelity_history: list | None = None):
        """Build the GRAPE objective closure `theta -> (loss, grad)`.

        Returned standalone so it can be reused outside `solve`. For example, 
        for gradient checks (`scipy.optimize.check_grad`, `approx_fprime`), 
        plotting the loss landscape, or driving a custom optimizer loop. `solve` is just one caller.

        If `fidelity_history` is passed, `1 - loss` is appended on every call.
        """
        system = problem.system.model
        objective = problem.objective
        # Encode boundary conditions from Qobj into numerical representation (np.ndarray)
        initial_encoded = system.encode_state(objective.initial)
        target_encoded = system.encode_state(objective.target)
        controls_encoded = [system.encode_operator(g) for g in system.control_generators()]
        # Extract info from parameterization needed for algorithm
        dt = self.parameterization.dt
        N = self.parameterization.n_timesteps

        def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            u = self.parameterization.to_amplitudes(theta)
            # GRAPE-specific steps
            # Compute propagators for every time step
            Us = _step_propagators(system, u, dt) # -> list[Qobj]
            # Compute forward pass (starting with objective's initial state - pre-encoded)
            forward_evolution = self._forward_pass(Us, initial_encoded)
            co_states = self._backward_pass(Us, target_encoded)
            # A slight inconsistency: here, we operate on Qobj since Objective.loss is defined on it
            # As long as performance is not a problem we leave it this way
            loss = objective.loss(system.decode_state(forward_evolution[-1]))
            grads = self._gradient(forward_evolution, co_states, controls_encoded, N, dt)

            if fidelity_history is not None:
                fidelity_history.append(1 - loss)

            # For PiecewiseConstant, dL/dtheta == dL/du flattened (Jacobian is identity).
            return loss, grads.ravel()

        return loss_and_grad

    def solve(self, problem: OptimalControlProblem, initial_param_values, callback=None) -> Result:
        """Main entry point."""
        system = problem.system.model
        # Initial values of parameter vector
        theta0 = self.parameterization.initial_theta(initial_param_values)

        fidelity_history = []
        loss_and_grad = self.build_loss_and_grad(problem, fidelity_history)

        opt_result = self.optimizer.minimize(
            loss_and_grad,
            x0=theta0,
            max_iter=self.optimizer_params.max_iter,
            tol=self.optimizer_params.tol,
            #bounds=self.parameterization.bounds(),
            **self.optimizer_params.extra
        )


        if not opt_result.success:
            print(
                f"Warning: optimiser {self.optimizer} used in GRAPE did not converge. \n"
                f" Detailed optimiser report: {opt_result}"
            )

        optimized_pulses = self.parameterization.to_amplitudes(opt_result.x)
        # TODO: this is neither efficient nor needed for a user in the result; more debugging data - what would be a good way to get it if required?
        # Maybe some kind of meta algorithm data
        dt = self.parameterization.dt
        initial_encoded = system.encode_state(problem.objective.initial)
        final_state = self._forward_pass(_step_propagators(system, optimized_pulses, dt), initial_encoded)[-1]
        return Result(
            optimized_pulses=optimized_pulses,
            fidelity=1 - opt_result.fun,
            n_iters=opt_result.nit,
            optimizer_info=opt_result,
            history=fidelity_history,
            final_state=system.decode_state(final_state)
        )
    # TODO: adjust signatures: a lot of data comes into the methods encoded w.r.t. system type
    def _forward_pass(self, propagators: list[np.ndarray], initial_state: np.ndarray) -> list[np.ndarray]:
        # Start with initial state and compute propagated state for every time step
        # using previously computed propagators
        # Important! Initial state comes encoded already
        N = len(propagators)
        forward_evolution = [None] * (N + 1) # +1 since we include the initial state
        forward_evolution[0] = initial_state
        for j, U in enumerate(propagators):
            forward_evolution[j + 1] = U @ forward_evolution[j]
        return forward_evolution

    def _backward_pass(self, propagators: list[np.ndarray], target_state: np.ndarray):
        # Based on previously computed step propagators and target state,
        # go "back in time" by computing co-states from target to the beginning
        # of the evolution
        N = len(propagators)
        co_states = [None] * (N + 1) # Plus one since we include target state
        co_states[N] = target_state
        for j in reversed(range(N)):
            co_states[j] = adj(propagators[j]) @ co_states[j + 1]
        return co_states
    
    # TODO: make signature more specific
    # TODO: try to make use of more Qobj specific functions to become more agnostic of backend
    def _gradient(self, forward_evolution: list[np.ndarray], co_states: list[np.ndarray], controls: list[np.ndarray], N: int, dt: float) -> np.ndarray:
        """Adjoint-method gradient. Requires forward + backward passes."""
        K = len(controls)
        # Compute scalar product between last co-state and last evolved state
        c = (adj(co_states[-1]) @ forward_evolution[-1]).item()  # <phi|psi_N>
        grads = np.zeros((K, N), dtype=float)
        for j in range(N):
            psi_j = forward_evolution[j]
            co_state_jp1 = co_states[j + 1]
            for k, Hc in enumerate(controls):
                # Scalar product between current co state and current evolved state
                inner = (adj(co_state_jp1) @ Hc @ psi_j).item()
                # TODO: for overlap, ensure the formula is right for both system type -> infer it from PM/Objective?
                grads[k, j] = -2.0 * dt * np.imag(np.conj(c) * inner)
        return grads


def define_problem():
    H0 = 0 * sigmaz()
    H_c = [sigmax() / 2, sigmax() / 2]
    system = ControlledSystem.closed(H0=H0, H_controls=H_c)

    T = 10
    N = 10
    times = np.linspace(0, T, N)
    K = system.n_controls

    param = PiecewiseConstant(K, times=times)

    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    objective = StateTransfer(initial_state, target_state)
    return param, OptimalControlProblem(system, objective)
