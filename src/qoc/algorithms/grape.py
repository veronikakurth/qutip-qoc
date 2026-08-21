import numpy as np
from typing import Union
from qutip import Qobj, basis, sigmax, sigmaz
from scipy.linalg import expm, expm_frechet
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


# Dev note: we pass system here since we need its motion generator method
def _step_propagators(system: System, u: np.ndarray, dt: float) -> tuple[list[Qobj], list[list[Qobj]]]:
    """Calculate per-step unitaries U_j = expm(prefactor * H_j * dt).
       Prefactor term depends on the system type"""
    # Infer number of time steps from control amplitudes shape
    steps_n = u.shape[1]
    controls = system.control_generators()
    K = len(controls)

    propagators = [None] * steps_n
    propagators_du = [[None] * K for _ in range(steps_n)]
    for j in range(steps_n):
        # Get a motion generator for time j (time-dependent Hamiltonian/Liouvillian with pulse and prefactor term)
        G_j = system.motion_generator_time_j(u, j)
        A = (G_j * dt).full()
        U = expm(A)
        propagators[j] = system.decode_operator(U)
        for k, G_k in enumerate(controls):
            dA = (G_k * dt).full()
            _, dU = expm_frechet(A, dA)
            propagators_du[j][k] = system.decode_operator(dU)
    return propagators, propagators_du

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
            Us, dUs = _step_propagators(system, u, dt) # -> list[Qobj]
            # Compute forward pass (starting with objective's initial state - pre-encoded)
            forward_evolution = self._forward_pass(Us, initial_encoded)
            co_states = self._backward_pass(Us, target_encoded)
            loss = objective.loss(forward_evolution[-1], target_encoded)
            grads = self._gradient(forward_evolution, co_states, controls_encoded, N, dt, dUs, target_encoded)
            
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
        final_state = self._forward_pass(_step_propagators(system, optimized_pulses, dt)[0], initial_encoded)[-1]
        return Result(
            optimized_pulses=optimized_pulses,
            fidelity=1 - opt_result.fun,
            n_iters=opt_result.nit,
            optimizer_info=opt_result,
            history=fidelity_history,
            final_state=system.decode_state(final_state)
        )
    # TODO: adjust signatures: a lot of data comes into the methods encoded w.r.t. system type
    def _forward_pass(self, propagators: list[Qobj], initial_state: Qobj) -> list[Qobj]:
        # Start with initial state and compute propagated state for every time step
        # using previously computed propagators
        N = len(propagators)
        forward_evolution = [None] * (N + 1) # +1 since we include the initial state
        forward_evolution[0] = initial_state
        for j, U in enumerate(propagators):
            forward_evolution[j + 1] = U @ forward_evolution[j]
        return forward_evolution

    def _backward_pass(self, propagators: list[Qobj], target_state: Qobj):
        # Based on previously computed step propagators and target state,
        # go "back in time" by computing co-states from target to the beginning
        # of the evolution
        N = len(propagators)
        co_states = [None] * (N + 1) # Plus one since we include target state
        co_states[N] = target_state
        for j in reversed(range(N)):
            co_states[j] = propagators[j].dag() @ co_states[j + 1]
        return co_states
    
    # TODO: ensure gradient formula is correct with respect to 1) task 2) chosen quality function (~ fidelity)

    def open_gradient(self, co_state, dU, forward_state, final_overlap):
        dc = co_state.dag() @ dU @ forward_state
        return float(np.real(dc))

    def closed_gradient(self, co_state, dU, forward_state, final_overlap):
        dc = co_state.dag() @ dU @ forward_state
        gradient = - 2 * np.real(np.conj(final_overlap) * dc)
        return gradient

    def _gradient(self, forward_evolution: list[Qobj], co_states: list[Qobj], controls: list[Qobj], N: int, dt: float, dUs: list[Qobj], target: Qobj) -> np.ndarray:
        """Adjoint-method gradient. Requires forward + backward passes."""
        K = len(controls)
        grad_func = self.open_gradient if target.type == "operator-ket" else self.closed_gradient 
        # Compute scalar product between last co-state and last evolved state
        grads = np.zeros((K, N), dtype=float)
        final_overlap = target.dag() @ forward_evolution[-1]
        for j in range(N):
            forward_state = forward_evolution[j]
            co_state = co_states[j + 1]
            for k, Gc in enumerate(controls):
                dU = dUs[j][k]
                gradient = grad_func(co_state, dU, forward_state, final_overlap)
                grads[k, j] = gradient
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
