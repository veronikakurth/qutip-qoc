import numpy as np
from qutip import Qobj, basis, sigmax, sigmaz
from scipy.linalg import expm

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


def _step_propagators(system: System, control_amplitudes: np.ndarray, dt: float) -> list:
    """Per-slice unitaries U_j = expm(-i H_j dt) for piecewise-constant controls."""
    N = control_amplitudes.shape[1]
    propagators = [None] * N
    for j in range(N):
        H_j = system.build_generator_time_j(control_amplitudes, j)
        # dev note: we always convert generator to dense here... will be an issue for performance
        propagators[j] = expm(-1j * H_j.full() * dt)
        # TODO: adapt this formula to work with closed and open systems universally (infer the factor)
    return propagators


# The procedure must be compatible with both state transfer and gate synthesis tasks
class GRAPE(Algorithm):

    def __init__(
        self,
        parameterization: PulseParameterization,
        optimizer: Optimizer | None = None,
        max_iter: int = 100,
        tol: float = 1e-8,
    ):
        super().__init__(simulator=None)  # GRAPE uses _step_propagators, not a Simulator
        if not isinstance(parameterization, PiecewiseConstant):
            raise NotImplementedError(
                f"GRAPE currently supports only PiecewiseConstant parameterization, "
                f"got {type(param).__name__}"
            )
        self.parameterization = parameterization
        self.optimizer = optimizer or ScipyLBFGS()
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem: OptimalControlProblem, initial_amplitudes=None, callback=None) -> Result:
        """Main entry point."""
        system = problem.system
        # TODO: do encoding of boundary conditions, drift and control generators here
        objective = problem.objective
        dt = self.parameterization.dt
        param = self.parameterization
        N = param.n_timesteps
        theta0 = self.parameterization.initial_theta(initial_amplitudes)
        if theta0.size != param.n_parameters:
            raise ValueError(
                f"initial_parameters has size {theta0.size}, parameterization "
                f"expects {param.n_parameters}"
            )

        history = []

        def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]: # TODO: can I provide final state from forward evolution as a callback
            u = param.to_amplitudes(theta)
            # GRAPE-specific steps
            Us = _step_propagators(system, u, dt)
            forward_evolution = self._forward_pass(Us, initial)
            co_states = self._backward_pass(Us, target)
            loss = objective.loss(Qobj(forward_evolution[-1], dims=initial.dims))
            grads = self._gradient(forward_evolution, co_states, system.controls, N, dt)

            history.append(1 - loss)
            if callback is not None:
                callback(theta, loss)

            # For PiecewiseConstant, dL/dtheta == dL/du flattened (Jacobian is identity).
            return loss, grads.ravel()
        opt_result = self.optimizer.minimize(
            loss_and_grad,
            x0=theta0,
            max_iter=self.max_iter,
            tol=self.tol,
            bounds=param.bounds(),
        )

        if not opt_result.success:
            print(
                f"Warning: optimiser {self.optimizer} used in GRAPE did not converge. \n"
                f" Detailed optimiser report: {opt_result}"
            )

        optimized_pulses = param.to_amplitudes(opt_result.x)
        # TODO: this is neither efficient nor needed for a user in the result; more debugging data - what would be a good way to get it if required?
        # Maybe some kind of meta algorithm data
        final_state = self._forward_pass(_step_propagators(system, optimized_pulses, dt), objective.initial)[-1]
        return Result(
            optimized_pulses=optimized_pulses,
            fidelity=1 - opt_result.fun,
            n_iters=opt_result.nit,
            optimizer_info=opt_result,
            history=history,
            final_state=final_state,
        )

    def _forward_pass(self, propagators: list, initial_state: Qobj) -> list:
        N = len(propagators)
        forward_evolution = [None] * (N + 1)
        forward_evolution[0] = initial_state.full().copy()
        for j, U in enumerate(propagators):
            forward_evolution[j + 1] = U @ forward_evolution[j]
        return forward_evolution

    def _backward_pass(self, propagators: list, target_state: Qobj):
        N = len(propagators)
        co_states = [None] * (N + 1)
        co_states[N] = target_state.full()
        for j in reversed(range(N)):
            co_states[j] = adj(propagators[j]) @ co_states[j + 1]
        return co_states

    def _gradient(self, forward_evolution: list, co_states: list, H_c: list, N: int, dt: float):
        """Adjoint-method gradient. Requires forward + backward passes."""
        K = len(H_c)
        H_c_arrays = [Hc.full() for Hc in H_c]
        c = (adj(co_states[-1]) @ forward_evolution[-1]).item()  # <phi|psi_N>
        # TODO: forward and co-states have to be reshaped in case of open system for matrix multiplication to work
        grads = np.zeros((K, N), dtype=float)
        for j in range(N):
            psi_j = forward_evolution[j]
            co_state_jp1 = co_states[j + 1]
            for k, Hc in enumerate(H_c_arrays):
                # TODO: for overlap, ensure the formula is right for both system type -> infer it from PM/Objective?
                inner = (adj(co_state_jp1) @ Hc @ psi_j).item()
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


if __name__ == "__main__":
    parameterization, problem = define_problem()
    solver = GRAPE(parameterization)
    result = solver.solve(problem)
    print(result)
