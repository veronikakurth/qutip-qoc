import numpy as np
from qutip import Qobj, basis, sigmax, sigmaz

from qoc.solvers.base import OCPSolver
from qoc.systems.base import System
from qoc.systems.controlled_system import ControlledSystem
from qoc.optimizers.base import Optimizer, ScipyLBFGS
from qoc.problem import OptimalControlProblem
from qoc.dynamics.propagator import StepPropagator
from qoc.result import Result
from qoc.objectives.state_transfer import StateTransfer

def adj(A):
    return np.conj(A).T

class GRAPE(OCPSolver):

    def __init__(self, optimizer: Optimizer, max_iter: int = 100, tol: float = 1e-8):
        self.optimizer = optimizer
        self._propagator = StepPropagator()

        self.max_iter = max_iter
        self.tol = tol


    def solve(self, problem: OptimalControlProblem) -> Result:
        """Main entry point."""
        system = problem.system.dynamics
        objective = problem.objective

        times = problem.times

        dt = times[-1]/len(times)

        u0 = problem.initial_pulses # (K, N)
        K, N = u0.shape

        def loss_and_grad(u: np.ndarray) -> tuple[float, np.ndarray]:
             
            # Compute step propagators
            Us = self._propagator.propagate(system, N, u.reshape(K, N), dt)
            forward_evolution = self._forward_pass(Us, objective.initial)
            co_states = self._backward_pass(Us, objective.target) 

            loss = float(objective.loss(Qobj(forward_evolution[-1])))
            grads = self._gradient(forward_evolution, co_states, system.H_controls, N, dt)

            return loss, grads.ravel()
        
        opt_result = self.optimizer.minimize(loss_and_grad, x0=u0, max_iter=self.max_iter, tol=self.tol)

        #return Result(optimized_pulses=opt_result.x)
        return opt_result
    
    def _forward_pass(self, propagators: list, initial_state: Qobj) -> list:
        # GRAPE's forward pass to compute forward evolution

        # forward_evolution = [np.eye(system.shape[0], dtype=complex)] # For gate synthesis
        # We know the size of the array in advance - pre-allocate memory
        N = len(propagators)
        forward_evolution = [None] * (N + 1)
        forward_evolution[0] = initial_state.full().copy()

        for j, state in enumerate(propagators):
            # Compute forward evolution up to slice j
            forward_evolution[j + 1] = state @ forward_evolution[j]

        return forward_evolution
   
    def _backward_pass(self, propagators: list, target_state: Qobj):
        # Propagators must be ordered from U_1 to U_N
        N = len(propagators)
        co_states = [None] * (N + 1)
        co_states[N] = target_state.full()

        for j in range(N):
            sub_co_state = co_states[-1]
            for i in reversed(range(N, j)):
                sub_co_state = adj(propagators[i]) @ sub_co_state
            co_states[j] = sub_co_state
        return co_states
   
    # TODO: make it compliant with Scipy's interface on user-passed functions
    def _gradient(self, forward_evolution: list, co_states: list, H_c: list, N: int, dt: float):
        """
        Calculate GRAPE gradients using a so-called adjoint method. It requires previously computed forward and backward pass
        """
        K = len(H_c)
        H_c_arrays = [Hc.full() for Hc in H_c]
        
        c = (adj(co_states[-1]) @ forward_evolution[-1]).item() # <phi|psi_N>

        grads = np.zeros((K, N), dtype=float)

        for j in range(N):
            psi_j = forward_evolution[j]
            co_state_jp1 = co_states[j + 1]
            for k, Hc in enumerate(H_c_arrays):
                inner = (adj(co_state_jp1) @ Hc @ psi_j).item()
                grads[k, j] = -2.0 * dt * np.imag(np.conj(c) * inner)
        return grads

def define_problem():
    # Helper function: Collect problem definition into OCPProblem container
    
    # Model parameters
    H0 = 0 * sigmaz()
    H_c = [sigmax() / 2, sigmax() / 2]
    system = ControlledSystem('closed', H0=H0, H_controls=H_c)
    # Simulation parameters
    T = 10
    N = 10
    times = np.linspace(0, T, N)
    # TODO: calculating multiple dts would allow for non-uniform discretization
    print(f"np.diff(times): {np.diff(times)}")
    # Initial conditions and target 
    control_amplitudes = np.array([[np.pi for t in times] for i in range(system.dynamics.n_controls)])
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    # Choose performance measure and aggregate it with initial conditions and target into a objective -> verify the objective is consistent
    objective = StateTransfer(initial_state, target_state)
    problem = OptimalControlProblem(system, objective, times, control_amplitudes)
    return problem

if __name__ == "__main__":
    problem = define_problem()
    solver = GRAPE(optimizer=ScipyLBFGS())
    result = solver.solve(problem) # TODO: to pass a universal "problem" object, it might make sense to split its arguments in different types of parameters so that its signature definition would not blow up.
    print(result)
