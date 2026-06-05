from .base import OCPSolver
from qoc.optimizers.base import Optimizer
from qoc.problem import OptimalControlProblem
from qoc.dynamics.propagator import StepPropagator
from qoc.result import Result

class GRAPE(OCPSolver):

    def __init__(self, optimizer: Optimizer, max_iter: int = 100, tol: float = 1e-8):
        self.optimizer = optimizer
        self._propagator = StepPropagator()

        self.max_iter = max_iter
        self.tol = tol


    def solve(self, problem: OptimalControlProblem) -> Result:
        """Main entry point."""
        system = problem.system
        objective = problem.objective

        times = problem.times

        u0 = problem.initial_pulses # (K, N)
        K, N = u0.shape

        def loss_and_grad(u: np.ndarray) -> tuple[float, np.ndarray]:
             
            # Compute step propagators
            Us = self._propagator.propagate()
            forward_evolution = self._forward_pass(Us, objective.initial)
            co_states = self._backward_pass(Us, objective.target) 

            loss = objective.loss(Qobj(forward_evolution[-1]))
            grad = self._gradient(forward_evolution, co_states, system.H_controls, N, dt)

            return loss, grad
        
        opt_result = self.optimizer.minimize(loss_and_grad, x0=u0, max_iter=self.max_iter, tol=self.tol)

        return Result(optimized_pulses=opt_result.x)
    
    def _forward_pass(self, propagators: list, initial_state: Qobj) -> list:
        # Returns: propagators and forward evolution
        propagators = [None] * N

        # forward_evolution = [np.eye(system.shape[0], dtype=complex)] # For gate synthesis
        # We know the size of the array in advance - pre-allocate memory
        forward_evolution = [None] * (N + 1)
        forward_evolution[0] = initial_state.full().copy()

        for j in range(N):
            # Build a full system Hamiltonian for time j (remember: controls are time dependent)
            slice_Hamiltonian = system.build_hamiltonian_time_j(control_amplitudes, j)
            # Compute a propagator for the jth time slice (corresponds to (j, j + 1) time interval)
            slice_propagator = eigendecomposition(slice_Hamiltonian, dt)
            # Compute forward evolution up to slice j

            forward_evolution[j + 1] = slice_propagator @ forward_evolution[j]

        return propagators, forward_evolution
   
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

    def _gradient(self, forward_evolution: list, co_states: list, H_c: list, N: int, dt: float):
        """
        Calculate GRAPE graidents using a so-called adjoint method. It requires previously computed forward and backward pass
        """
        grads = np.array([[None for t in range(N)] * len(H_c)])
        for j in range(N):
            for k, Hc in enumerate(H_c):
                control_operator = -1j * Hc * dt
                grads[k, j] = 2 * np.real(adj(co_states[j]) @ control_operator.full() @ forward_evolution[j - 1])
        return grads

    def run(H0, H_c, u, target_state, T, n_iter, alpha):
        # Basic case: state transfer objective + pure states
        N = u.shape[1]
        dt = T/N
        K = len(H_c)

        for i in range(n_iter):
            # 1. Do forward pass
            slice_propagators, forward_evolution = forward_pass(system, initial_state, u, N, dt)
            # 2. Fidelity
            current = forward_evolution[N]
            fidelity = StateFidelity().compute(Qobj(current), Qobj(target_state))

            if fidelity > 1 - 1e-6:
                print("Converged!")
                break
            # 3. Do backward pass
            co_states = backward_pass(slice_propagators, target_state)
            # 4. Compute gradients
            grads = grape_gradient(forward_evolution, co_states, H_c, N, dt)
            # 5. Update controls
            updated_controls = control_update(u, grads, 0.1)
            print(f"Iteration: {i}")
            print(f"previous controls: \n {control_amplitudes}")
            print(f"updated controls: \n {updated_controls}")
        return updated_controls, fidelity
