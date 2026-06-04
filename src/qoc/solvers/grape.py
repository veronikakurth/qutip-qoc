from .base import OCPSolver
from qoc.optimizers.base import Optimizer
from qoc.problem import OptimalControlProblem
from qoc.result import Result

class GRAPE(OCPSolver):

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def solve(self, problem: OptimalControlProblem) -> Result:
        """Main entry point."""
        pass

    def grape_gradient(forward_evolution: list, co_states: list, H_c: list, N: int, dt: float):
        # adjoint method, requires previously computed forward and backward pass
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
