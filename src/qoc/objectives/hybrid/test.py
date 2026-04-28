import qutip as qt
from base import Objective, FunctionalObjective
from state_fidelity_func import state_fidelity
from gate_fidelity_func import gate_fidelity
from state_fidelity import StateFidelityObjective
from gate_fidelity import GateFidelityObjective
from qutip.qip.operations import rx

def mixture(states, probs):
    return sum(p * qt.ket2dm(s) for s, p in zip(states, probs))

# One possible way to construct Objective and compute how close the system's state is to the target
# Requires the user to provide a function that does the computation. 
# For us it might be difficult to validate it but it is flexible.
if __name__ == '__main__':

    # State transfer (pure state)
    current = qt.basis(2, 0)
    target = qt.basis(2, 1)
    # Functional approach
    objective = FunctionalObjective(target, state_fidelity) 
    fidelity_pure = objective.compute(current)
    print(f"Fidelity for pure state transfer (functional approach):\n", fidelity_pure)
    # Class-based approach
    objective = StateFidelityObjective(target)
    fidelity_pure = objective.compute(current)
    print(f"Fidelity for pure state transfer (class-based approach):\n", fidelity_pure)
    # State transfer (mixed state)
    mixed_state_target = mixture([qt.basis(2, 0), qt.basis(2, 1)], [0.6, 0.4])
    mixed_state_current = mixture([qt.basis(2, 0), qt.basis(2, 1)], [0.5, 0.5])

    objective_mixed = FunctionalObjective(mixed_state_target, state_fidelity) 
    fidelity_mixed = objective_mixed.compute(mixed_state_current)
    print(f"Fidelity for mixed state transfer (functional approach):\n",fidelity_mixed)
    objective_mixed = StateFidelityObjective(mixed_state_target) 
    fidelity_mixed = objective_mixed.compute(mixed_state_current)
    print(f"Fidelity for mixed state transfer (class-based approach):\n",fidelity_mixed)

    # Gate synthesis
    
    # Ideal Hadamard gate (1 qubit)
    H_ideal = qt.gates.hadamard_transform(1)
    # Perturbed Hadamard: ideal + small random unitary rotation
    theta = 0.01
    U_pert = H_ideal * rx(theta)
    # Compute fidelity between two unitaries
    objective_gate = FunctionalObjective(H_ideal, gate_fidelity)
    gate_fidelity = objective_gate.compute(U_pert)
    print(f"Fidelity for gate synthesis (functional appproach):\n {gate_fidelity}")
    objective_gate = GateFidelityObjective(H_ideal)
    gate_fidelity = objective_gate.compute(U_pert)
    print(f"Fidelity for gate synthesis (class-based appproach):\n {gate_fidelity}")
