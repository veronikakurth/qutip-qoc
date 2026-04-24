from qutip import basis
from base import Objective
from state_transfer import state_transfer_fidelity

# One possible way to construct Objective and compute how close the system's state is to the target
# Requires the user to provide a function that does the computation. 
# For us it might be difficult to validate it but it is flexible.
if __name__ == '__main__':
    current = basis(2, 0)
    target = basis(2, 1)
    objective = Objective(target, state_transfer_fidelity) 
    fidelity = objective.compute(current)
    print(fidelity)
