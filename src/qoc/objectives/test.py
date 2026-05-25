import numpy as np

from qutip import basis, identity, sigmax, sigmaz, ket2dm

from qoc.objectives.state_transfer import StateTransfer
from qoc.performance.state_fidelity import StateFidelity

from qoc.objectives.gate_synthesis import GateSynthesis
from qoc.performance.gate_fidelity import GateFidelity

from qoc.systems.closed import ClosedSystem

H0 = 0 * sigmaz()
H_c = sigmax() / 2

system = ClosedSystem(H0=H0, H_controls=[H_c])

times = np.linspace(0, 1, 100)
pi_pulse = np.full((1, 100), np.pi)


# State transfer ojective: pure states


objective = StateTransfer(initial=basis(2, 0), target=basis(2, 1), performance_measure=StateFidelity())

final = system.evolve(objective.initial, pi_pulse, times)

print(objective.compute(final))
