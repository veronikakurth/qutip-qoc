# Give possibility to also define custom dynamics in addition to choosing a type
# For example, introduce a new argument, "system_model" which is None by default, but if provided with a correct type (System), it defines the dynamics of the controlled system
from .closed import ClosedSystem
from .open import OpenSystem
from qutip import Qobj

# Facade class
# This design makes a strong assumption on two main classes of system existing, namely, open and closed.
# Open questions: can we support also more custom cases when OpenSystem is not enough? In that case, we probably want a new subclass be a subclass of OpenSystem for the logic to work
class ControlledSystem:
    def __init__(self, H0: Qobj, H_controls: list[Qobj], is_closed: bool, **system_params):
        self.dynamics = SystemFactory.create(is_closed, {"H0": H0, "H_controls": H_controls} | system_params)

# Currently, system factory seems to be a bit redundant for this binary system classification
class SystemFactory:

    def create(is_closed: bool, params: dict):
        # Return instantiated System object
        if is_closed:
            return ClosedSystem(**params)
        else:
            return OpenSystem(**params)
