# Give possibility to also define custom dynamics in addition to choosing a type
# For example, introduce a new argument, "system_model" which is None by default, but if provided with a correct type (System), it defines the dynamics of the controlled system
from .closed import ClosedSystem
from .open import OpenSystem
from qutip import Qobj
from typing import Literal

# Facade class
# TODO: add more properties that would mirror the interface of System class
class ControlledSystem:
    # TODO: the constructor is not user-friendly at the moment: no type hinting for system parameters. Shall we at least add controllable/non-controllable part of dynamics
    # TODO: how to give hints for 'kind'?
    def __init__(self, H0: Qobj, H_controls: list[Qobj], kind: Literal["closed", "open"], **system_params):
        self.dynamics = SystemFactory.create(kind, {"H0": H0, "H_controls": H_controls} | system_params)

# Factory class: decides on exact implementation of System based on "kind" parameter
# Alternatively, it could also decide based on passed parameters
class SystemFactory:

    registry = {
        "closed": ClosedSystem,
        "open": OpenSystem
    }

    def create(kind: Literal["closed", "open"], params: dict):
        # Return instantiated System object
        return SystemFactory.registry.get(kind)(**params)
