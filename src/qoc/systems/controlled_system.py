# Give possibility to also define custom dynamics in addition to choosing a type
# For example, introduce a new argument, "system_model" which is None by default, but if provided with a correct type (System), it defines the dynamics of the controlled system
from .closed import ClosedSystem

# Facade class
class ControlledSystem:
    
    def __init__(self, kind="closed", **system_init_params):
        self.dynamics = SystemFactory.create(kind, system_init_params)

# Factory class: decides on exact implementation of System based on "kind" parameter
# Alternatively, it could also decide based on passed parameters
class SystemFactory:

    registry = {
        "closed": ClosedSystem,
    }

    def create(kind, params):
        # Return instantiated System object
        return SystemFactory.registry.get(kind)(**params)
