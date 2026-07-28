# Give possibility to also define custom dynamics in addition to choosing a type
# For example, introduce a new argument, "system_model" which is None by default, but if provided with a correct type (System), it defines the dynamics of the controlled system
from .closed import ClosedSystem
from .open import OpenSystem
from qutip import Qobj
from typing import Literal
import inspect


# Facade class
# TODO: add more properties that would mirror the interface of System class
class ControlledSystem:
    # TODO: the constructor is not user-friendly at the moment: no type hinting for system parameters. Shall we at least add controllable/non-controllable part of dynamics
    # TODO: how to give hints for 'kind'?
    def __init__(
        self,
        H0: Qobj,
        H_controls: list[Qobj],
        kind: Literal["closed", "open"],
        **system_params,
    ):
        self.dynamics = SystemFactory.create(
            kind, {"H0": H0, "H_controls": H_controls} | system_params
        )


# Factory class: decides on exact implementation of System based on "kind" parameter
# Alternatively, it could also decide based on passed parameters
class SystemFactory:

    registry = {"closed": ClosedSystem, "open": OpenSystem}

    @staticmethod
    def create(kind: Literal["closed", "open"], params: dict):
        # Return instantiated System object
        cls = SystemFactory.registry.get(kind)

        if cls is None:
            valid = ", ".join(repr(c) for c in SystemFactory.registry)
            raise ValueError(
                f"Unknown system kind {kind!r}. Expected one of: \n" f"{valid}"
            )
        sig = inspect.signature(cls.__init__)
        accepted = {name for name in sig.parameters if name != "self"}
        required = {
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        missing = required - params.keys()

        if missing:
            raise TypeError(
                f"A '{kind}' system requires {sorted(missing)}, "
                f"which were not provided. \n"
                f"Parameters for a '{kind}' system: {sorted(accepted)}."
            )

        unexpected = params.keys() - accepted
        if unexpected:
            raise TypeError(
                f"A '{kind}' system does not accept {sorted(unexpected)}. "
                f"Accepted parameters: {sorted(accepted)}. "
            )

        return cls(**params)
