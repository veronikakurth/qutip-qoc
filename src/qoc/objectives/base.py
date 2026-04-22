from qutip import Qobj

from qoc.fom.base import FigureOfMerit

# Arguments for being explicit w.r.t. objective (it may be hard to infer it from initial and target in case of state transfer for open systems (dynamiics described by density matrix) and GateSynthesis for any system (described by unitary)

# Here, it feels natural to talk about fidelity (or another measure of closeness to the target) and make it a part of Objective. In itself, the measure may involve target, so passing a target can be optional. 

# Class for validation
class Objective:
    """
    Describes what the optimisation should achieve. Possible objectives: state transfer and gate synthesis.

    Holds the initial condition, the target, and the measure that
    scores how close the propagated result is to the target.

    The concrete type (StateTransfer vs GateSynthesis) determines what
    initial/target must be - validation lives in the subclasses.
    """

    def __init__(self, initial: Qobj, figure_of_merit: FigureOfMerit, target: Qobj = None):
        self.initial = initial
        self.target = target

        if not isinstance(figure_of_merit, FigureOfMerit):
            raise TypeError()

        self.figure_of_merit = figure_of_merit

    def _infer_target_from_figure_of_merit(self, figure_of_merit: FigureOfMerit):
        try:
            target = figure_of_merit.get_target() # TODO: get target could be abstract
        except NotImplementedError as exc:
            raise ValueErorr(f"Failed to infer target from figure of merit ({str(figure_of_merit)}). \n Not possible to proceed with State Transfer Objective. Either provide target explicitly to your problem or fix your custom figure of merit.")
        return target
