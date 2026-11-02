"""Shared helpers for the user-facing ``__repr__`` of qoc objects.

Summaries describe their operators and states, they never print the data:
a summary of an n-level system or a two-qubit gate has to stay readable, and
``repr(system.drift)`` or ``repr(objective.target)`` still gives the matrix.
"""

from qutip import Qobj


def describe_qobj(q: Qobj | None) -> str:
    """One-line descriptor of a Qobj: what it is and what space it lives on.

    ``q.type`` covers states, operators, superoperators and encoded vectors
    alike ('ket', 'oper', 'super', 'operator-ket'), so one descriptor serves
    both systems and objectives.
    """
    if q is None:
        return "none"
    return f"{q.type}, dims={q.dims}"


def describe_qobj_list(items: list, empty: str = "none") -> str:
    """Descriptor for a list of Qobjs: how many, and what they look like.

    Entries of such a list (control Hamiltonians, collapse operators) share
    a space in practice, so the first one stands for all of them.
    """
    if not items:
        return empty
    return f"{len(items)} x {describe_qobj(items[0])}"


def format_summary(
    title: str, rows: list[tuple[str, str]], note: str | None = None
) -> str:
    """Render a titled block of aligned ``label : value`` rows.

    Parameters
    ----------
    title : str
        Heading, normally the class name.
    rows : list[tuple[str, str]]
        Label/value pairs, in display order.
    note : str | None, optional
        Single line placed between the title and the rows, e.g. a system's
        equation of motion.
    """
    lines = [title]
    if note:
        lines.append(f"  {note}")
    if rows:
        width = max(len(label) for label, _ in rows)
        lines += [f"  {label:<{width}} : {value}" for label, value in rows]
    return "\n".join(lines)
