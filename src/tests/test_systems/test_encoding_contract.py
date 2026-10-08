"""The System encoding contract (ADR 0004, *The encoding contract*).

Algorithms seed the adjoint pass with ``encode(G)`` for a measure's physical
gradient G. That is exact only if encode/decode are mutually inverse, linear and
inner-product preserving, so those are checked here, including on the
non-Hermitian, unnormalized objects a gradient can be.
"""
import numpy as np
import pytest
from qutip import Qobj, rand_dm, rand_ket, rand_unitary, sigmam, sigmax, sigmaz

from qoc.systems.controlled_system import ControlledSystem


def _inner(a: Qobj, b: Qobj) -> complex:
    """Hilbert-Schmidt inner product tr(a^dag b); a plain overlap for vectors."""
    product = a.dag() @ b
    return complex(product.tr() if isinstance(product, Qobj) else product)


def _random_operator(rng, n: int) -> Qobj:
    """Arbitrary complex operator: neither Hermitian nor normalized, like a gradient."""
    return Qobj(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))


def _random_ket(rng, n: int) -> Qobj:
    """Arbitrary complex vector, not normalized."""
    return Qobj(rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1)))


@pytest.fixture
def closed():
    return ControlledSystem.closed(H0=sigmaz(), H_controls=[sigmax()])


@pytest.fixture
def open_():
    return ControlledSystem.open(H0=sigmaz(), H_controls=[sigmax()], c_ops=[sigmam()])


class TestClosedStateEncoding:

    def test_round_trip(self, closed):
        psi = rand_ket(2)
        assert closed.decode_state(closed.encode_state(psi)) == psi

    def test_preserves_inner_product(self, closed):
        rng = np.random.default_rng(0)
        x, y = _random_ket(rng, 2), _random_ket(rng, 2)
        assert np.isclose(_inner(closed.encode_state(x), closed.encode_state(y)), _inner(x, y))

    def test_linear(self, closed):
        rng = np.random.default_rng(1)
        x, y = _random_ket(rng, 2), _random_ket(rng, 2)
        a, b = 0.3 - 1.2j, 2.0 + 0.5j
        assert closed.encode_state(a * x + b * y) == a * closed.encode_state(x) + b * closed.encode_state(y)


class TestOpenStateEncoding:

    def test_round_trip(self, open_):
        rho = rand_dm(2)
        assert open_.decode_state(open_.encode_state(rho)) == rho

    def test_round_trip_on_non_state(self, open_):
        # Gradients are encoded too; encode must not reject non-physical operators
        G = _random_operator(np.random.default_rng(2), 2)
        assert open_.decode_state(open_.encode_state(G)) == G

    def test_preserves_inner_product(self, open_):
        rng = np.random.default_rng(3)
        A, B = _random_operator(rng, 2), _random_operator(rng, 2)
        assert np.isclose(_inner(open_.encode_state(A), open_.encode_state(B)), _inner(A, B))

    def test_linear(self, open_):
        rng = np.random.default_rng(4)
        A, B = _random_operator(rng, 2), _random_operator(rng, 2)
        a, b = 0.3 - 1.2j, 2.0 + 0.5j
        assert open_.encode_state(a * A + b * B) == a * open_.encode_state(A) + b * open_.encode_state(B)


class TestClosedOperatorEncoding:

    def test_round_trip(self, closed):
        U = rand_unitary(2)
        assert closed.decode_operator(closed.encode_operator(U)) == U

    def test_preserves_inner_product(self, closed):
        rng = np.random.default_rng(5)
        A, B = _random_operator(rng, 2), _random_operator(rng, 2)
        assert np.isclose(_inner(closed.encode_operator(A), closed.encode_operator(B)), _inner(A, B))


@pytest.mark.xfail(reason="ADR 0004: OpenSystem's operator pair is not inverse yet; "
                          "open-system gate synthesis is unsupported", strict=True)
def test_open_operator_round_trip(open_):
    U = rand_unitary(2)
    assert open_.decode_operator(open_.encode_operator(U)) == U
