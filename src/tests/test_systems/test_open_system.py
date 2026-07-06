"""
Unit tests for OpenSystem driven through the Simulator (MESolveSim).

These exercise the OpenSystem.build_hamiltonian + c_ops path only. They do
NOT touch build_hamiltonian_time_j / dims / shape / c_ops validation, which
are not yet implemented on OpenSystem.

Physics conventions (verified against qutip 5.2):
  - basis(2, 1) is the excited state; |0><1| == sigmap() decays it (T1).
  - c_ops=[sqrt(gamma/2) * sigmaz] dephases coherences at rate gamma (T2).
"""
import numpy as np
import pytest
from qutip import sigmax, sigmaz, basis, ket2dm

from qoc.systems.open import OpenSystem
from qoc.systems.closed import ClosedSystem
from qoc.dynamics.simulator import MESolveSim, SESolveSim


@pytest.fixture
def times():
    return np.linspace(0, 5, 50)


@pytest.fixture
def sigma_minus():
    """Lowering operator |0><1| that decays the excited state basis(2, 1)."""
    return basis(2, 0) * basis(2, 1).dag()


class TestClosedLimit:
    """With no dissipation, open evolution must reduce to closed evolution."""

    def test_no_dissipation_matches_closed(self, times):
        H0, Hc = sigmaz() / 2, sigmax() / 2
        u = np.full((1, len(times)), 1.0)
        closed = ClosedSystem(H0, [Hc])
        open_ = OpenSystem(H0, [Hc], c_ops=[])

        ket = SESolveSim().evolve(closed, u, times, basis(2, 0)).final
        rho = MESolveSim().evolve(open_, u, times, ket2dm(basis(2, 0))).final

        assert (rho - ket2dm(ket)).norm() < 1e-5


class TestDissipation:
    """Analytical open-system solutions with known closed forms."""

    def test_t1_decay(self, times, sigma_minus):
        gamma = 0.5
        system = OpenSystem(0 * sigmaz(), [sigmax() / 2], c_ops=[np.sqrt(gamma) * sigma_minus])
        u = np.zeros((1, len(times)))

        res = MESolveSim().evolve(system, u, times, ket2dm(basis(2, 1)), return_trajectory=True)
        pops = [rho[1, 1].real for rho in res.trajectory]

        assert np.allclose(pops, np.exp(-gamma * times), atol=1e-3)

    def test_t2_dephasing(self, times):
        gamma = 0.5
        system = OpenSystem(0 * sigmaz(), [sigmax() / 2], c_ops=[np.sqrt(gamma / 2) * sigmaz()])
        u = np.zeros((1, len(times)))
        plus = (basis(2, 0) + basis(2, 1)).unit()

        res = MESolveSim().evolve(system, u, times, ket2dm(plus), return_trajectory=True)
        coherence = [abs(rho[0, 1]) for rho in res.trajectory]

        # rho_01(0) = 0.5, decays as exp(-gamma t)
        assert np.allclose(coherence, 0.5 * np.exp(-gamma * times), atol=1e-3)

    def test_dephasing_preserves_populations(self, times):
        gamma = 0.5
        system = OpenSystem(0 * sigmaz(), [sigmax() / 2], c_ops=[np.sqrt(gamma / 2) * sigmaz()])
        u = np.zeros((1, len(times)))
        plus = (basis(2, 0) + basis(2, 1)).unit()

        res = MESolveSim().evolve(system, u, times, ket2dm(plus), return_trajectory=True)
        pops = [rho[1, 1].real for rho in res.trajectory]

        assert np.allclose(pops, 0.5, atol=1e-3)


class TestPhysicalValidity:
    """CPTP sanity: the density matrix stays a valid physical state."""

    @pytest.fixture
    def trajectory(self, times, sigma_minus):
        gamma = 0.5
        system = OpenSystem(sigmaz() / 2, [sigmax() / 2], c_ops=[np.sqrt(gamma) * sigma_minus])
        u = np.full((1, len(times)), 1.0)
        return MESolveSim().evolve(
            system, u, times, ket2dm(basis(2, 1)), return_trajectory=True
        ).trajectory

    def test_trace_preserved(self, trajectory):
        for j, rho in enumerate(trajectory):
            assert abs(rho.tr() - 1) < 1e-6, f"trace not preserved at step {j}"

    def test_hermitian(self, trajectory):
        for j, rho in enumerate(trajectory):
            assert (rho - rho.dag()).norm() < 1e-9, f"rho not Hermitian at step {j}"

    def test_positive_semidefinite(self, trajectory):
        for j, rho in enumerate(trajectory):
            assert rho.eigenenergies().min() > -1e-9, f"rho not PSD at step {j}"


class TestEvolutionResultContract:
    """The EvolutionResult returned for an open system is well-formed."""

    def test_final_is_operator(self, times):
        system = OpenSystem(sigmaz() / 2, [sigmax() / 2], c_ops=[])
        u = np.zeros((1, len(times)))
        res = MESolveSim().evolve(system, u, times, ket2dm(basis(2, 0)))
        assert res.final.isoper

    def test_trajectory_matches_times_when_requested(self, times):
        system = OpenSystem(sigmaz() / 2, [sigmax() / 2], c_ops=[])
        u = np.zeros((1, len(times)))
        res = MESolveSim().evolve(
            system, u, times, ket2dm(basis(2, 0)), return_trajectory=True
        )
        assert len(res.trajectory) == len(times)
        assert res.trajectory[-1] == res.final

    def test_no_trajectory_by_default(self, times):
        system = OpenSystem(sigmaz() / 2, [sigmax() / 2], c_ops=[])
        u = np.zeros((1, len(times)))
        res = MESolveSim().evolve(system, u, times, ket2dm(basis(2, 0)))
        assert res.trajectory is None
