import numpy as np
import torch

from src.data.synthetic import simulate_gjr
from src.data.windows import sliding_windows, stationary_bootstrap
from src.flow.cfm_model import VelocityNet, cfm_loss
from src.flow.sample import martingale_correct, sample_standardised
from src.validation import stylized_facts as sf


def test_hill_recovers_pareto_index():
    rng = np.random.default_rng(0)
    x = (rng.pareto(3.0, 200_000) + 1.0)          # tail index 3
    a, se = sf.hill_alpha(x, 0.02)
    assert abs(a - 3.0) < 4 * se, (a, se)


def test_ljung_box_calibration():
    """iid noise: sq-return LB p-values ~ U(0,1) (reject rate near nominal 5%); GARCH: reject."""
    rng = np.random.default_rng(1)
    pv = [sf.ljung_box_sq(rng.standard_normal((1, 800)), 10)[1] for _ in range(300)]
    assert 0.01 < np.mean(np.array(pv) < 0.05) < 0.11
    g = simulate_gjr(3000, 0.05, seed=2)[None, :]
    assert sf.ljung_box_sq(g, 10)[1] < 1e-6


def test_energy_test_null_and_power():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((200, 20)); B = rng.standard_normal((200, 20))
    assert sf.energy_test(A, B, n_perm=99)[1] > 0.01
    C = 1.3 * rng.standard_normal((200, 20))
    assert sf.energy_test(A, C, n_perm=99)[1] < 0.05


def test_stationary_bootstrap_preserves_marginal():
    x = simulate_gjr(2000, 0.05, seed=3)
    b = stationary_bootstrap(x, 52, 400, 20.0, np.random.default_rng(0)).ravel()
    assert sf.ks_two_sample(x, b)[1] > 0.001


def test_martingale_correction_exact():
    S = np.exp(np.cumsum(0.05 * np.random.default_rng(0).standard_normal((5000, 52)), axis=1))
    S = np.concatenate([np.ones((5000, 1)), S], axis=1)
    C = martingale_correct(S, 100.0, 0.03, 0.01, 1.0)
    t = np.linspace(0, 1, 53)
    assert np.allclose(C.mean(0), 100.0 * np.exp(0.02 * t))


def test_velocity_net_shapes_and_zero_init():
    m = VelocityNet(52, ch=32)
    x = torch.randn(8, 52)
    assert m(x, torch.rand(8)).shape == (8, 52)
    assert float(m(x, torch.rand(8)).detach().abs().max()) == 0.0      # zero-init output head


def _cfm_floor(s2, n=100001):
    """Irreducible CFM loss per coordinate for x1 ~ N(0, s2): E_t[Var(u) - Cov(u,x_t)^2/Var(x_t)]."""
    t = np.linspace(0, 1, n)
    v = (1 - t) ** 2 + t ** 2 * s2
    c = -(1 - t) + t * s2
    return float(np.mean((1 + s2) - c ** 2 / v))


def test_flow_learns_simple_gaussian():
    """Data N(0, 2^2) iid. Final loss must sit within 3% of the analytic floor and the
    sampled std within 5% of 2: the objective, network and ODE sampler agree end to end."""
    torch.manual_seed(0)
    N = 32
    m = VelocityNet(N, ch=32, dilations=(1, 2, 4, 8))
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    losses = []
    for _ in range(1000):
        loss = cfm_loss(m, 2.0 * torch.randn(256, N))
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    floor = _cfm_floor(4.0)
    assert abs(np.mean(losses[-200:]) / floor - 1) < 0.03, (np.mean(losses[-200:]), floor)
    s = sample_standardised(m.eval(), 2000, ode_steps=20, seed=0)
    assert abs(s.std() / 2.0 - 1) < 0.05, s.std()
