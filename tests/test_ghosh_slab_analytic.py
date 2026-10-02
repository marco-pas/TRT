import numpy as np

from trt_jfnk.benchmarks.ghosh_slab_analytic import (
    GhoshSlabAnalyticBenchmark,
    compute_transcendental_roots,
    ghosh_slab_steady_state,
)


def test_transcendental_roots():
    b = 1.0
    n_roots = 30
    roots = compute_transcendental_roots(b=b, n_roots=n_roots)

    assert len(roots) == n_roots
    # Roots must be strictly increasing and positive
    assert np.all(np.diff(roots) > 0.0)
    assert roots[0] > 0.0

    # First few roots match Ghosh (2014) Figure 2
    assert np.isclose(roots[0], 1.22826, rtol=1e-4)
    assert np.isclose(roots[1], 3.61221, rtol=1e-4)
    assert np.isclose(roots[2], 6.54624, rtol=1e-4)

    # Each root must satisfy the transcendental equation within numerical precision
    residual = (4.0 * roots**2 - 3.0) * np.sin(roots * b) - 4.0 * np.sqrt(3.0) * roots * np.cos(roots * b)
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_steady_state_and_boundary_conditions():
    b = 1.0
    x = np.linspace(0.0, b, 50)
    u_inf = ghosh_slab_steady_state(x, b=b)

    # Check theoretical values at boundaries
    expected_u0 = (3.0 * b + 2.0 * np.sqrt(3.0)) / (3.0 * b + 4.0 * np.sqrt(3.0))
    expected_ub = (2.0 * np.sqrt(3.0)) / (3.0 * b + 4.0 * np.sqrt(3.0))
    assert np.isclose(u_inf[0], expected_u0, rtol=1e-12)
    assert np.isclose(u_inf[-1], expected_ub, rtol=1e-12)

    # Check that Robin BCs are analytically satisfied
    # u(0) - 2/sqrt(3) * u'(0) = 1
    # u(b) + 2/sqrt(3) * u'(b) = 0
    slope = -3.0 / (3.0 * b + 4.0 * np.sqrt(3.0))
    bc_left = expected_u0 - (2.0 / np.sqrt(3.0)) * slope
    bc_right = expected_ub + (2.0 / np.sqrt(3.0)) * slope

    assert np.isclose(bc_left, 1.0, atol=1e-14)
    assert np.isclose(bc_right, 0.0, atol=1e-14)


def test_analytical_solution_profiles_match_ghosh_paper():
    benchmark = GhoshSlabAnalyticBenchmark(b=1.0, epsilon=0.1, n_roots=35)
    x = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

    # 1. At tau = 0, both radiation and material energy must be zero (within series truncation error)
    u_0, v_0 = benchmark.evaluate(x, tau=0.0)
    np.testing.assert_allclose(u_0, 0.0, atol=1e-2)
    np.testing.assert_allclose(v_0, 0.0, atol=1e-2)

    # 2. At early time tau = 0.01:
    # Radiation front is penetrating from left (u(0) ~ 0.24, u(1) ~ 0.005)
    # Material is still cold (v(0) ~ 0.006)
    u_early, v_early = benchmark.evaluate(x, tau=0.01)
    assert 0.23 < u_early[0] < 0.25
    assert u_early[-1] < 0.01
    assert v_early[0] < 0.01  # Material lags behind radiation
    assert np.all(u_early >= v_early - 1e-4)

    # 3. At intermediate time tau = 0.10:
    u_mid, v_mid = benchmark.evaluate(x, tau=0.1)
    assert 0.44 < u_mid[0] < 0.46
    assert 0.03 < v_mid[0] < 0.05

    # 4. At tau = 1.0:
    u_1, v_1 = benchmark.evaluate(x, tau=1.0)
    assert 0.53 < u_1[0] < 0.56
    assert 0.30 < v_1[0] < 0.34

    # 5. At late time tau = 10.0:
    # Near equilibrium (u ~ v) and close to linear steady state
    u_late, v_late = benchmark.evaluate(x, tau=10.0)
    u_inf = benchmark.steady_state(x)
    np.testing.assert_allclose(u_late, v_late, atol=2e-3)
    np.testing.assert_allclose(u_late, u_inf, atol=2e-3)


def test_timeseries_evaluation():
    benchmark = GhoshSlabAnalyticBenchmark(b=1.0, epsilon=0.1, n_roots=30)
    x = np.linspace(0.0, 1.0, 11)
    times = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    u_grid, v_grid = benchmark.evaluate_timeseries(x, times)
    assert u_grid.shape == (len(times), len(x))
    assert v_grid.shape == (len(times), len(x))

    # At any spatial location inside the slab, energy density must increase monotonically in time
    for j in range(len(x)):
        assert np.all(np.diff(u_grid[:, j]) > -1e-5)
        assert np.all(np.diff(v_grid[:, j]) > -1e-5)
