"""Unit tests for stiff multiscale MMS benchmark."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from trt_jfnk.benchmarks.mms_stiff import (
    StiffMMSConfig,
    StiffMMSProblem1D,
    SymbolicStiffTRTMMS,
)
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy, build_scale_context


def test_stiff_mms_symbolic_periodicity():
    """Verify that exact fields and symbolic source terms are strictly periodic at x=0 and x=1."""
    config = StiffMMSConfig(T_cold=10.0, T_hot=10000.0, length=1.0)
    symbolic = SymbolicStiffTRTMMS(config)

    t = 0.001
    x_left = 0.0
    x_right = 1.0

    # Periodicity of fields and manufactured sources
    assert np.isclose(symbolic.E_exact(x_left, t), symbolic.E_exact(x_right, t), atol=1e-10)
    assert np.isclose(symbolic.T_exact(x_left, t), symbolic.T_exact(x_right, t), atol=1e-10)
    assert np.isclose(symbolic.QE(x_left, t), symbolic.QE(x_right, t), atol=1e-8)
    assert np.isclose(symbolic.QT(x_left, t), symbolic.QT(x_right, t), atol=1e-8)


def test_stiff_mms_safe_eval_at_center():
    """Verify that safe evaluation at singular center point x0=0.5 returns finite non-NaN values."""
    config = StiffMMSConfig(T_cold=10.0, T_hot=10000.0, x0=0.5)
    symbolic = SymbolicStiffTRTMMS(config)

    t = 0.001
    xc = np.array([0.5])

    E_val = symbolic.E_exact(xc, t)
    T_val = symbolic.T_exact(xc, t)
    QE_val = symbolic.QE(xc, t)
    QT_val = symbolic.QT(xc, t)

    assert np.all(np.isfinite(E_val)) and not np.any(np.isnan(E_val))
    assert np.all(np.isfinite(T_val)) and not np.any(np.isnan(T_val))
    assert np.all(np.isfinite(QE_val)) and not np.any(np.isnan(QE_val))
    assert np.all(np.isfinite(QT_val)) and not np.any(np.isnan(QT_val))

    # Check peak values at t=0
    T_peak = float(symbolic.T_exact(np.array([0.5]), 0.0)[0])
    assert np.isclose(T_peak, config.T_hot, rtol=1e-5)


def test_stiff_mms_negative_diffusion_matches_jvp():
    """Verify that the sparse periodic negative diffusion matrix matches JAX JVP under stiffness."""
    problem = StiffMMSProblem1D(nx=16, config=StiffMMSConfig(T_cold=10.0, T_hot=10000.0))
    rng = np.random.RandomState(42)

    E_test = jnp.array(rng.uniform(1e4, 1e8, size=problem.nx))
    T_test = jnp.array(rng.uniform(10.0, 1000.0, size=problem.nx))
    v_test = jnp.array(rng.uniform(-1e3, 1e3, size=problem.nx))
    u_test = problem.pack(E_test, T_test)

    # JAX JVP of diffusion with respect to E
    action = jax.jvp(lambda e: problem.diffusion(e, T_test), (E_test,), (v_test,))[1]

    # Sparse matrix multiplication
    A = problem.negative_diffusion_matrix(u_test)
    expected = -A @ np.asarray(v_test)

    np.testing.assert_allclose(action, expected, rtol=1e-11, atol=1e-5)


def test_stiff_mms_schur_preconditioner_inversion():
    """Verify that Schur preconditioner solves the physical block Jacobian system exactly."""
    problem = StiffMMSProblem1D(nx=12, config=StiffMMSConfig(T_cold=10.0, T_hot=10000.0))
    u = problem.initial(0.0)
    dt = 1e-4
    context = build_scale_context(u, problem.size, ScalePolicy("state"))

    a, b, c, d = map(np.asarray, problem.local_blocks(u, dt))
    A = np.diag(a) + dt * problem.negative_diffusion_matrix(u).toarray()
    M = np.block([[A, np.diag(b)], [np.diag(c), np.diag(d)]])

    rhs = jnp.linspace(-1.0, 1.0, problem.total_size)
    prec_fn = problem.preconditioner(dt, kind="schur")(u, context)
    sol = prec_fn(rhs)

    # In scaled coordinates: context.residual_scale * rhs / context.state_scale
    scaled_rhs = np.asarray(context.residual_scale * rhs)
    unscaled_sol = np.linalg.solve(M, scaled_rhs)
    expected = jnp.asarray(unscaled_sol / np.asarray(context.state_scale))

    np.testing.assert_allclose(sol, expected, rtol=1e-8, atol=1e-10)


def test_stiff_mms_newton_convergence_and_order():
    """Verify that scale-equilibrated JFNK converges robustly and exhibits O(dx^2) refinement."""
    config = StiffMMSConfig(T_cold=10.0, T_hot=10000.0)
    dt = 1e-5
    t_final = 5e-5  # 5 time steps
    options = NewtonOptions(
        atol=1e-10,
        rtol=1e-8,
        krylov=KrylovOptions(rtol=1e-8, atol=1e-10, max_iterations=60),
    )

    errors_T = []
    resolutions = [32, 64]

    for nx in resolutions:
        problem = StiffMMSProblem1D(nx=nx, config=config)
        state = problem.initial(0.0)
        n_steps = int(round(t_final / dt))
        prec = problem.preconditioner(dt, kind="schur")

        for s in range(n_steps):
            time = (s + 1) * dt
            old_state = state
            res = newton_krylov(
                lambda u: problem.residual(u, old_state, dt, time),
                old_state,
                problem.size,
                ScalePolicy("state"),
                options,
                problem.admissible,
                prec,
            )
            assert res.converged, f"Newton failed at step {s + 1} for nx={nx}"
            # Expect rapid quadratic Newton convergence with Schur preconditioning
            assert res.iterations <= 3, f"Expected <= 3 Newton iterations, got {res.iterations}"
            state = res.state

        _, _, _, _, _, rel_T = problem.compute_errors(state, t_final)
        errors_T.append(rel_T)

    rate_T = np.log2(errors_T[0] / errors_T[1])
    # Between 32 and 64, asymptotic rate approaches 2.0
    assert rate_T > 1.70, f"Expected asymptotic spatial order > 1.70, got {rate_T:.3f}"

