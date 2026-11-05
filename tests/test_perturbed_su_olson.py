"""Unit tests for Perturbed Su-Olson non-linear benchmark problem."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from trt_jfnk.benchmarks.perturbed_su_olson import (
    PerturbedSuOlsonConfig,
    PerturbedSuOlsonProblem1D,
)
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy, build_scale_context


def test_perturbed_su_olson_config_consistency():
    """Verify that delta=0 recovers cubic Debye capacity, and delta=0.01 matches Taylor expansion."""
    config_base = PerturbedSuOlsonConfig(delta=0.0, cv0=0.0, alpha=4.0)
    config_pert = PerturbedSuOlsonConfig(delta=0.01, cv0=0.0, alpha=4.0)

    T_test = np.array([0.8, 1.0, 1.2, 1.5])
    # At delta = 0, e(T) = T^4 exactly
    np.testing.assert_allclose(config_base.internal_energy(T_test), T_test**4, rtol=1e-12)

    # Taylor expansion: e_delta(T) - e_0(T) = delta * T^4 * (ln(T) - 1/4) + O(delta^2)
    diff_exact = config_pert.internal_energy(T_test) - config_base.internal_energy(T_test)
    diff_taylor = 0.01 * (T_test**4) * (np.log(T_test) - 0.25)
    np.testing.assert_allclose(diff_exact, diff_taylor, rtol=2e-2)


def test_perturbed_su_olson_negative_diffusion_matches_jvp():
    """Verify that discrete periodic diffusion matches JAX JVP."""
    problem = PerturbedSuOlsonProblem1D(nx=16)
    rng = np.random.RandomState(42)

    E_test = jnp.array(rng.uniform(0.1, 1.0, size=problem.nx))
    v_test = jnp.array(rng.uniform(-0.1, 0.1, size=problem.nx))

    action = jax.jvp(problem.diffusion, (E_test,), (v_test,))[1]
    A = problem.negative_diffusion_matrix()
    expected = -A @ np.asarray(v_test)

    np.testing.assert_allclose(action, expected, rtol=1e-12, atol=1e-12)


def test_perturbed_su_olson_schur_preconditioner_inversion():
    """Verify that Schur complement preconditioner inverts the physical block Jacobian."""
    problem = PerturbedSuOlsonProblem1D(nx=16)
    u = problem.initial()
    dt = 0.001
    context = build_scale_context(u, problem.size, ScalePolicy("state"))

    a, b, c, d = map(np.asarray, problem.local_blocks(u, dt))
    a = np.broadcast_to(a, (problem.nx,))
    c = np.broadcast_to(c, (problem.nx,))
    A = np.diag(a) + dt * problem.negative_diffusion_matrix().toarray()
    M = np.block([[A, np.diag(b)], [np.diag(c), np.diag(d)]])

    rhs = jnp.linspace(-1.0, 1.0, problem.total_size)
    prec_fn = problem.preconditioner(dt, kind="schur")(u, context)
    sol = prec_fn(rhs)

    scaled_rhs = np.asarray(context.residual_scale * rhs)
    unscaled_sol = np.linalg.solve(M, scaled_rhs)
    expected = jnp.asarray(unscaled_sol / np.asarray(context.state_scale))

    np.testing.assert_allclose(sol, expected, rtol=1e-10, atol=1e-10)


def test_perturbed_su_olson_newton_convergence():
    """Verify that JFNK with Schur preconditioner converges in <= 2 Newton iterations."""
    problem = PerturbedSuOlsonProblem1D(nx=32, config=PerturbedSuOlsonConfig(delta=0.01))
    dt = 0.001
    options = NewtonOptions(
        atol=1e-10,
        rtol=1e-8,
        krylov=KrylovOptions(rtol=1e-8, atol=1e-10, max_iterations=60),
    )

    state = problem.initial()
    prec = problem.preconditioner(dt, kind="schur")

    for s in range(5):
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
        assert res.converged, f"Newton failed at step {s + 1}"
        assert res.iterations <= 2, f"Expected <= 2 Newton iters, got {res.iterations}"
        state = res.state
