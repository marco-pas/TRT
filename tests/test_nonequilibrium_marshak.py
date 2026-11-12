"""Unit tests for 1D Two-Temperature Non-Equilibrium Marshak wave problem."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from trt_jfnk.benchmarks.nonequilibrium_marshak import (
    NonequilibriumMarshakConfig,
    NonequilibriumMarshakProblem1D,
)
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy, build_scale_context


def test_nonequilibrium_marshak_initial_and_boundary():
    """Verify initial cold equilibrium state and incoming boundary flux."""
    problem = NonequilibriumMarshakProblem1D(nx=16)
    u0 = problem.initial()
    E0, T0 = problem.unpack(u0)

    # All cells should start at cold equilibrium: T = 0.2, E = 0.2^4 = 0.0016
    np.testing.assert_allclose(T0, 0.2, rtol=1e-14)
    np.testing.assert_allclose(E0, 0.2**4, rtol=1e-14)

    # Boundary incoming flux at t=0 should be positive (heating domain)
    fx = problem.face_fluxes(E0, T0)
    assert float(fx[0]) > 0.0, f"Expected incoming flux > 0, got {float(fx[0])}"
    assert float(fx[-1]) == 0.0, f"Expected reflecting outer flux == 0, got {float(fx[-1])}"


def test_nonequilibrium_marshak_schur_preconditioner_inversion():
    """Verify that Schur complement preconditioner inverts the physical block Jacobian."""
    problem = NonequilibriumMarshakProblem1D(nx=16)
    u = problem.initial()
    dt = 0.001
    context = build_scale_context(u, problem.size, ScalePolicy("state"))

    a, b, c, d = map(np.asarray, problem.local_blocks(u, dt))
    A = np.diag(a) + dt * problem.negative_diffusion_matrix(u).toarray()
    M = np.block([[A, np.diag(b)], [np.diag(c), np.diag(d)]])

    rhs = jnp.linspace(-1.0, 1.0, problem.total_size)
    prec_fn = problem.preconditioner(dt, kind="schur")(u, context)
    sol = prec_fn(rhs)

    scaled_rhs = np.asarray(context.residual_scale * rhs)
    unscaled_sol = np.linalg.solve(M, scaled_rhs)
    expected = jnp.asarray(unscaled_sol / np.asarray(context.state_scale))

    np.testing.assert_allclose(sol, expected, rtol=1e-10, atol=1e-10)


def test_nonequilibrium_marshak_newton_convergence():
    """Verify that JFNK with Schur preconditioner converges in <= 4 Newton iterations."""
    problem = NonequilibriumMarshakProblem1D(nx=32)
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
            ScalePolicy("state", (problem.config.cold**4, problem.config.cold)),
            options,
            problem.admissible,
            prec,
        )
        assert res.converged, f"Newton failed at step {s + 1}"
        assert res.iterations <= 4, f"Expected <= 4 Newton iters, got {res.iterations}"
        state = res.state

